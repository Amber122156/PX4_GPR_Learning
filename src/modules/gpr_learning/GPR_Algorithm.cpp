#include "GPR_Algorithm.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <limits>

constexpr size_t GPR_Algorithm::MAX_HISTORY;
constexpr size_t GPR_Algorithm::MAX_FFT;
constexpr size_t GPR_Algorithm::MAX_CLUSTERS;
constexpr size_t GPR_Algorithm::MAX_HARMONICS;
constexpr size_t GPR_Algorithm::MAX_FEATURES;
constexpr size_t GPR_Algorithm::MAX_CORRELATION;
constexpr size_t GPR_Algorithm::SCRATCH_CAPACITY;

namespace
{
constexpr double PI = 3.1415926535897932384626433832795;
constexpr double TWO_PI = 2 * PI;
constexpr double EPS = std::numeric_limits<double>::epsilon();

// 周期折叠：将任意相位映射到 [0, 2π)，使同一周期位置具有统一表示。
double wrap(double x)
{
	x = std::fmod(x, TWO_PI);
	return x < 0 ? x + TWO_PI : x;
}

// 两相位的差值按最短圆周方向表示，范围为 [-π, π)。
double wrap_pi(double x) { return wrap(x + PI) - PI; }
double clamp(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }

// 顺序已由批次入口和内部数据流建立；参考时间来自已处理的样本。
double seconds(uint64_t end, uint64_t start) { return double(end - start) * 1e-6; }
bool positive(double x) { return std::isfinite(x) && x > 0; }
bool nonnegative(double x) { return std::isfinite(x) && x >= 0; }

// 中位数比均值更不易受少数异常间隔影响；排序会改写传入的工作区。
double median(double *values, size_t n)
{
	std::sort(values, values + n);
	return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) * 0.5;
}

// 每阶谐波同时用正弦和余弦表示，模型可以自由学习该阶的幅值与相位。
// 不加入常数特征，使模型专注于周期扰动，避免直接减去测量的直流偏置。
void features(double phase, size_t harmonics, double *out)
{
	for (size_t h = 0; h < harmonics; ++h) {
		out[2 * h] = std::sin((h + 1) * phase);
		out[2 * h + 1] = std::cos((h + 1) * phase);
	}
}
}

int GPR_Algorithm::configure(const Config &c)
{
	// 先检查参数关系和固定工作区容量，再接收配置
	const uint64_t m = uint64_t(c.fft_window_samples) * c.fft_padding_factor;
	const bool valid = c.fft_window_samples >= 4 && c.fft_window_samples <= MAX_HISTORY
		&& c.fft_window_samples % 2 == 0 && c.fft_padding_factor >= 1 && m <= MAX_FFT
		&& (m & (m - 1)) == 0 && c.fft_hop_samples >= 1
		&& positive(c.frequency_min_hz) && positive(c.frequency_max_hz) && c.frequency_max_hz > c.frequency_min_hz
		&& positive(c.frequency_tau_s) && positive(c.min_peak_ratio) && positive(c.max_frequency_stale_s)
		&& positive(c.memory_cycles) && c.cluster_count <= MAX_CLUSTERS
		&& c.harmonic_count >= 1 && c.harmonic_count <= MAX_HARMONICS && c.cluster_count >= 2 * c.harmonic_count + 1
		&& positive(c.kernel_scale) && positive(c.noise_floor) && nonnegative(c.jitter)
		&& c.kmeans_max_iterations > 0 && c.init_max_iterations > 0 && positive(c.kmeans_tolerance_rad)
		&& size_t(c.init_max_iterations) * c.optimizer_eval_multiplier >= size_t(c.cluster_count) + 1
		&& positive(c.gap_min_ratio) && c.gap_min_ratio < 1 && std::isfinite(c.gap_max_ratio) && c.gap_max_ratio > 1
		&& nonnegative(c.correction_gain) && positive(c.correction_limit_rad) && c.correction_limit_rad <= PI
		&& std::isfinite(c.correction_min_coefficient) && c.correction_min_coefficient >= -1 && c.correction_min_coefficient <= 1
		&& c.correlation_samples >= 4 && c.correlation_samples <= MAX_CORRELATION
		&& c.min_correction_samples >= 2 && c.min_correction_samples <= MAX_HISTORY
		&& positive(c.optimizer_tol_x) && positive(c.optimizer_tol_fun) && c.optimizer_eval_multiplier > 0
		&& c.cholesky_attempts > 0 && std::isfinite(c.jitter_growth) && c.jitter_growth > 1;
	_configured = valid;
	_config = c;
	reset();
	return valid ? 0 : -EINVAL;
}

void GPR_Algorithm::reset()
{
	++_generation;
	_head = _count = _training_count = 0;
	_samples = _last_fft_sample = _last_input_time_us = 0;
	_prediction_total = _block_prediction_total = 0;
	_last_train_cycle = UINT64_MAX;
	_have_input = false;
	_training_history = nullptr;
}

int GPR_Algorithm::validate_batch(const Sample *samples, size_t count)
{
	if (!_configured || (count && !samples)) { return -EINVAL; }
	uint64_t previous = _last_input_time_us;
	for (size_t i = 0; i < count; ++i) {
		if (!std::isfinite(samples[i].value) || ((i || _have_input) && samples[i].timestamp_sample_us <= previous)) { return -EINVAL; }
		previous = samples[i].timestamp_sample_us;
	}
	if (count) { _last_input_time_us = previous; _have_input = true; }
	return 0;
}

void GPR_Algorithm::advance_phase(const Sample &sample, const Model &model, Prediction &p) const
{
	p = {};
	p.timestamp_sample_us = sample.timestamp_sample_us;
	p.periodic = p.accumulated_phase_rad = p.phase_rad = std::numeric_limits<double>::quiet_NaN();
	if (!model.frequency_valid) { return; }
	p.accumulated_phase_rad = model.accumulated_phase_rad + TWO_PI * model.frequency_hz * seconds(sample.timestamp_sample_us, model.reference_time_us);
	p.phase_rad = p.accumulated_phase_rad + model.phase_offset_rad;
	p.phase_valid = !model.stale && seconds(sample.timestamp_sample_us, model.last_good_fft_us) <= _config.max_frequency_stale_s;
}

void GPR_Algorithm::evaluate_model(const Model &model, Prediction &p) const
{
	if (!model.valid || !p.phase_valid) { return; }
	double b[MAX_FEATURES] {};
	features(wrap(p.phase_rad), model.harmonic_count, b);
	// 当前业务只输出均值；协方差保留在模型中，供诊断或显式方差查询使用。
	p.periodic = 0;
	for (size_t i = 0; i < size_t(2 * model.harmonic_count); ++i) { p.periodic += b[i] * model.weights[i]; }
	p.valid = true;
}

GPR_Algorithm::Prediction GPR_Algorithm::process_sample(const Sample &sample, const Model &model, bool learning)
{
	Prediction prediction {};
	advance_phase(sample, model, prediction);
	evaluate_model(model, prediction);
	if (learning) { append_sample(sample, prediction); }
	return prediction;
}

void GPR_Algorithm::append_sample(const Sample &sample, const Prediction &p)
{
	auto &entry = _history[_head];
	entry.time_us = sample.timestamp_sample_us;
	entry.raw = sample.value;
	entry.accumulated_phase_rad = p.accumulated_phase_rad;
	entry.phase = wrap(p.phase_rad);
	entry.periodic = p.periodic;
	entry.phase_valid = p.phase_valid;
	entry.prediction_valid = p.valid;
	_head = (_head + 1) % MAX_HISTORY;
	_count = std::min(_count + 1, MAX_HISTORY);
	++_samples;
	if (p.valid) { ++_prediction_total; }
}

bool GPR_Algorithm::capture_update(const Model &model, UpdateRequest &r)
{
	if (!_count) { return false; }
	const auto &last = history(_count - 1);
	const bool frequency = _count >= _config.fft_window_samples
		&& (_last_fft_sample == 0 || _samples - _last_fft_sample >= _config.fft_hop_samples);
	const bool expired = model.frequency_valid && !model.stale && !last.phase_valid;
	bool train = false;
	uint64_t cycle = 0;
	if (model.frequency_valid && !model.stale && last.phase_valid) {
		cycle = uint64_t(std::floor(last.accumulated_phase_rad / TWO_PI));
		const double required = TWO_PI * (model.valid ? 1 : _config.memory_cycles);
		train = last.accumulated_phase_rad - model.block_start_accumulated_phase_rad >= required - 1e-8
			&& _count >= _config.cluster_count
			&& (!model.valid || _prediction_total - _block_prediction_total >= _config.min_correction_samples)
			&& cycle != _last_train_cycle;
	}
	if (!frequency && !train && !expired) { return false; }

	r.end_time_us = last.time_us;
	r.base_model = model;
	r.generation = _generation;
	r.frequency = frequency;
	r.train = train;
	r.predictions_at_end = _prediction_total;
	if (frequency) { _last_fft_sample = _samples; }
	if (train) { _last_train_cycle = cycle; }

	// 只复制所需历史段；保留前一个边界样本，供覆盖检查和周期相关使用。
	size_t first = train ? 0 : _count;
	if (train) {
		const double cutoff = last.accumulated_phase_rad - TWO_PI * _config.memory_cycles;
		for (size_t i = 0; i < _count; ++i) {
			const auto &h = history(i);
			const bool in_memory = h.time_us > model.memory_start_us && h.phase_valid && h.accumulated_phase_rad >= cutoff;
			const bool in_block = model.valid && h.time_us > model.block_end_us && h.prediction_valid;
			if (in_memory || in_block) { first = i ? i - 1 : 0; break; }
		}
	}
	if (frequency) { first = std::min(first, _count - _config.fft_window_samples); }
	r.count = _count - first;
	r.history_truncated = first == 0 && _samples > MAX_HISTORY;
	for (size_t i = 0; i < r.count; ++i) { r.history[i] = history(first + i); }
	return true;
}

void GPR_Algorithm::finish_update(const UpdateRequest &r, const Model &model)
{
	if (model.model_updated) { _block_prediction_total = r.predictions_at_end; }
}

double GPR_Algorithm::median_interval(const UpdateRequest &r, size_t start)
{
	for (size_t i = start + 1; i < r.count; ++i) { _scratch[i - start - 1] = seconds(r.history[i].time_us, r.history[i - 1].time_us); }
	return median(_scratch, r.count - start - 1);
}

int GPR_Algorithm::estimate_frequency(const UpdateRequest &r, Model &m)
{
	// 采样率从真实时间戳获得
	const size_t n = _config.fft_window_samples, size = n * _config.fft_padding_factor;
	const size_t start = r.count - n;
	const double fs = 1 / median_interval(r, start);
	if (_config.frequency_max_hz >= fs / 2 || _config.harmonic_count * _config.frequency_max_hz >= fs / 2) { return -ERANGE; }
	double mean = 0, window_energy = 0;
	for (size_t i = start; i < r.count; ++i) { mean += r.history[i].raw; }
	mean /= n;
	// 只对真实窗口去均值和加 Hann 窗，然后补零；补零不增加新的观测信息。
	for (size_t i = 0; i < size; ++i) {
		const double w = i < n ? 0.5 - 0.5 * std::cos(TWO_PI * i / (n - 1)) : 0;
		_fft[i] = i < n ? (r.history[start + i].raw - mean) * w : 0;
		window_energy += w * w;
	}
	// 基 2 FFT：先按位倒序排列，再逐级做蝶形运算；不支持的长度在配置阶段拒绝。
	for (size_t i = 1, j = 0; i < size; ++i) {
		size_t bit = size >> 1;
		for (; j & bit; bit >>= 1) { j ^= bit; }
		j ^= bit;
		if (i < j) { std::swap(_fft[i], _fft[j]); }
	}
	for (size_t len = 2; len <= size; len <<= 1) {
		const std::complex<double> step(std::cos(-TWO_PI / len), std::sin(-TWO_PI / len));
		for (size_t i = 0; i < size; i += len) {
			std::complex<double> w(1, 0);
			for (size_t j = 0; j < len / 2; ++j) {
				const auto u = _fft[i + j], v = w * _fft[i + j + len / 2];
				_fft[i + j] = u + v; _fft[i + j + len / 2] = u - v; w *= step;
			}
		}
	}
	size_t first = size, last = 0, peak_bin = 0, bins = 0;
	double peak = -1;
	// 构造单边功率谱并仅在指定基频范围内寻找峰值，避免选中无关频率。
	for (size_t i = 0; i <= size / 2; ++i) {
		_power[i] = std::norm(_fft[i]) / (fs * window_energy) * ((i == 0 || i == size / 2) ? 1 : 2);
		const double frequency = i * fs / size;
		if (frequency >= _config.frequency_min_hz && frequency <= _config.frequency_max_hz) {
			first = std::min(first, i); last = i; _scratch[bins++] = _power[i];
			if (_power[i] > peak) { peak = _power[i]; peak_bin = i; }
		}
	}
	if (bins < 3) { return -ERANGE; }
	m.fft_attempted = true;
	m.fft_quality = peak / std::max(median(_scratch, bins), EPS);
	// 峰值须明显高于频带背景，且不能落在搜索边界，才认为它是可信的基频候选。
	m.fft_accepted = m.fft_quality >= _config.min_peak_ratio && peak_bin > first && peak_bin < last;
	double shift = 0;
	// 用峰附近三个对数功率点作抛物线插值，细化 FFT 离散频点之间的峰位置。
	if (peak_bin > 0 && peak_bin < size / 2) {
		const double tiny = std::numeric_limits<double>::min();
		const double a = std::log(std::max(_power[peak_bin - 1], tiny));
		const double b = std::log(std::max(_power[peak_bin], tiny));
		const double c = std::log(std::max(_power[peak_bin + 1], tiny));
		const double den = a - 2 * b + c;
		if (std::abs(den) > EPS) { shift = clamp(0.5 * (a - c) / den, -0.5, 0.5); }
	}
	m.fft_frequency_hz = (peak_bin + shift) * fs / size;
	if (m.fft_accepted) {
		const bool start_memory = !m.frequency_valid || m.stale;
		// 首次建立累计相位起点；以后只低通频率，不因估频更新而重置累计相位。
		if (!m.frequency_valid) { m.frequency_hz = m.fft_frequency_hz; m.accumulated_phase_rad = m.phase_offset_rad = 0; }
		else {
			// 低通步长取距上次成功估频的真实时间，估频被拒绝时不更新此参考。
			const double a = 1 - std::exp(-seconds(r.end_time_us, m.last_good_fft_us) / _config.frequency_tau_s);
			m.frequency_hz += a * (m.fft_frequency_hz - m.frequency_hz);
		}
		if (start_memory) {
			// 只接收此参考时间之后按在线频率推进得到的相位样本，不回填此前的估频数据。
			m.memory_start_us = r.end_time_us;
			m.block_start_accumulated_phase_rad = m.accumulated_phase_rad;
		}
		m.last_good_fft_us = r.end_time_us; m.frequency_valid = true;
	}
	return 0;
}

int GPR_Algorithm::prepare_training(const UpdateRequest &r, Model &m)
{
	_training_history = r.history;
	_training_count = 0;
	const double cutoff = m.accumulated_phase_rad - TWO_PI * _config.memory_cycles;
	for (size_t i = 0; i < r.count; ++i) {
		const auto &entry = r.history[i];
		if (entry.time_us > m.memory_start_us && entry.phase_valid && entry.accumulated_phase_rad >= cutoff) {
			_training_indices[_training_count] = uint16_t(i);
			_phase_sin[_training_count] = std::sin(entry.phase);
			_phase_cos[_training_count] = std::cos(entry.phase);
			++_training_count;
		}
	}
	if (r.history_truncated && r.history[0].time_us > m.memory_start_us) {
		const auto &oldest = r.history[0];
		if (oldest.phase_valid && oldest.accumulated_phase_rad > cutoff) { return -ENOSPC; }
	}
	return _training_count >= _config.cluster_count ? 0 : -EDOM;
}

int GPR_Algorithm::update(const UpdateRequest &r, Model &candidate)
{
	// 请求由采样路径构造且独占，不再次检查输入字段、历史数值或时间顺序。
	candidate = r.base_model;
	candidate.generation = r.generation;
	candidate.fft_attempted = candidate.fft_accepted = candidate.model_updated = false;
	candidate.correction = {};
	if (candidate.frequency_valid) {
		candidate.accumulated_phase_rad += TWO_PI * candidate.frequency_hz * seconds(r.end_time_us, candidate.reference_time_us);
	}
	candidate.reference_time_us = r.end_time_us;
	if (r.frequency) { const int ret = estimate_frequency(r, candidate); if (ret != 0) { return ret; } }
	if (!candidate.frequency_valid) { return 0; }
	candidate.stale = seconds(r.end_time_us, candidate.last_good_fft_us) > _config.max_frequency_stale_s;
	if (candidate.stale) { candidate.valid = false; return 0; }
	if (!r.train) { return 0; }

	int ret = prepare_training(r, candidate);
	if (ret != 0) { return ret; }
	const bool initial = !candidate.valid;
	if (!initial) { ret = correct_phase(r, candidate); if (ret != 0) { return ret; } }
	ret = update_clusters(r); if (ret != 0) { return ret; }
	ret = compute_cluster_statistics(r); if (ret != 0) { return ret; }
	ret = fit_periodic_model(r, candidate); if (ret != 0) { return ret; }
	candidate.block_start_accumulated_phase_rad = initial ? candidate.accumulated_phase_rad : candidate.block_start_accumulated_phase_rad + TWO_PI;
	candidate.block_end_us = r.end_time_us;
	return 0;
}

int GPR_Algorithm::correct_phase(const UpdateRequest &r, Model &m)
{
	// 仅对已完成周期中的原始测量和当时实际输出的预测做相关，不能用新模型重算历史预测。
	size_t count = 0;
	for (size_t i = 0; i < r.count; ++i) {
		if (r.history[i].time_us > r.base_model.block_end_us && r.history[i].prediction_valid) { _scratch[count++] = double(i); }
	}
	if (count < _config.min_correction_samples) { return -EDOM; }
	if (r.history_truncated && r.history[0].time_us > r.base_model.block_end_us) { return -ENOSPC; }
	const auto &first = r.history[size_t(_scratch[0])], &last = r.history[size_t(_scratch[count - 1])];
	const size_t grid = _config.correlation_samples;
	double mean_a = 0, mean_b = 0;
	size_t right = 1;
	const double duration = seconds(last.time_us, first.time_us);
	// 两路数据插值到相同的等时间网格，末端不重复取样，便于进行圆周平移。
	for (size_t i = 0; i < grid; ++i) {
		const double u = duration * i / grid;
		while (right + 1 < count && seconds(r.history[size_t(_scratch[right])].time_us, first.time_us) < u) { ++right; }
		const auto &a = r.history[size_t(_scratch[right - 1])], &b = r.history[size_t(_scratch[right])];
		const double alpha = (u - seconds(a.time_us, first.time_us)) / seconds(b.time_us, a.time_us);
		_corr_raw[i] = a.raw + alpha * (b.raw - a.raw);
		_corr_pred[i] = a.periodic + alpha * (b.periodic - a.periodic);
		mean_a += _corr_raw[i]; mean_b += _corr_pred[i];
	}
	mean_a /= grid; mean_b /= grid;
	// 分别去均值并计算能量，降低直流偏置和整体幅值对相位匹配的影响。
	double norm_a = 0, norm_b = 0;
	for (size_t i = 0; i < grid; ++i) {
		_corr_raw[i] -= mean_a; _corr_pred[i] -= mean_b;
		norm_a += _corr_raw[i] * _corr_raw[i]; norm_b += _corr_pred[i] * _corr_pred[i];
	}
	const double den = std::sqrt(norm_a) * std::sqrt(norm_b);
	double coefficient = 0, best = 0;
	if (den > EPS) {
		coefficient = -std::numeric_limits<double>::infinity();
		size_t peak = 0;
		for (size_t j = 0; j < grid; ++j) {
			// 正 lag 表示预测波形需向前取值，因此应使后续学习相位前进。
			const int lag = int(j) - int(grid / 2);
			double sum = 0;
			for (size_t i = 0; i < grid; ++i) { sum += _corr_raw[i] * _corr_pred[(int(i) + lag + int(grid)) % int(grid)]; }
			_corr[j] = sum / den;
			if (_corr[j] > coefficient) { coefficient = _corr[j]; peak = j; }
		}
		double sub = 0;
		// 在相关峰附近作三点插值，将整数网格偏移细化为亚网格相位偏移。
		if (peak > 0 && peak + 1 < grid) {
			const double d = _corr[peak - 1] - 2 * _corr[peak] + _corr[peak + 1];
			if (std::abs(d) > EPS) { sub = clamp(0.5 * (_corr[peak - 1] - _corr[peak + 1]) / d, -0.5, 0.5); }
		}
		best = TWO_PI * (double(peak) - double(grid / 2) + sub) / grid;
	}
	auto &c = m.correction;
	c.candidate_rad = wrap_pi(best);
	c.limited_rad = clamp(c.candidate_rad, -_config.correction_limit_rad, _config.correction_limit_rad);
	c.allowed = _config.correction_enabled && coefficient >= _config.correction_min_coefficient;
	// 顺序为“候选量限幅 → 通过门槛 → 乘增益”；候选量和实际应用量分开记录。
	c.applied_rad = c.allowed ? _config.correction_gain * c.limited_rad : 0;
	c.offset_after_rad = m.phase_offset_rad + c.applied_rad;
	c.status = !_config.correction_enabled ? CorrectionStatus::Disabled : !c.allowed ? CorrectionStatus::LowCorrelation
		: std::fpclassify(_config.correction_gain) == FP_ZERO ? CorrectionStatus::ZeroGain
		: std::fpclassify(c.applied_rad) == FP_ZERO ? CorrectionStatus::ZeroShift : CorrectionStatus::Applied;
	m.phase_offset_rad = c.offset_after_rad;
	return 0;
}

double GPR_Algorithm::sphere_objective(const double *centers) const
{
	// 单纯形候选可以超出一周；每次评价仅规范中心，样本相位已在入库时规范。
	double normalized[MAX_CLUSTERS] {};
	for (size_t j = 0; j < _config.cluster_count; ++j) { normalized[j] = wrap(centers[j]); }
	double result = 0;
	for (size_t i = 0; i < _training_count; ++i) {
		const double phase = training(i).phase;
		double nearest = PI;
		for (size_t j = 0; j < _config.cluster_count; ++j) {
			const double distance = std::abs(phase - normalized[j]);
			nearest = std::min(nearest, std::min(distance, TWO_PI - distance));
		}
		// 保留原半角余弦目标，每个样本只计算一次余弦。
		result += 1 - std::cos(nearest / 2);
	}
	return result;
}

void GPR_Algorithm::assign_clusters(const double *centers)
{
	// 半角余弦代价随最短角距单调增加；直接比较角距，平局保留较小编号。
	for (size_t i = 0; i < _training_count; ++i) {
		const double phase = training(i).phase;
		double nearest = TWO_PI;
		for (size_t j = 0; j < _config.cluster_count; ++j) {
			const double distance = std::abs(phase - centers[j]);
			const double circular = std::min(distance, TWO_PI - distance);
			if (circular < nearest) { nearest = circular; _labels[i] = uint16_t(j); }
		}
	}
}

void GPR_Algorithm::optimize_centers()
{
	// 用 Nelder–Mead 单纯形搜索改善初始中心分布，再交给圆周聚类做局部收敛。
	// 每个顶点代表一组完整中心，目标是降低全部训练相位到最近中心的总代价。
	const size_t k = _config.cluster_count;
	const size_t budget = size_t(_config.init_max_iterations) * _config.optimizer_eval_multiplier;
	for (size_t i = 0; i <= k; ++i) {
		for (size_t j = 0; j < k; ++j) { _simplex[i][j] = TWO_PI * j / k; }
		if (i > 0) { _simplex[i][i - 1] = i == 1 ? 0.00025 : 1.05 * _simplex[i][i - 1]; }
		_objective[i] = sphere_objective(_simplex[i]);
	}
	const auto order = [&]() {
		// 稳定插入排序保证代价相同时顺序确定，同时避免额外堆内存分配。
		for (size_t i = 1; i <= k; ++i) {
			for (size_t j = i; j > 0 && _objective[j] < _objective[j - 1]; --j) {
				std::swap(_objective[j], _objective[j - 1]);
				for (size_t d = 0; d < k; ++d) { std::swap(_simplex[j][d], _simplex[j - 1][d]); }
			}
		}
	};
	order();
	size_t evaluations = k + 1;
	for (size_t iteration = 1; iteration < _config.init_max_iterations
	     && evaluations < budget; ++iteration) {
		double spread_x = 0, spread_f = 0, max_x = _simplex[0][0];
		for (size_t j = 1; j < k; ++j) { max_x = std::max(max_x, _simplex[0][j]); }
		for (size_t i = 1; i <= k; ++i) {
			spread_f = std::max(spread_f, std::abs(_objective[i] - _objective[0]));
			for (size_t j = 0; j < k; ++j) { spread_x = std::max(spread_x, std::abs(_simplex[i][j] - _simplex[0][j])); }
		}
		const double eps_x = std::nextafter(std::abs(max_x), INFINITY) - std::abs(max_x);
		const double eps_f = std::nextafter(std::abs(_objective[0]), INFINITY) - std::abs(_objective[0]);
		if (spread_x <= std::max(_config.optimizer_tol_x, 10 * eps_x)
		    && spread_f <= std::max(_config.optimizer_tol_fun, 10 * eps_f)) { break; }
		double center[MAX_CLUSTERS] {}, reflected[MAX_CLUSTERS] {}, trial[MAX_CLUSTERS] {};
		for (size_t j = 0; j < k; ++j) {
			for (size_t i = 0; i < k; ++i) { center[j] += _simplex[i][j]; }
			center[j] /= k; reflected[j] = 2 * center[j] - _simplex[k][j];
		}
		const double fr = sphere_objective(reflected); ++evaluations;
		bool shrink = false;
		// 反射优于最好点则尝试扩张；否则视效果接受反射或执行收缩，均无改善时整体缩小。
		if (fr < _objective[0]) {
			if (evaluations < budget) {
				for (size_t j = 0; j < k; ++j) { trial[j] = 3 * center[j] - 2 * _simplex[k][j]; }
				const double fe = sphere_objective(trial); ++evaluations;
				for (size_t j = 0; j < k; ++j) { _simplex[k][j] = fe < fr ? trial[j] : reflected[j]; }
				_objective[k] = std::min(fe, fr);
			} else {
				std::copy(reflected, reflected + k, _simplex[k]); _objective[k] = fr;
			}
		} else if (fr < _objective[k - 1]) {
			std::copy(reflected, reflected + k, _simplex[k]); _objective[k] = fr;
		} else {
			const bool outside = fr < _objective[k];
			if (evaluations < budget) {
				for (size_t j = 0; j < k; ++j) { trial[j] = outside ? 1.5 * center[j] - 0.5 * _simplex[k][j] : 0.5 * (center[j] + _simplex[k][j]); }
				const double fc = sphere_objective(trial); ++evaluations;
				if (outside ? fc <= fr : fc < _objective[k]) { std::copy(trial, trial + k, _simplex[k]); _objective[k] = fc; }
				else { shrink = true; }
			} else if (outside) {
				std::copy(reflected, reflected + k, _simplex[k]); _objective[k] = fr;
			}
		}
		if (shrink) {
			// 每个新顶点评价前检查预算；未评价的顶点保持原值和原目标值。
			for (size_t i = 1; i <= k && evaluations < budget; ++i) {
				for (size_t j = 0; j < k; ++j) { trial[j] = 0.5 * (_simplex[0][j] + _simplex[i][j]); }
				const double value = sphere_objective(trial); ++evaluations;
				std::copy(trial, trial + k, _simplex[i]); _objective[i] = value;
			}
		}
		order();
	}
	for (size_t j = 0; j < k; ++j) { _centers[j] = wrap(_simplex[0][j]); }
}

void GPR_Algorithm::lloyd()
{
	// 交替进行“最近中心分配”和“圆周均值更新”，直到中心移动足够小或达到次数上限。
	for (size_t iteration = 0; iteration < _config.kmeans_max_iterations; ++iteration) {
		double old[MAX_CLUSTERS] {}, sum_sin[MAX_CLUSTERS] {}, sum_cos[MAX_CLUSTERS] {};
		std::copy(_centers, _centers + _config.cluster_count, old);
		assign_clusters(_centers);
		std::fill(_counts, _counts + _config.cluster_count, 0);
		for (size_t i = 0; i < _training_count; ++i) {
			const size_t label = _labels[i];
			sum_sin[label] += _phase_sin[i]; sum_cos[label] += _phase_cos[i]; ++_counts[label];
		}
		for (size_t j = 0; j < _config.cluster_count; ++j) {
			const size_t count = _counts[j];
			if (count) { _centers[j] = wrap(std::atan2(sum_sin[j] / count, sum_cos[j] / count)); }
			else {
				// 空簇不能提供观测统计，将其中心放到当前覆盖最差的样本处。
				double farthest = -1; size_t selected = 0;
				for (size_t i = 0; i < _training_count; ++i) {
					const double phase = training(i).phase;
					double nearest = TWO_PI;
					for (size_t c = 0; c < _config.cluster_count; ++c) {
						const double distance = std::abs(phase - _centers[c]);
						nearest = std::min(nearest, std::min(distance, TWO_PI - distance));
					}
					if (nearest > farthest) { farthest = nearest; selected = i; }
				}
				_centers[j] = training(selected).phase;
			}
		}
		double moved = 0;
		for (size_t j = 0; j < _config.cluster_count; ++j) {
			const double distance = std::abs(_centers[j] - old[j]);
			moved = std::max(moved, std::min(distance, TWO_PI - distance));
		}
		if (moved < _config.kmeans_tolerance_rad) { break; }
	}
}

int GPR_Algorithm::update_clusters(const UpdateRequest &r)
{
	const size_t k = _config.cluster_count;
	_reclustered = !r.base_model.valid;
	if (_reclustered) { optimize_centers(); lloyd(); }
	else {
		// 在线阶段先沿用已有中心分配样本，只做一次圆周均值更新，减少重复求解。
		std::copy(r.base_model.centers, r.base_model.centers + k, _centers);
		assign_clusters(_centers);
		double sum_sin[MAX_CLUSTERS] {}, sum_cos[MAX_CLUSTERS] {};
		std::fill(_counts, _counts + k, 0);
		for (size_t i = 0; i < _training_count; ++i) {
			const size_t label = _labels[i];
			sum_sin[label] += _phase_sin[i]; sum_cos[label] += _phase_cos[i]; ++_counts[label];
		}
		for (size_t j = 0; j < k; ++j) {
			const size_t count = _counts[j];
			if (count) { _centers[j] = wrap(std::atan2(sum_sin[j] / count, sum_cos[j] / count)); } else { _reclustered = true; }
		}
		double sorted[MAX_CLUSTERS] {};
		std::copy(_centers, _centers + k, sorted); std::sort(sorted, sorted + k);
		// 同时检查过密和过疏的中心间距，以及前面发现的空簇，决定是否完整重新聚类。
		for (size_t j = 0; j < k; ++j) {
			const double gap = (j + 1 < k ? sorted[j + 1] : sorted[0] + TWO_PI) - sorted[j];
			if (gap < _config.gap_min_ratio * TWO_PI / k || gap > _config.gap_max_ratio * TWO_PI / k) { _reclustered = true; }
		}
		if (_reclustered) { lloyd(); }
	}
	std::sort(_centers, _centers + k);
	assign_clusters(_centers);
	return 0;
}

int GPR_Algorithm::compute_cluster_statistics(const UpdateRequest &)
{
	const size_t k = _config.cluster_count;
	double sum_sin[MAX_CLUSTERS] {}, sum_cos[MAX_CLUSTERS] {};
	std::fill(_counts, _counts + k, 0);
	std::fill(_means, _means + k, 0);
	std::fill(_noise, _noise + k, 0);
	// 按标签累计，保持各簇内部的原样本顺序；相位三角函数由当前快照缓存提供。
	double all_mean = 0, all_variance = 0;
	for (size_t i = 0; i < _training_count; ++i) {
		const size_t label = _labels[i];
		const double raw = training(i).raw;
		++_counts[label]; sum_sin[label] += _phase_sin[i]; sum_cos[label] += _phase_cos[i];
		_means[label] += raw; all_mean += raw;
	}
	all_mean /= _training_count;
	for (size_t j = 0; j < k; ++j) {
		const size_t count = _counts[j];
		if (!count) { return -EDOM; }
		_centers[j] = wrap(std::atan2(sum_sin[j] / count, sum_cos[j] / count)); _means[j] /= count;
	}
	// 第二遍累计离均差，保留无偏样本方差与单点簇回退整体方差的语义。
	for (size_t i = 0; i < _training_count; ++i) {
		const size_t label = _labels[i];
		const double raw = training(i).raw, d = raw - _means[label], all_d = raw - all_mean;
		_noise[label] += d * d; all_variance += all_d * all_d;
	}
	all_variance /= (_training_count - 1);
	for (size_t j = 0; j < k; ++j) {
		const size_t count = _counts[j];
		_noise[j] = std::max(count > 1 ? _noise[j] / (count - 1) : all_variance, _config.noise_floor);
	}
	for (size_t i = 1; i < k; ++i) {
		for (size_t j = i; j > 0 && _centers[j] < _centers[j - 1]; --j) {
			std::swap(_centers[j], _centers[j - 1]); std::swap(_means[j], _means[j - 1]);
			std::swap(_noise[j], _noise[j - 1]); std::swap(_counts[j], _counts[j - 1]);
		}
	}
	return 0;
}

int GPR_Algorithm::fit_periodic_model(const UpdateRequest &r, Model &m)
{
	const size_t k = _config.cluster_count, d = 2 * _config.harmonic_count;
	const double scale = 2 * _config.kernel_scale;
	double common = 0;
	for (size_t i = 0; i < k; ++i) { common += _means[i]; }
	common /= k;
	double rhs[MAX_FEATURES] {};
	double jitter = _config.jitter;
	bool success = false;
	// 在特征空间求 P = I/scale + BᵀR⁻¹B，R 的对角为簇内方差加 jitter。
	// 只构造下三角；默认三阶谐波对应 6 维系统，簇数不再决定分解维数。
	for (size_t attempt = 0; attempt < _config.cholesky_attempts; ++attempt) {
		for (size_t i = 0; i < d; ++i) {
			rhs[i] = 0;
			for (size_t j = 0; j <= i; ++j) { _precision[i][j] = i == j ? 1 / scale : 0; }
		}
		for (size_t cluster = 0; cluster < k; ++cluster) {
			double b[MAX_FEATURES] {};
			features(_centers[cluster], _config.harmonic_count, b);
			const double inverse_noise = 1 / (_noise[cluster] + jitter);
			const double target = _means[cluster] - common;
			for (size_t i = 0; i < d; ++i) {
				const double weighted = b[i] * inverse_noise;
				rhs[i] += weighted * target;
				for (size_t j = 0; j <= i; ++j) { _precision[i][j] += weighted * b[j]; }
			}
		}
		success = true;
		for (size_t i = 0; i < d && success; ++i) {
			for (size_t j = 0; j <= i; ++j) {
				double a = _precision[i][j];
				for (size_t h = 0; h < j; ++h) { a -= _lower[i][h] * _lower[j][h]; }
				if (i == j) {
					if (!positive(a)) { success = false; break; }
					_lower[i][j] = std::sqrt(a);
				} else { _lower[i][j] = a / _lower[j][j]; }
			}
		}
		if (success) { break; }
		// 重试仍增大 R 中的观测对角扰动，并重新构造 P，不直接修改 P 的对角。
		jitter = std::max(_config.jitter_growth * jitter, EPS);
	}
	if (!success) { return -ERANGE; }
	// 同一次 Cholesky 分解求 Pw=q 和 PC=I，不调用一般矩阵求逆。
	const auto solve = [&](double *x) {
		for (size_t i = 0; i < d; ++i) {
			for (size_t j = 0; j < i; ++j) { x[i] -= _lower[i][j] * x[j]; }
			x[i] /= _lower[i][i];
		}
		for (size_t i = d; i-- > 0;) {
			for (size_t j = i + 1; j < d; ++j) { x[i] -= _lower[j][i] * x[j]; }
			x[i] /= _lower[i][i];
		}
	};
	std::copy(rhs, rhs + d, m.weights);
	solve(m.weights);
	for (size_t column = 0; column < d; ++column) {
		double x[MAX_FEATURES] {};
		x[column] = 1;
		solve(x);
		for (size_t i = 0; i < d; ++i) { m.covariance[i][column] = x[i]; }
	}
	// 新计算出的模型在完成处检查一次，采样路径不重复校验权重和协方差。
	for (size_t i = 0; i < d; ++i) {
		if (!std::isfinite(m.weights[i])) { return -ERANGE; }
		for (size_t j = 0; j < d; ++j) { if (!std::isfinite(m.covariance[i][j])) { return -ERANGE; } }
	}
	for (size_t i = 0; i < k; ++i) {
		m.centers[i] = _centers[i];
	}
	m.harmonic_count = _config.harmonic_count; m.cluster_count = _config.cluster_count;
	// 所有计算成功后才标记模型有效，并记录训练截止时间以供因果性检查。
	m.training_end_us = r.end_time_us; m.valid = m.model_updated = true;
	return 0;
}
