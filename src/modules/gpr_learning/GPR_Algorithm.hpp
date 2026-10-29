#pragma once

#include <cstddef>
#include <cstdint>
#include <complex>

/** 周期扰动后端。配置和外部样本只在入口校验；内部函数使用已确定的单位和顺序。
 * process_sample 先预测后记录，不生成快照。调用者取得唯一空闲学习槽后调用 capture_update。
 * update 只读取任务独占的请求；finish_update 在批次入口接收结果时更新计数边界。
 * 同时只执行一个 update；configure/reset 在学习槽空闲时调用。
 */
class GPR_Algorithm
{
public:
	static constexpr size_t MAX_HISTORY = 2048;
	static constexpr size_t MAX_FFT = 8192;
	static constexpr size_t MAX_CLUSTERS = 28;
	static constexpr size_t MAX_HARMONICS = 3;
	static constexpr size_t MAX_FEATURES = 2 * MAX_HARMONICS;
	static constexpr size_t MAX_CORRELATION = 256;
	static constexpr size_t SCRATCH_CAPACITY = MAX_HISTORY > MAX_FFT / 2 + 1 ? MAX_HISTORY : MAX_FFT / 2 + 1;

	/* 时间 s，频率 Hz，相位 rad，原始角速度 rad/s。 */
	struct Config {
		// 真实样本窗、更新步长、补零倍数；补零不延长观测时间。
		uint32_t fft_window_samples = 256;
		uint32_t fft_hop_samples = 128;
		uint32_t fft_padding_factor = 2;
		double frequency_min_hz = 8;
		double frequency_max_hz = 20;
		double frequency_tau_s = 0.25;
		double min_peak_ratio = 5;
		double max_frequency_stale_s = 0.5;

		// 记忆长度按不含修正偏置的累计相位裁剪。
		double memory_cycles = 8;
		uint16_t cluster_count = 28;
		uint16_t harmonic_count = 3;
		double kernel_scale = 1; // 核的实际系数 s = 2 * kernel_scale。
		double noise_floor = 1e-6; // 簇内样本方差下限，(rad/s)^2，不除以样本数。
		double jitter = 1e-10; // 加入各簇观测噪声 R 的对角扰动。
		uint16_t kmeans_max_iterations = 40;
		uint16_t init_max_iterations = 180;
		double kmeans_tolerance_rad = 1e-7;
		double gap_min_ratio = 0.5;
		double gap_max_ratio = 1.5;

		bool correction_enabled = true;
		double correction_gain = 1;
		double correction_limit_rad = 1.5; // 乘增益之前的候选修正量限幅。
		double correction_min_coefficient = 0.5;
		uint32_t correlation_samples = 128;
		uint32_t min_correction_samples = 16;

		double optimizer_tol_x = 1e-5;
		double optimizer_tol_fun = 1e-7;
		uint16_t optimizer_eval_multiplier = 8; // 总评价预算 = init_max_iterations * 此值。
		uint16_t cholesky_attempts = 10;
		double jitter_growth = 10;
	};

	/* 接入层转换后的机体系 Y 轴原始角速度。间隔统一由整数微秒时间戳计算。 */
	struct Sample {
		uint64_t timestamp_sample_us;
		float value;
	};

	enum class CorrectionStatus : uint8_t {
		None, Disabled, LowCorrelation, ZeroGain, ZeroShift, Applied
	};

	struct Correction {
		double candidate_rad {}, limited_rad {}, applied_rad {}, offset_after_rad {};
		bool allowed {};
		CorrectionStatus status {CorrectionStatus::None};
	};

	/* 波形模型与相位参考整体交接；一个采样批次内只读。 */
	struct Model {
		uint32_t generation {};
		uint64_t training_end_us {}, reference_time_us {}, last_good_fft_us {};
		double accumulated_phase_rad {}, phase_offset_rad {}, frequency_hz {};
		bool frequency_valid {}, valid {}, stale {};
		uint16_t harmonic_count {}, cluster_count {};
		double weights[MAX_FEATURES] {};
		double covariance[MAX_FEATURES][MAX_FEATURES] {};
		double centers[MAX_CLUSTERS] {};
		uint64_t memory_start_us {};
		double block_start_accumulated_phase_rad {};
		uint64_t block_end_us {};
		bool fft_attempted {}, fft_accepted {}, model_updated {};
		double fft_frequency_hz {}, fft_quality {};
		Correction correction {};
	};

	struct Prediction {
		uint64_t timestamp_sample_us {};
		double accumulated_phase_rad {}, phase_rad {}, periodic {};
		bool phase_valid {}, valid {};
	};

	struct HistorySample {
		uint64_t time_us {};
		double raw {}, accumulated_phase_rad {}, phase {}, periodic {};
		bool phase_valid {}, prediction_valid {};
	};

	/* 唯一学习槽的固定快照。后台结束且结果接收前，调用者不改写此对象。 */
	struct UpdateRequest {
		uint64_t end_time_us {};
		Model base_model {};
		uint32_t generation {};
		bool frequency {}, train {};
		size_t count {};
		bool history_truncated {}; // 本快照从已被环形覆盖过的最旧记录开始。
		uint64_t predictions_at_end {};
		HistorySample history[MAX_HISTORY] {};
	};

	/* 外部配置入口：校验一次并重置。返回 0 或 -EINVAL。 */
	int configure(const Config &config);
	void reset();

	/* 外部样本入口：整个批次只校验一次，包括跨批次时间单调性。
	 * 成功后，process_sample 按相同顺序处理该批次中的样本。
	 */
	int validate_batch(const Sample *samples, size_t count);

	/* 内部可信样本：先预测，learning=true 时再记录。没有重复输入检查或快照复制。 */
	Prediction process_sample(const Sample &sample, const Model &model, bool learning);

	/* 仅在唯一学习槽空闲时、批次末调用。有新事件才制作一次快照。
	 * FFT-only 只复制真实 FFT 窗口；训练复制记忆与相关所需范围。
	 * 仅处理频率过期时不复制历史。事件判断使用增量计数。
	 */
	bool capture_update(const Model &model, UpdateRequest &request);

	/* 单个后台任务处理内部生成的请求，不重新检查输入历史。
	 * -EDOM: 样本不足/空簇；-ERANGE: 频带或数值求解失败；
	 * -ENOSPC: 历史容量无法覆盖所需范围。0 不保证已经完成训练。
	 */
	int update(const UpdateRequest &request, Model &candidate);

	/* 成功结果在批次入口接收后调用，保留任务运行期间新增的预测计数。 */
	void finish_update(const UpdateRequest &request, const Model &model);

private:
	void advance_phase(const Sample &sample, const Model &model, Prediction &prediction) const;
	void evaluate_model(const Model &model, Prediction &prediction) const;
	void append_sample(const Sample &sample, const Prediction &prediction);

	int estimate_frequency(const UpdateRequest &request, Model &candidate);
	int correct_phase(const UpdateRequest &request, Model &candidate);
	int prepare_training(const UpdateRequest &request, Model &candidate);
	int update_clusters(const UpdateRequest &request);
	int compute_cluster_statistics(const UpdateRequest &request);
	int fit_periodic_model(const UpdateRequest &request, Model &candidate);
	double sphere_objective(const double *centers) const;
	void assign_clusters(const double *centers);
	void lloyd();
	void optimize_centers();
	double median_interval(const UpdateRequest &request, size_t start);

	const HistorySample &history(size_t i) const { return _history[(_head + MAX_HISTORY - _count + i) % MAX_HISTORY]; }
	const HistorySample &training(size_t i) const { return _training_history[_training_indices[i]]; }

	Config _config {};
	bool _configured {}, _have_input {};
	uint32_t _generation {};
	uint64_t _last_input_time_us {}, _samples {}, _last_fft_sample {};
	uint64_t _prediction_total {}, _block_prediction_total {}; // 总有效预测数与已处理周期末的计数。
	uint64_t _last_train_cycle {UINT64_MAX};
	size_t _head {}, _count {};
	HistorySample _history[MAX_HISTORY] {};

	// 仅学习任务使用。索引指向本次独占请求，不复制整条训练记录。
	const HistorySample *_training_history {};
	uint16_t _training_indices[MAX_HISTORY] {}, _labels[MAX_HISTORY] {};
	size_t _training_count {};
	double _phase_sin[MAX_HISTORY] {}, _phase_cos[MAX_HISTORY] {};
	double _centers[MAX_CLUSTERS] {}, _means[MAX_CLUSTERS] {}, _noise[MAX_CLUSTERS] {};
	uint32_t _counts[MAX_CLUSTERS] {};
	bool _reclustered {};
	double _precision[MAX_FEATURES][MAX_FEATURES] {}, _lower[MAX_FEATURES][MAX_FEATURES] {};
	double _simplex[MAX_CLUSTERS + 1][MAX_CLUSTERS] {}, _objective[MAX_CLUSTERS + 1] {};
	std::complex<double> _fft[MAX_FFT] {};
	double _scratch[SCRATCH_CAPACITY] {}, _power[MAX_FFT / 2 + 1] {};
	double _corr_raw[MAX_CORRELATION] {}, _corr_pred[MAX_CORRELATION] {}, _corr[MAX_CORRELATION] {};
};
