#include "../GPR_Algorithm.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{
using Algorithm = GPR_Algorithm;
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2 * PI;

void require(bool condition, const char *expression, int line)
{
	if (!condition) { throw std::runtime_error(std::string(expression) + " (line " + std::to_string(line) + ")"); }
}

#define CHECK(expression) require(bool(expression), #expression, __LINE__)

double wrap(double phase)
{
	phase = std::fmod(phase, TWO_PI);
	return phase < 0 ? phase + TWO_PI : phase;
}

struct Fixture {
	std::unique_ptr<Algorithm> algorithm {new Algorithm};
	std::unique_ptr<Algorithm::UpdateRequest> request {new Algorithm::UpdateRequest};

	explicit Fixture(const Algorithm::Config &config = Algorithm::Config{})
	{
		CHECK(algorithm->configure(config) == 0);
	}

	void cycles(double count, double frequency = 10, double rate = 1000, double shift = 0.4)
	{
		auto &r = *request;
		r = {};
		const size_t steps = size_t(std::llround(count * rate / frequency));
		CHECK(steps + 1 <= Algorithm::MAX_HISTORY);
		r.count = steps + 1;
		r.train = true;
		r.generation = 1;
		auto &base = r.base_model;
		base.reference_time_us = base.last_good_fft_us = base.block_end_us = 1000000;
		base.training_end_us = 999000;
		base.memory_start_us = 999999;
		base.frequency_hz = frequency;
		base.frequency_valid = base.valid = true;
		base.harmonic_count = Algorithm::MAX_HARMONICS;
		base.cluster_count = Algorithm::MAX_CLUSTERS;
		base.weights[0] = 1;
		for (size_t i = 0; i < Algorithm::MAX_FEATURES; ++i) { base.covariance[i][i] = 0.2; }
		for (size_t i = 0; i < Algorithm::MAX_CLUSTERS; ++i) { base.centers[i] = TWO_PI * i / Algorithm::MAX_CLUSTERS; }
		for (size_t i = 0; i <= steps; ++i) {
			auto &h = r.history[i];
			h.time_us = 1000000 + uint64_t(std::llround(i * 1e6 / rate));
			h.accumulated_phase_rad = TWO_PI * frequency * double(h.time_us - 1000000) * 1e-6;
			h.phase = wrap(h.accumulated_phase_rad);
			h.raw = std::sin(h.accumulated_phase_rad + shift);
			h.periodic = std::sin(h.accumulated_phase_rad);
			h.phase_valid = h.prediction_valid = true;
		}
		r.end_time_us = r.history[steps].time_us;
	}
};

void same_waveform(const Algorithm::Model &actual, const Algorithm::Model &expected)
{
	CHECK(actual.phase_offset_rad == expected.phase_offset_rad);
	CHECK(actual.training_end_us == expected.training_end_us);
	CHECK(actual.block_end_us == expected.block_end_us);
	CHECK(actual.block_start_accumulated_phase_rad == expected.block_start_accumulated_phase_rad);
	CHECK(actual.correction.status == Algorithm::CorrectionStatus::None);
	for (size_t i = 0; i < Algorithm::MAX_FEATURES; ++i) {
		CHECK(actual.weights[i] == expected.weights[i]);
		for (size_t j = 0; j < Algorithm::MAX_FEATURES; ++j) { CHECK(actual.covariance[i][j] == expected.covariance[i][j]); }
	}
	for (size_t i = 0; i < Algorithm::MAX_CLUSTERS; ++i) { CHECK(actual.centers[i] == expected.centers[i]); }
}

double prediction_rms(Fixture &f, const Algorithm::Model &model, double frequency, double shift)
{
	const auto &r = *f.request;
	const double phase_at_end = r.history[r.count - 1].accumulated_phase_rad;
	double error = 0;
	for (size_t i = 1; i <= 100; ++i) {
		const uint64_t elapsed_us = uint64_t(std::llround(i * 1e6 / (100 * frequency)));
		const Algorithm::Sample sample {r.end_time_us + elapsed_us, 0};
		const auto prediction = f.algorithm->process_sample(sample, model, false);
		CHECK(prediction.valid);
		const double actual = std::sin(phase_at_end + TWO_PI * frequency * elapsed_us * 1e-6 + shift);
		error += (prediction.periodic - actual) * (prediction.periodic - actual);
	}
	return std::sqrt(error / 100);
}

void phase_correction_and_fit()
{
	Fixture f;
	f.cycles(1);
	Algorithm::Model model;
	const auto result = f.algorithm->update(*f.request, model);
	CHECK(result.training_attempted && result.training_error == 0);
	CHECK(model.model_updated);
	CHECK(std::abs(model.correction.applied_rad - 0.4) < 0.003);
	const double rms = prediction_rms(f, model, 10, 0.4);
	CHECK(rms < 0.01);
	std::printf("  corrected prediction RMS: %.6f\n", rms);
}

void mixed_historical_coordinates()
{
	Fixture f;
	f.cycles(4, 10, 1000, 0.95);
	auto &r = *f.request;
	r.base_model.phase_offset_rad = 0.65;
	r.base_model.block_end_us = 1300000;
	r.base_model.block_start_accumulated_phase_rad = 6 * PI;
	for (size_t j = 0; j < Algorithm::MAX_CLUSTERS; ++j) {
		r.base_model.centers[j] = wrap(r.base_model.centers[j] + 0.65);
	}
	std::sort(r.base_model.centers, r.base_model.centers + Algorithm::MAX_CLUSTERS);
	for (size_t i = 0; i < r.count; ++i) {
		auto &h = r.history[i];
		const double old_offset = i < 150 ? -0.2 : i < 300 ? 0.2 : 0.65;
		h.phase = wrap(h.accumulated_phase_rad + old_offset);
		h.periodic = std::sin(h.accumulated_phase_rad + old_offset);
	}
	Algorithm::Model model;
	const auto result = f.algorithm->update(r, model);
	CHECK(result.training_error == 0 && model.model_updated);
	CHECK(std::abs(model.phase_offset_rad - 0.95) < 0.004);
	CHECK(prediction_rms(f, model, 10, 0.95) < 0.01);
}

void arbitrary_batch_lengths()
{
	for (double cycles : {1.0, 1.28, 2.0}) {
		Fixture f;
		f.cycles(cycles);
		Algorithm::Model model;
		const auto result = f.algorithm->update(*f.request, model);
		CHECK(result.training_error == 0 && model.model_updated);
		CHECK(std::abs(model.correction.candidate_rad - 0.4) < 0.003);
		CHECK(prediction_rms(f, model, 10, 0.4) < 0.01);
	}
}

void frequency_ramp_window()
{
	Fixture f;
	f.cycles(1.25);
	auto &r = *f.request;
	constexpr double duration = 0.125, initial_frequency = 8, final_frequency = 16;
	for (size_t i = 0; i < r.count; ++i) {
		auto &h = r.history[i];
		const double t = double(h.time_us - 1000000) * 1e-6;
		h.accumulated_phase_rad = TWO_PI * (initial_frequency * t
					 + 0.5 * (final_frequency - initial_frequency) * t * t / duration);
		h.phase = wrap(h.accumulated_phase_rad);
		h.raw = std::sin(h.accumulated_phase_rad + 0.4);
		h.periodic = std::sin(h.accumulated_phase_rad);
	}
	r.base_model.reference_time_us = r.end_time_us;
	r.base_model.last_good_fft_us = r.end_time_us;
	r.base_model.accumulated_phase_rad = r.history[r.count - 1].accumulated_phase_rad;
	r.base_model.frequency_hz = final_frequency;
	Algorithm::Model model;
	const auto result = f.algorithm->update(r, model);
	CHECK(result.training_error == 0 && model.model_updated);
	CHECK(std::abs(model.correction.candidate_rad - 0.4) < 0.005);
	CHECK(prediction_rms(f, model, final_frequency, 0.4) < 0.01);
}

void incomplete_or_invalid_prediction_window()
{
	Fixture f;
	f.cycles(0.8);
	Algorithm::Model model;
	auto result = f.algorithm->update(*f.request, model);
	CHECK(result.training_error == -EDOM);
	same_waveform(model, f.request->base_model);
	f.cycles(1.28);
	f.request->history[70].prediction_valid = false;
	result = f.algorithm->update(*f.request, model);
	CHECK(result.training_error == -EDOM);
	same_waveform(model, f.request->base_model);
}

void truncated_window_coverage()
{
	Algorithm::Config config;
	config.memory_cycles = 1;
	Fixture f(config);
	f.cycles(2);
	auto &r = *f.request;
	for (size_t i = 0; i + 50 < r.count; ++i) { r.history[i] = r.history[i + 50]; }
	r.count -= 50;
	r.history_truncated = true;
	Algorithm::Model model;
	auto result = f.algorithm->update(r, model);
	// Missing older block data is harmless when the latest complete cycle is retained.
	CHECK(result.training_error == 0 && model.model_updated);
	for (size_t i = 0; i + 75 < r.count; ++i) { r.history[i] = r.history[i + 75]; }
	r.count -= 75;
	result = f.algorithm->update(r, model);
	CHECK(result.training_error == -ENOSPC);
	same_waveform(model, r.base_model);
}

void delayed_background_delivery()
{
	Algorithm::Config config;
	config.memory_cycles = 2;
	Fixture f(config);
	Algorithm::Model model {}, candidate {};
	Algorithm::UpdateResult result {};
	bool pending = false, delayed = false;
	unsigned due = 0, successful_online = 0, predictions = 0;
	for (unsigned first = 1; first <= 1496; first += 8) {
		if (pending && first >= due) {
			if (candidate.model_updated) {
				CHECK(candidate.block_start_accumulated_phase_rad == candidate.accumulated_phase_rad);
				CHECK(candidate.block_end_us == f.request->end_time_us);
				if (f.request->base_model.valid) { ++successful_online; }
			}
			model = candidate;
			f.algorithm->finish_update(*f.request, model);
			pending = false;
		}
		Algorithm::Sample samples[8];
		for (unsigned j = 0; j < 8; ++j) {
			const unsigned i = first + j;
			samples[j] = {1000000 + uint64_t(i) * 2000, float(std::sin(TWO_PI * 12 * i / 500.0))};
		}
		CHECK(f.algorithm->validate_batch(samples, 8) == 0);
		for (const auto &sample : samples) { predictions += f.algorithm->process_sample(sample, model, true).valid; }
		if (!pending && f.algorithm->capture_update(model, *f.request)) {
			unsigned delay = 0;
			if (f.request->train && model.valid) {
				CHECK(f.request->history[f.request->count - 1].accumulated_phase_rad
				      - model.block_start_accumulated_phase_rad >= TWO_PI - 1e-8);
				CHECK(f.request->end_time_us - model.block_end_us >= 83000);
				if (!delayed) { delay = 105; delayed = true; }
			}
			result = f.algorithm->update(*f.request, candidate);
			CHECK(result.frequency_error == 0);
			due = first + 8 + delay;
			pending = true;
		}
	}
	CHECK(delayed && successful_online > 10 && predictions > 600);
}

void frequency_request(Fixture &f, uint64_t end_us, uint64_t old_reference_us)
{
	f.cycles(6.12, 12, 500, 0.8);
	auto &r = *f.request;
	const uint64_t offset = end_us - r.end_time_us;
	for (size_t i = 0; i < r.count; ++i) { r.history[i].time_us += offset; }
	r.end_time_us = end_us;
	r.frequency = true;
	r.train = false;
	r.base_model.reference_time_us = r.base_model.last_good_fft_us = old_reference_us;
	r.base_model.memory_start_us = old_reference_us - 100000;
	r.base_model.block_end_us = old_reference_us;
	r.base_model.phase_offset_rad = 0.3;
	for (size_t i = 0; i < r.count; ++i) {
		r.history[i].phase = wrap(r.history[i].accumulated_phase_rad + 0.3);
		r.history[i].periodic = std::sin(r.history[i].phase);
	}
	for (size_t i = 0; i < Algorithm::MAX_CLUSTERS; ++i) {
		r.base_model.centers[i] = wrap(r.base_model.centers[i] + 0.3);
	}
	std::sort(r.base_model.centers, r.base_model.centers + Algorithm::MAX_CLUSTERS);
}

void expired_model_requires_retraining()
{
	Fixture f;
	frequency_request(f, 5000000, 1000000);
	Algorithm::Model model;
	auto result = f.algorithm->update(*f.request, model);
	CHECK(result.frequency_error == 0 && model.fft_accepted);
	CHECK(!model.valid && !model.stale && !model.model_updated);
	CHECK(model.memory_start_us == 5000000 && model.block_end_us == 5000000);
	CHECK(model.block_start_accumulated_phase_rad == model.accumulated_phase_rad);
	CHECK(model.phase_offset_rad == 0);
	CHECK(!f.algorithm->process_sample({5002000, 0}, model, false).valid);

	bool recovered = false;
	for (unsigned i = 1; i <= 1000; ++i) {
		const Algorithm::Sample sample {5000000 + uint64_t(i) * 2000,
			float(std::sin(TWO_PI * 12 * i / 500.0 + 0.8))};
		CHECK(f.algorithm->validate_batch(&sample, 1) == 0);
		f.algorithm->process_sample(sample, model, true);
		if (i % 8 == 0 && f.algorithm->capture_update(model, *f.request)) {
			Algorithm::Model candidate;
			result = f.algorithm->update(*f.request, candidate);
			CHECK(result.frequency_error == 0);
			if (candidate.model_updated) {
				CHECK(candidate.accumulated_phase_rad - model.block_start_accumulated_phase_rad >= 8 * TWO_PI - 1e-8);
				recovered = true;
			}
			model = candidate;
			f.algorithm->finish_update(*f.request, model);
			if (recovered) { break; }
		}
		CHECK(!model.valid);
	}
	CHECK(recovered && model.valid);
}

void short_gap_preserves_template()
{
	Fixture f;
	frequency_request(f, 5000000, 4800000);
	Algorithm::Model model;
	const auto result = f.algorithm->update(*f.request, model);
	CHECK(result.frequency_error == 0 && model.fft_accepted);
	CHECK(model.valid && !model.stale && !model.model_updated);
	CHECK(model.memory_start_us == f.request->base_model.memory_start_us);
	same_waveform(model, f.request->base_model);
}

void rejected_fft_keeps_expired_template_disabled()
{
	Fixture f;
	frequency_request(f, 5000000, 1000000);
	for (size_t i = 0; i < f.request->count; ++i) { f.request->history[i].raw = 0; }
	Algorithm::Model model;
	const auto result = f.algorithm->update(*f.request, model);
	CHECK(result.frequency_error == 0 && !model.fft_accepted);
	CHECK(model.stale && !model.valid && !model.model_updated);
	CHECK(model.last_good_fft_us == 1000000);
}

void successful_frequency_survives_training_failure()
{
	Fixture f;
	frequency_request(f, 5000000, 4800000);
	auto &r = *f.request;
	r.train = true;
	r.history_truncated = true;
	r.base_model.frequency_hz = 11;
	r.base_model.reference_time_us = r.end_time_us;
	r.base_model.accumulated_phase_rad = r.history[r.count - 1].accumulated_phase_rad;
	r.base_model.memory_start_us = r.history[0].time_us - 1;
	r.base_model.block_end_us = r.history[0].time_us;
	Algorithm::Model model;
	const auto result = f.algorithm->update(r, model);
	CHECK(result.frequency_attempted && result.frequency_error == 0 && model.fft_accepted);
	CHECK(result.training_attempted && result.training_error == -ENOSPC);
	CHECK(model.frequency_hz > 11.4 && model.last_good_fft_us == r.end_time_us);
	CHECK(model.valid && !model.model_updated);
	same_waveform(model, r.base_model);
	CHECK(model.reference_time_us == r.end_time_us);
}

void fit_failure_rolls_back_correction()
{
	Algorithm::Config config;
	config.kernel_scale = config.noise_floor = config.jitter = 1e308;
	config.cholesky_attempts = 1;
	Fixture f(config);
	f.cycles(1.28);
	Algorithm::Model model;
	const auto result = f.algorithm->update(*f.request, model);
	CHECK(result.training_attempted && result.training_error == -ERANGE);
	CHECK(model.valid && !model.model_updated);
	same_waveform(model, f.request->base_model);
}

void combined_frequency_and_waveform_success()
{
	Algorithm::Config config;
	config.memory_cycles = 2;
	Fixture f(config);
	frequency_request(f, 5000000, 4800000);
	auto &r = *f.request;
	r.train = true;
	r.base_model.reference_time_us = r.end_time_us;
	r.base_model.accumulated_phase_rad = r.history[r.count - 1].accumulated_phase_rad;
	r.base_model.block_end_us = r.history[0].time_us;
	r.base_model.memory_start_us = r.history[0].time_us - 1;
	Algorithm::Model model;
	const auto result = f.algorithm->update(r, model);
	CHECK(result.frequency_attempted && result.frequency_error == 0 && model.fft_accepted);
	CHECK(result.training_attempted && result.training_error == 0 && model.model_updated);
	CHECK(model.block_end_us == r.end_time_us);
	CHECK(model.block_start_accumulated_phase_rad == model.accumulated_phase_rad);
	CHECK(prediction_rms(f, model, 12, 0.8) < 0.015);
}

void frequency_error_still_publishes_expiry()
{
	Fixture f;
	frequency_request(f, 5000000, 1000000);
	auto &r = *f.request;
	// A deliberately unsupported sampling band exercises the frequency-error path.
	for (size_t i = 0; i < r.count; ++i) { r.history[i].time_us = 5000000 - (r.count - 1 - i) * 250; }
	Algorithm::Model model;
	const auto result = f.algorithm->update(r, model);
	CHECK(result.frequency_attempted && result.frequency_error == -ERANGE);
	CHECK(!result.training_attempted && model.stale && !model.valid);
	CHECK(model.last_good_fft_us == r.base_model.last_good_fft_us);
}
}

int main()
{
	struct Case { const char *name; void (*run)(); };
	const Case cases[] {
		{"phase correction and fit", phase_correction_and_fit},
		{"mixed historical coordinates", mixed_historical_coordinates},
		{"noninteger and multiple-cycle batches", arbitrary_batch_lengths},
		{"frequency ramp phase grid", frequency_ramp_window},
		{"incomplete or invalid prediction window", incomplete_or_invalid_prediction_window},
		{"truncated window coverage", truncated_window_coverage},
		{"delayed background result delivery", delayed_background_delivery},
		{"expired model requires retraining", expired_model_requires_retraining},
		{"short gap preserves template", short_gap_preserves_template},
		{"rejected FFT keeps expiry", rejected_fft_keeps_expired_template_disabled},
		{"frequency survives training failure", successful_frequency_survives_training_failure},
		{"fit failure rolls back correction", fit_failure_rolls_back_correction},
		{"combined update success", combined_frequency_and_waveform_success},
		{"frequency error publishes expiry", frequency_error_still_publishes_expiry},
	};
	unsigned failures = 0;
	for (const auto &test : cases) {
		try {
			test.run();
			std::printf("PASS %s\n", test.name);
		} catch (const std::exception &error) {
			++failures;
			std::printf("FAIL %s: %s\n", test.name, error.what());
		}
	}
	std::printf("%zu cases, %u failures\n", sizeof(cases) / sizeof(cases[0]), failures);
	return failures ? 1 : 0;
}
