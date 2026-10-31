#include "GPR_Learning.hpp"
#include "GPR_Hook.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <inttypes.h>
#include <new>
#include <pthread.h>
#include <px4_platform_common/getopt.h>
#include <px4_platform_common/log.h>
#include <px4_platform_common/tasks.h>
#include <uORB/topics/sensor_selection.h>

namespace
{
pthread_mutex_t entry_mutex = PTHREAD_MUTEX_INITIALIZER;
GPR_Learning *entry_instance = nullptr;
}

void gpr_process_gyro_fifo(sensor_gyro_fifo_s &fifo)
{
	pthread_mutex_lock(&entry_mutex);
	if (entry_instance) { entry_instance->process_fifo(fifo); }
	pthread_mutex_unlock(&entry_mutex);
}

GPR_Learning::GPR_Learning(uint32_t device_id) :
	ModuleParams(nullptr),
	_device_id(device_id),
	_calibration(device_id)
{
}

GPR_Learning::~GPR_Learning()
{
	if (_semaphore_initialized) { px4_sem_destroy(&_learning_wake); }
}

bool GPR_Learning::init()
{
	parameters_update();
	if (_algorithm.configure(GPR_Algorithm::Config{}) != 0) { return false; }
	if (px4_sem_init(&_learning_wake, 0, 0) != 0) { return false; }
	_semaphore_initialized = true;
	px4_sem_setprotocol(&_learning_wake, SEM_PRIO_NONE);
	return true;
}

GPR_Learning *GPR_Learning::instantiate(int argc, char *argv[])
{
	uint32_t device_id = 0;
	int option_index = 1;
	const char *option_argument = nullptr;
	int option;

	while ((option = px4_getopt(argc, argv, "d:", &option_index, &option_argument)) != EOF) {
		if (option != 'd') { print_usage("unknown option"); return nullptr; }
		char *end = nullptr;
		errno = 0;
		const unsigned long long parsed = std::strtoull(option_argument, &end, 0);
		if (errno != 0 || *end != '\0' || parsed == 0 || parsed > UINT32_MAX) {
			print_usage("invalid gyro device id");
			return nullptr;
		}
		device_id = static_cast<uint32_t>(parsed);
	}

	auto *instance = new (std::nothrow) GPR_Learning(device_id);
	if (instance && !instance->init()) {
		delete instance;
		instance = nullptr;
	}
	if (instance) {
		pthread_mutex_lock(&entry_mutex);
		entry_instance = instance;
		pthread_mutex_unlock(&entry_mutex);
	}
	return instance;
}

int GPR_Learning::task_spawn(int argc, char *argv[])
{
	(void)argc;
	_task_id = px4_task_spawn_cmd("gpr_learning", SCHED_DEFAULT, SCHED_PRIORITY_DEFAULT - 10,
				     PX4_STACK_ADJUSTED(4096), (px4_main_t)&run_trampoline, (char *const *)argv);
	if (_task_id < 0) {
		_task_id = -1;
		return -errno;
	}
	return wait_until_running();
}

void GPR_Learning::run()
{
	while (!should_exit()) {
		if (px4_sem_wait(&_learning_wake) == 0 && !should_exit()) { RunLearning(); }
	}
	// No model is released until the last input call and the current update finish.
	pthread_mutex_lock(&entry_mutex);
	entry_instance = nullptr;
	pthread_mutex_unlock(&entry_mutex);
}

void GPR_Learning::request_stop()
{
	// Publish the request immediately. The input loop observes it between points.
	ModuleBase<GPR_Learning>::request_stop();
	px4_sem_post(&_learning_wake);
}

void GPR_Learning::parameters_update()
{
	updateParams();
	_calibration.ParametersUpdate();
}

void GPR_Learning::process_fifo(sensor_gyro_fifo_s &fifo)
{
	if (should_exit()) { return; }

	if (_device_id == 0) {
		uORB::Subscription selection_sub {ORB_ID(sensor_selection)};
		sensor_selection_s selection {};
		if (!selection_sub.copy(&selection) || selection.gyro_device_id == 0) { return; }
		_calibration.set_device_id(selection.gyro_device_id);
		_device_id = selection.gyro_device_id;
	}
	if (fifo.device_id != _device_id) { return; }

	parameter_update_s parameter_update {};
	if (_parameter_update_sub.update(&parameter_update)) { parameters_update(); }
	_calibration.SensorCorrectionsUpdate();
	_param_gpr_enable.update();

	// Completed jobs are received only at a batch boundary, including while off.
	if (!_param_gpr_enable.get()) {
		receive_model();
		return;
	}

	// Validate FIFO structure once before forming the trusted scalar batch.
	const size_t count = fifo.samples;
	constexpr size_t capacity = sizeof(fifo.x) / sizeof(fifo.x[0]);
	if (count == 0 || count > capacity || !std::isfinite(fifo.dt) || fifo.dt <= 0
	    || !std::isfinite(fifo.scale) || fifo.scale <= 0) { return; }
	const double span_us = (count - 1) * double(fifo.dt);
	if (double(fifo.timestamp_sample) < span_us) { return; }

	GPR_Algorithm::Sample samples[capacity];
	for (size_t i = 0; i < count; ++i) {
		const matrix::Vector3f raw {fifo.scale * fifo.x[i], fifo.scale * fifo.y[i], fifo.scale * fifo.z[i]};
		const uint64_t offset_us = static_cast<uint64_t>(std::llround((count - 1 - i) * double(fifo.dt)));
		samples[i].timestamp_sample_us = fifo.timestamp_sample - offset_us;
		samples[i].value = _calibration.Correct(raw)(1);
	}
	if (_algorithm.validate_batch(samples, count) != 0) { return; }

	receive_model();
	// R maps the FIFO frame to body. R^T e_y is row 1 of R.
	for (size_t axis = 0; axis < 3; ++axis) { fifo.gpr_axis[axis] = _calibration.rotation()(1, axis); }
	for (size_t i = 0; i < count; ++i) {
		_param_gpr_enable.update();
		if (should_exit() || !_param_gpr_enable.get()) { break; }
		// Once entered, this point completes even if a stop arrives during prediction.
		const auto prediction = _algorithm.process_sample(samples[i], _model, true);
		if (prediction.valid) { fifo.gpr_periodic[i] = static_cast<float>(prediction.periodic); }
	}

	_param_gpr_enable.update();
	if (!should_exit() && _param_gpr_enable.get()
	    && _learning_slot.load(std::memory_order_acquire) == LearningSlot::Idle
	    && _algorithm.capture_update(_model, _request)) {
		_learning_slot.store(LearningSlot::Running, std::memory_order_release);
		px4_sem_post(&_learning_wake);
	}
}

void GPR_Learning::RunLearning()
{
	if (_learning_slot.load(std::memory_order_acquire) != LearningSlot::Running) { return; }
	_learning_status = _algorithm.update(_request, _candidate);
	_learning_slot.store(LearningSlot::Complete, std::memory_order_release);
}

void GPR_Learning::receive_model()
{
	if (_learning_slot.load(std::memory_order_acquire) != LearningSlot::Complete) { return; }
	if (_learning_status == 0) {
		_model = _candidate;
		_algorithm.finish_update(_request, _model);
	}
	_learning_slot.store(LearningSlot::Idle, std::memory_order_release);
}

int GPR_Learning::print_status()
{
	pthread_mutex_lock(&entry_mutex);
	const uint32_t device_id = _device_id;
	pthread_mutex_unlock(&entry_mutex);

	PX4_INFO("gyro device: %" PRIu32 ", learning slot: %u",
		 device_id, static_cast<unsigned>(_learning_slot.load(std::memory_order_acquire)));
	return 0;
}

int GPR_Learning::custom_command(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	print_usage("unknown command");
	return -EINVAL;
}

int GPR_Learning::print_usage(const char *reason)
{
	if (reason) { PX4_WARN("%s", reason); }
	PRINT_MODULE_DESCRIPTION(
		"Learn periodic body-Y gyro disturbance at the common FIFO entry.\n"
		"One fixed physical gyro and one background learning task.");
	PRINT_MODULE_USAGE_NAME("gpr_learning", "estimator");
	PRINT_MODULE_USAGE_COMMAND("start");
	PRINT_MODULE_USAGE_PARAM_INT('d', 0, 0, 0, "Physical gyro device id; default binds once selection is available", true);
	PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
	return 0;
}

int gpr_learning_main(int argc, char *argv[])
{
	return GPR_Learning::main(argc, argv);
}
