#pragma once

#include "GPR_Algorithm.hpp"

#include <atomic>
#include <lib/sensor_calibration/Gyroscope.hpp>
#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/sem.h>
#include <uORB/Subscription.hpp>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/sensor_gyro_fifo.h>

extern "C" __EXPORT int gpr_learning_main(int argc, char *argv[]);

/** One selected gyro, one model and one learning slot.
 * The driver calls process_fifo at its common publication entry. Raw integer
 * samples remain unchanged; the applied body-Y correction is attached to the
 * same FIFO so its FIFO and averaged consumers use the same compensation.
 */
class GPR_Learning : public ModuleBase<GPR_Learning>, public ModuleParams
{
public:
	explicit GPR_Learning(uint32_t device_id);
	~GPR_Learning() override;

	static int task_spawn(int argc, char *argv[]);
	static GPR_Learning *instantiate(int argc, char *argv[]);
	static int custom_command(int argc, char *argv[]);
	static int print_usage(const char *reason = nullptr);
	int print_status() override;

	void run() override;
	void request_stop() override;

	/** External FIFO entry. The driver clears applied correction fields first.
	 * Only this fixed physical device is processed. Units/extent/timing are
	 * validated once here; algorithm helpers receive the converted batch.
	 */
	void process_fifo(sensor_gyro_fifo_s &fifo);

private:
	bool init();
	void parameters_update();
	void receive_model();
	void RunLearning();

	const uint32_t _device_id;
	calibration::Gyroscope _calibration;
	uORB::Subscription _parameter_update_sub {ORB_ID(parameter_update)};
	DEFINE_PARAMETERS(
		(ParamInt<px4::params::GPR_ENABLE>) _param_gpr_enable
	)

	px4_sem_t _learning_wake {};
	bool _semaphore_initialized {};
	enum class LearningSlot : uint8_t { Idle, Running, Complete };
	std::atomic<LearningSlot> _learning_slot {LearningSlot::Idle};

	GPR_Algorithm _algorithm;
	GPR_Algorithm::UpdateRequest _request {};
	GPR_Algorithm::Model _candidate {}, _model {};
	int _learning_status {};
};