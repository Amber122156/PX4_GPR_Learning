# The common gyro driver entry links to this module in flat builds.
# This module depends on calibration only; it never links back to the driver.
px4_add_module(
	MODULE modules__gpr_learning
	MAIN gpr_learning
	STACK_MAIN 4096
	SRCS
		GPR_Learning.cpp
		GPR_Algorithm.cpp
	DEPENDS
		sensor_calibration
)

# parameters.c is discovered by PX4's parameter-generation step.