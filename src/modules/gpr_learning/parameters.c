/**
 * Periodic gyro disturbance learning and compensation.
 *
 * Enables learning and application for the fixed physical gyro selected when
 * gpr_learning starts. Original integer measurements remain available; the
 * common FIFO carries the applied correction for consistent downstream use.
 *
 * @boolean
 * @group GPR Learning
 */
PARAM_DEFINE_INT32(GPR_ENABLE, 0);

/* Numerical Config defaults remain unchanged; FFT tuning is a separate step. */