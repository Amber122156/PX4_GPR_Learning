#pragma once

struct sensor_gyro_fifo_s;

/** Called after driver rotation, device id and scale are set, before FIFO
 * publication and trapezoidal averaging. Raw xyz stay unchanged. The driver
 * initializes gpr_periodic/gpr_axis to zero before this optional entry.
 * The single-instance attachment and stop quiescence are defined by the module.
 */
void gpr_process_gyro_fifo(sensor_gyro_fifo_s &fifo);