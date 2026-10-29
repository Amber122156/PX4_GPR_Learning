uint64 timestamp          # time since system start (microseconds)
uint64 timestamp_sample

uint32 device_id          # unique device ID for the sensor that does not change between power cycles

float32 dt                # delta time between samples (microseconds)
float32 scale

uint8 samples             # number of valid samples

int16[32] x               # angular velocity in the FRD board frame X-axis in rad/s
int16[32] y               # angular velocity in the FRD board frame Y-axis in rad/s
int16[32] z               # angular velocity in the FRD board frame Z-axis in rad/s

float32[32] gpr_periodic  # applied body Y angular velocity correction (rad/s), zero when inactive
float32[3] gpr_axis       # body Y unit vector in FIFO board frame (R^T e_y)

uint8 ORB_QUEUE_LENGTH = 4
