# GPR regression tests

The standalone host tests compile the production algorithm without PX4, a flight
simulator, or external test libraries. Run from the repository root:

```sh
cmake -S src/modules/gpr_learning/test -B /tmp/gpr-learning-test -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/gpr-learning-test
ctest --test-dir /tmp/gpr-learning-test --output-on-failure
```

The cases cover phase-coordinate changes across historical model versions,
complete-cycle correlation with noninteger batches and changing frequency,
missing predictions and truncated windows, delayed background result delivery,
expired and short-gap recovery, and frequency-only publication after waveform
failures. The delay test uses the public sample/capture/update/finish interface;
it simulates result latency rather than measuring a target CPU.

`update()` returns separate frequency/training errors. Its candidate is always a
consistent state that can be received at a batch boundary: successful frequency
tracking survives training failure, while incomplete correction/fit work stays
private. A zero error does not imply that an operation was attempted; check the
corresponding `*_attempted` field and the model's `fft_accepted`/`model_updated`.

`gpr_learning status` reports operation counts and the last nonzero error for
each branch. `-EDOM` means insufficient samples/predictions or an empty cluster,
`-ENOSPC` means missing required history, and `-ERANGE` means an unusable
frequency band or numerical failure. Rejected FFT peaks are counted as attempts
without acceptance, rather than numerical errors.

These tests establish host functional behavior, not target timing or flight
performance. Existing sample-rate defaults, memory capacities, enable behavior,
device binding, and driver/simulator entry points are unchanged.
