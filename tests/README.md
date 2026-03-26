# Stress Tests

Hardware integration stress scripts for the 200 Hz excavator stack.

Both scripts set their working directory to the repository root automatically, so
they can be launched either from the repo root or from inside `tests/`.

## Full Stack Zero-Command Stress

Runs the real hardware, controller, service, protocol encode/decode, and telemetry path.
The controller is forced into direct mode and still sends explicit zero-valued PWM named
commands every cycle. It does not skip the PWM write path.

```bash
sudo python tests/stress_full_stack.py --rate-hz 200 --duration-s 60
```

Useful RT options:

```bash
sudo python tests/stress_full_stack.py --rate-hz 200 --duration-s 60 --fifo-priority 75 --lock-memory --control-core 2 --io-core 3
```

## USB Reader Stress

Runs the Pico USB IMU reader by itself to measure the practical host-side ceiling.

```bash
sudo python tests/stress_usb_reader.py --target-hz 200 --duration-s 30
```

```bash
sudo python tests/stress_usb_reader.py --target-hz 200 --duration-s 30 --fifo-priority 75 --lock-memory --cpu-core 3
```
