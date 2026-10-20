# kaivuriprokkis

Native C++ stack for the MASI excavator on Jetson Orin Nano: safety/valve control, IMU and FK,
dual RealSense IR (+ depth) capture, dataset collection (VLA episodes, hydraulics/MLP, IK/PID)
and TensorRT pi0.5 inference.

The Python stack (`simple_drive.py`, `lerobot_vla/`, `learned_control/`, ROS/Isaac tooling, …) is
frozen on branch [`python`](https://github.com/eetmie/kaivuriprokkis/tree/python). Port from there
when something is needed again.

| Path | Contents |
| --- | --- |
| [`native_stack/`](native_stack/README.md) | Services, collectors, inference, tests, notes. Start here. |
| `configuration_files/profiles/` | Robot calibration and servo profiles read by the native services |
| [`pico_imu_reader/`](pico_imu_reader/README.md) | Pico IMU firmware (the serial protocol `kaivuri-control` parses) |
| `pico_imu_simulator/` | Arduino IMU stream simulator for hardware-free tests |
| [`reference/`](reference/README.md) | Frozen Python sources the native parity tests compare against |
| `setup_jetson.sh` | One-time board setup: I2C, device groups, realtime limits, CPU governor |

Build and run: [native_stack/README.md](native_stack/README.md#build). Recording:
[native_stack/notes/RECORDING.md](native_stack/notes/RECORDING.md).
