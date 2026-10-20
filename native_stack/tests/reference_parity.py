"""Optional Spark-only reference test: imports original PWM/FK code without hardware."""
from dataclasses import replace
import ast
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import yaml

binary, repo = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / "reference"))
from modules.pwm.config import ChannelConfig
from modules.pwm.controller import PWMController
# Execute the original function bodies without their JIT decorators/import chain.
# This needs NumPy and PyYAML on Spark, but never Numba or the full controller.
names = {"_rot_axis_angle_quat", "_quat_mul", "_quat_rotate_vec", "_fk_core"}
tree = ast.parse((repo / "reference/modules/ik/kinematics.py").read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(functions) == len(names)
for function in functions:
    function.decorator_list = []
reference_math = {"np": np}
exec(compile(ast.Module(body=functions, type_ignores=[]), "original_fk", "exec"), reference_math)

checks, max_pulse_error, max_fk_error = 0, 0., 0.
for profile_name in ("jetson", "jetson_bucket", "jetson_bucket_dz0"):
    profile = repo / "configuration_files/profiles" / profile_name
    servo = yaml.safe_load((profile / "servo_config.yaml").read_text())
    control = yaml.safe_load((profile / "control_config.yaml").read_text())
    configs = {}
    for name, raw in servo["CHANNEL_CONFIGS"].items():
        if name == "pump":
            continue
        fields = {key: value for key, value in raw.items() if key in ChannelConfig.__dataclass_fields__}
        if fields["center"] in ("None", "none"):
            fields["center"] = None
        cfg = ChannelConfig(**fields)
        configs[(name, 0)] = cfg
        configs[(name, 1)] = replace(cfg, ramp_enable=True, ramp_skip_deadband=True, ramp_limit=500)
        configs[(name, 2)] = replace(cfg, dither_taper=True, dither_taper_us=50, gamma=1.5)
    reference = PWMController.__new__(PWMController)
    reference._channel_ramp_state = {}
    reference._pwm_period_us = 1e6 / servo["pwm_frequency"]
    axes = np.array([j["axis"] for j in control["robot"]["joints"]], dtype=np.float32)
    offsets = np.array([j["parent_to_joint_xyz"] for j in control["robot"]["joints"]], dtype=np.float32)
    tip = np.array(control["robot"]["tool"]["parent_to_tip_xyz"], dtype=np.float32)
    previous = None
    output = subprocess.check_output([str(binary), str(profile)], text=True)
    for line in output.splitlines():
        row = json.loads(line)
        if row["type"] == "pwm":
            key = row["name"], row["variant"]
            if previous != key:
                reference._channel_ramp_state = {}
                previous = key
            pulse = reference._pulse_from_value(configs[key], row["value"], row["now"], apply_ramp=True)
            error = abs(pulse - row["pulse"])
            max_pulse_error = max(max_pulse_error, error)
            assert error < 1e-7, (profile_name, row, pulse)
            # Floating-point operation order can straddle an exact count boundary by one.
            assert abs(reference._count_from_pulse_us(pulse) - row["count"]) <= 1
        else:
            *_, position, orientation = reference_math["_fk_core"](np.array(row["angles"], dtype=np.float32), axes, offsets, tip)
            error = max(np.max(np.abs(position - row["position"])), np.max(np.abs(orientation - row["orientation"])))
            max_fk_error = max(max_fk_error, float(error))
            assert error < 5e-7, (profile_name, row, position, orientation)
        checks += 1
print(f"{checks} original-code parity cases passed; max PWM error {max_pulse_error:.3g} us, max FK component error {max_fk_error:.3g}")
