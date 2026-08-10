"""simple_drive input-source and sine-generator tests.

Covers the local-gamepad path (axis signs, button mask, disconnect handling)
and the D-pad-selected sine target. The real pad cannot be driven from a
test, so LocalGamepadInput is exercised against a stub XboxController.

Known test gaps
---------------
- UDPInput.poll() decoding is not covered here; it is unchanged wire handling
  that the existing UDP client exercises end to end.
- Trigger -> track mapping is forward-only by construction (triggers read
  0..1); no test asserts reverse track drive because the local path cannot
  produce it.
"""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import simple_drive  # noqa: E402
from simple_drive import (  # noqa: E402
    SINE_TARGET_MODES,
    BTN_A, BTN_B, BTN_X, BTN_Y,
    BTN_DPAD_DOWN, BTN_DPAD_LEFT, BTN_DPAD_RIGHT, BTN_DPAD_UP,
    LocalGamepadInput,
    SineExcitationGenerator,
)


class _StubPad:
    """Stands in for modules.gamepad.XboxController."""

    def __init__(self, **state):
        self.connected = True
        self.state = {
            'LeftJoystickX': 0.0, 'LeftJoystickY': 0.0,
            'RightJoystickX': 0.0, 'RightJoystickY': 0.0,
            'LeftTrigger': 0.0, 'RightTrigger': 0.0,
            'A': 0, 'B': 0, 'X': 0, 'Y': 0,
            'UpDPad': 0, 'DownDPad': 0, 'LeftDPad': 0, 'RightDPad': 0,
        }
        self.state.update(state)
        self.stopped = False

    def read(self):
        if not self.connected:
            return {k: (0.0 if isinstance(v, float) else 0) for k, v in self.state.items()}
        return dict(self.state)

    def is_connected(self):
        return self.connected

    def stop_monitoring(self):
        self.stopped = True


def _source(**state):
    src = LocalGamepadInput()
    src._pad = _StubPad(**state)
    return src


# ---------------------------------------------------------------------------
# Axis mapping
# ---------------------------------------------------------------------------

class LocalGamepadAxisTests(unittest.TestCase):
    """Signs must match clients.input_handler.InputHandler.GAMEPAD_DIRECT."""

    def test_left_x_drives_slew_inverted(self):
        axes, _ = _source(LeftJoystickX=1.0).poll()
        self.assertAlmostEqual(axes['left_rl'], -1.0)

    def test_right_y_drives_boom_upright(self):
        axes, _ = _source(RightJoystickY=1.0).poll()
        self.assertAlmostEqual(axes['right_ud'], +1.0)

    def test_left_y_drives_arm_inverted(self):
        axes, _ = _source(LeftJoystickY=1.0).poll()
        self.assertAlmostEqual(axes['left_ud'], -1.0)

    def test_right_x_drives_bucket_inverted(self):
        axes, _ = _source(RightJoystickX=1.0).poll()
        self.assertAlmostEqual(axes['right_rl'], -1.0)

    def test_triggers_drive_tracks(self):
        axes, _ = _source(RightTrigger=0.5, LeftTrigger=0.25).poll()
        self.assertAlmostEqual(axes['right_paddle'], 0.5)
        self.assertAlmostEqual(axes['left_paddle'], 0.25)

    def test_signs_match_input_handler_direct_map(self):
        # The GUI client applies these signs before encoding to UDP; the local
        # path must agree or the two sources drive opposite directions.
        from clients.input_handler import InputHandler
        axis_to_key = {
            'slew':   'left_rl',
            'boom':   'right_ud',
            'arm':    'left_ud',
            'bucket': 'right_rl',
        }
        for joint, (axis_name, sign) in InputHandler.GAMEPAD_DIRECT.items():
            axes, _ = _source(**{axis_name: 1.0}).poll()
            self.assertAlmostEqual(
                axes[axis_to_key[joint]], float(sign),
                msg=f"{joint} via {axis_name} disagrees with GAMEPAD_DIRECT",
            )


# ---------------------------------------------------------------------------
# Button mask
# ---------------------------------------------------------------------------

class LocalGamepadButtonTests(unittest.TestCase):

    def test_face_buttons_map_to_low_bits(self):
        for key, bit in (('A', BTN_A), ('B', BTN_B), ('X', BTN_X), ('Y', BTN_Y)):
            _, mask = _source(**{key: 1}).poll()
            self.assertEqual(mask, 1 << bit, msg=f"button {key}")

    def test_dpad_maps_to_high_bits(self):
        for key, bit in (('UpDPad', BTN_DPAD_UP), ('DownDPad', BTN_DPAD_DOWN),
                         ('LeftDPad', BTN_DPAD_LEFT), ('RightDPad', BTN_DPAD_RIGHT)):
            _, mask = _source(**{key: 1}).poll()
            self.assertEqual(mask, 1 << bit, msg=f"dpad {key}")

    def test_simultaneous_presses_combine(self):
        _, mask = _source(A=1, UpDPad=1).poll()
        self.assertEqual(mask, (1 << BTN_A) | (1 << BTN_DPAD_UP))

    def test_no_press_is_zero_mask(self):
        _, mask = _source().poll()
        self.assertEqual(mask, 0)


# ---------------------------------------------------------------------------
# Disconnect behaviour
# ---------------------------------------------------------------------------

class LocalGamepadDisconnectTests(unittest.TestCase):

    def test_disconnect_zeroes_axes_and_reports_not_live(self):
        src = _source(LeftJoystickX=1.0, RightJoystickY=-1.0, A=1)
        src._pad.connected = False
        axes, mask = src.poll()
        self.assertFalse(src.is_live())
        self.assertEqual(mask, 0)
        for name, value in axes.items():
            self.assertAlmostEqual(value, 0.0, msg=f"axis {name} not zeroed on disconnect")

    def test_close_stops_the_monitor_thread(self):
        src = _source()
        pad = src._pad
        src.close()
        self.assertTrue(pad.stopped)


# ---------------------------------------------------------------------------
# Sine target stepping (D-pad)
# ---------------------------------------------------------------------------

class SineTargetTests(unittest.TestCase):

    def test_step_target_moves_one_mode(self):
        gen = SineExcitationGenerator()
        start = gen.target_idx
        gen.step_target(+1)
        self.assertEqual(gen.target_name, SINE_TARGET_MODES[start + 1][0])
        gen.step_target(-1)
        self.assertEqual(gen.target_name, SINE_TARGET_MODES[start][0])

    def test_step_target_wraps_at_both_ends(self):
        """Wraps rather than clamps: cycling should never dead-end on the pad."""
        gen = SineExcitationGenerator()
        for _ in range(len(SINE_TARGET_MODES)):
            gen.step_target(+1)
        self.assertEqual(gen.target_idx, 0)
        gen.step_target(-1)
        self.assertEqual(gen.target_idx, len(SINE_TARGET_MODES) - 1)

    def test_modes_cover_singles_and_pairs(self):
        names = [n for n, _ in SINE_TARGET_MODES]
        self.assertEqual(
            names,
            ['all', 'lift', 'tilt', 'scoop', 'lift+tilt', 'lift+scoop', 'tilt+scoop'])
        for _, joints in SINE_TARGET_MODES:
            for joint in joints:
                self.assertIn(joint, simple_drive.JOINT_NAMES)

    def test_target_gates_which_joints_move(self):
        for idx, (name, joints) in enumerate(SINE_TARGET_MODES):
            gen = SineExcitationGenerator(enabled=True, seed=5)
            gen.target_idx = idx
            gen.start_time = 0.0
            peak = {j: 0.0 for j in simple_drive.JOINT_NAMES}
            t = 0.0
            for _ in range(2000):
                t += 1.0 / simple_drive.SAMPLING_FREQUENCY
                for joint, value in gen.get_all(t).items():
                    peak[joint] = max(peak[joint], abs(value))
            driven = {j for j, v in peak.items() if v > 0.0}
            self.assertEqual(driven, set(joints), f"target {name}")

    def test_amplitude_is_drawn_not_set(self):
        """Amplitude is per-joint and random; there is no operator knob left."""
        self.assertFalse(hasattr(SineExcitationGenerator, 'step_amplitude'))
        lo, hi = SineExcitationGenerator.AMPLITUDE_RANGE
        seen = set()
        for seed in range(40):
            gen = SineExcitationGenerator(seed=seed)
            for joint in simple_drive.JOINT_NAMES:
                amp = gen._params[joint]['amp']
                self.assertGreaterEqual(amp, lo)
                self.assertLessEqual(amp, hi)
                seen.add(round(amp, 6))
        self.assertGreater(len(seen), 100)   # genuinely varying, not quantized

    def test_reseed_starts_an_independent_draw(self):
        gen = SineExcitationGenerator(seed=1234)
        before = (gen.seed, gen._params['boom']['amp'])
        gen.reseed()
        self.assertNotEqual(gen.seed, before[0])
        self.assertNotEqual(gen._params['boom']['amp'], before[1])

    def test_output_never_leaves_the_valve_range(self):
        """Noise rides on top of the deterministic term, so clamp at source."""
        gen = SineExcitationGenerator(enabled=True, seed=11)
        gen.start_time = 0.0
        t = 0.0
        for _ in range(4000):
            t += 1.0 / simple_drive.SAMPLING_FREQUENCY
            for value in gen.get_all(t).values():
                self.assertGreaterEqual(value, -1.0)
                self.assertLessEqual(value, 1.0)

    def test_excitation_stays_within_valve_bandwidth(self):
        """The whole point of the retune: no energy the hydraulics cannot track."""
        gen = SineExcitationGenerator(enabled=True, seed=7)
        gen.start_time = 0.0
        fs = simple_drive.SAMPLING_FREQUENCY
        signal = []
        t = 0.0
        for _ in range(fs * 200):
            t += 1.0 / fs
            signal.append(gen.get_all(t)['boom'])
        import numpy as np
        power = np.abs(np.fft.rfft(signal)) ** 2
        freqs = np.fft.rfftfreq(len(signal), 1.0 / fs)
        above_1hz = power[1:][freqs[1:] > 1.0].sum() / power[1:].sum()
        self.assertLess(above_1hz, 0.005)


# ---------------------------------------------------------------------------
# Sine randomization
# ---------------------------------------------------------------------------

class SineRandomizationTests(unittest.TestCase):
    """Excitation must differ per channel and per run, but stay band-limited."""

    _PARAM_KEYS = ('f_env', 'f_car', 'f_rate', 'depth', 'phi1', 'phi2')

    def test_every_joint_gets_distinct_parameters(self):
        gen = SineExcitationGenerator(seed=42)
        for key in self._PARAM_KEYS:
            values = [gen._params[j][key] for j in simple_drive.JOINT_NAMES]
            self.assertEqual(len(set(values)), len(values),
                             msg=f"{key} repeats across joints: {values}")

    def test_same_seed_reproduces_the_session(self):
        a = SineExcitationGenerator(seed=99)
        b = SineExcitationGenerator(seed=99)
        self.assertEqual(a._params, b._params)

    def test_different_seeds_give_different_excitation(self):
        a = SineExcitationGenerator(seed=1)
        b = SineExcitationGenerator(seed=2)
        self.assertNotEqual(a._params, b._params)

    def test_seed_is_recorded_when_not_supplied(self):
        gen = SineExcitationGenerator()
        self.assertIsInstance(gen.seed, int)
        # Reconstructing from the recorded seed must reproduce the draw.
        self.assertEqual(SineExcitationGenerator(seed=gen.seed)._params, gen._params)

    def test_toggle_on_redraws_parameters(self):
        gen = SineExcitationGenerator(seed=5)
        before = {j: dict(p) for j, p in gen._params.items()}
        gen.toggle()      # on → re-draw
        self.assertNotEqual(gen._params, before)

    def test_peak_frequency_never_exceeds_the_ceiling(self):
        """The FM depth clamp is what enforces this; without it draws exceed 2 Hz.

        Above the ceiling the valves stop tracking the excitation, so the
        recording captures noise instead of actuator response.
        """
        ceiling = SineExcitationGenerator.MAX_INSTANT_FREQ_HZ
        worst = 0.0
        for seed in range(400):
            gen = SineExcitationGenerator(seed=seed)
            for joint in simple_drive.JOINT_NAMES:
                worst = max(worst, gen.peak_freq_hz(joint))
        self.assertLessEqual(worst, ceiling + 1e-9,
                             msg=f"peak instantaneous frequency reached {worst:.4f} Hz")

    def test_noise_is_present_and_band_limited(self):
        """Noise must add content without chattering the valves.

        Checked as a lag-1 autocorrelation at the 100 Hz sample rate: white
        noise would sit near 0, while a 1.2 Hz-cornered process stays highly
        correlated sample to sample. The latter is what the valves can follow.
        """
        import numpy as np

        gen = SineExcitationGenerator(enabled=True, seed=13)
        gen.start_time = 0.0
        dt = 1.0 / simple_drive.SAMPLING_FREQUENCY

        trace = {j: [] for j in simple_drive.JOINT_NAMES}
        t = 0.0
        for _ in range(6000):
            t += dt
            gen._advance_noise(t)
            for j in simple_drive.JOINT_NAMES:
                trace[j].append(gen._noise[j])

        for j, values in trace.items():
            x = np.array(values)
            self.assertGreater(x.std(), 0.3, msg=f"{j}: noise is not actually moving")
            lag1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
            self.assertGreater(lag1, 0.85,
                               msg=f"{j}: noise lag-1 autocorr {lag1:.3f} — too close to "
                                   f"white, which the valves cannot track")

    def test_noise_state_stays_bounded(self):
        gen = SineExcitationGenerator(enabled=True, seed=17)
        gen.start_time = 0.0
        t = 0.0
        for _ in range(20000):
            t += 1.0 / simple_drive.SAMPLING_FREQUENCY
            gen._advance_noise(t)
            for j in simple_drive.JOINT_NAMES:
                self.assertLessEqual(abs(gen._noise[j]),
                                     SineExcitationGenerator.NOISE_CLIP_SIGMA)

    def test_noise_level_is_independent_of_loop_rate(self):
        """Exact OU discretization: jitter changes timing, not noise level."""
        import numpy as np

        def std_at(rate_hz):
            gen = SineExcitationGenerator(enabled=True, seed=23)
            gen.start_time = 0.0
            t, out = 0.0, []
            for _ in range(8000):
                t += 1.0 / rate_hz
                gen._advance_noise(t)
                out.append(gen._noise['boom'])
            return float(np.std(out))

        fast, slow = std_at(100.0), std_at(25.0)
        self.assertAlmostEqual(fast, slow, delta=0.15,
                               msg=f"noise std moved with loop rate: {fast:.3f} vs {slow:.3f}")

    def test_get_all_advances_noise_but_get_signal_does_not(self):
        gen = SineExcitationGenerator(enabled=True, seed=29)
        gen.start_time = 0.0
        gen.get_all(0.01)
        gen.get_all(0.02)
        frozen = dict(gen._noise)
        for _ in range(5):
            gen.get_signal('boom', 0.03)
        self.assertEqual(gen._noise, frozen)
        gen.get_all(0.03)
        self.assertNotEqual(gen._noise, frozen)

    def test_channels_are_decorrelated(self):
        """The point of the randomization.

        With the old fixed per-joint phases, slew/arm and boom/bucket ran at
        |r| = 0.90 — near-collinear channels a blackbox model cannot separate.
        """
        import numpy as np

        gen = SineExcitationGenerator(enabled=True, seed=7)
        gen.start_time = 0.0
        t = np.arange(0.0, 300.0, 0.01)
        signals = {j: np.array([gen.get_signal(j, x) for x in t])
                   for j in simple_drive.JOINT_NAMES}

        for i, a in enumerate(simple_drive.JOINT_NAMES):
            for b in simple_drive.JOINT_NAMES[i + 1:]:
                r = abs(float(np.corrcoef(signals[a], signals[b])[0, 1]))
                self.assertLess(r, 0.30, msg=f"{a}/{b} correlated at |r|={r:.3f}")


if __name__ == "__main__":
    unittest.main()
