"""
Input handling for excavator GUI — keyboard + gamepad merge logic.

Extracted from client_gui.py to make input testable and decoupled from UI.
"""

import math
from typing import Optional, Set, Tuple

from modules.excavator_target_state import IKControlSpace


class InputHandler:
    """Processes keyboard and gamepad input into movement deltas."""

    # Gamepad axis mappings: (axis_name, sign)
    GAMEPAD_IK_CARTESIAN = {
        'dx': ('LeftJoystickX', +1),
        'dz': ('LeftJoystickY', +1),
        'dy': ('RightJoystickX', -1),
    }
    GAMEPAD_IK_RADIAL = {
        'dr': ('LeftJoystickY', +1),
        'd_slew': ('LeftJoystickX', +1),
        'dz': ('RightJoystickY', -1),
    }
    GAMEPAD_DIRECT = {
        'slew':   ('LeftJoystickX', -1),
        'boom':   ('RightJoystickY', +1),
        'arm':    ('LeftJoystickY', -1),
        'bucket': ('RightJoystickX', -1),
    }

    FAST_MULTIPLIER = 5.0
    GAMEPAD_POS_MULTIPLIER = 2.5

    def tick_ik(
        self,
        keys_down: Set[str],
        controller_state: Optional[dict],
        step_pos: float,
        step_rot: float,
        control_space: IKControlSpace = IKControlSpace.CARTESIAN,
        radius: float = 0.6,
    ) -> Tuple[float, float, float, float]:
        """Compute IK-mode deltas from keyboard + gamepad.

        Returns:
            Cartesian: (dx, dy, dz, drot)
            Radial: (dr, d_slew_deg, dz, drot)
        """
        if control_space == IKControlSpace.RADIAL:
            return self.tick_ik_radial(keys_down, controller_state, step_pos, step_rot, radius)
        return self.tick_ik_cartesian(keys_down, controller_state, step_pos, step_rot)

    def tick_ik_cartesian(
        self,
        keys_down: Set[str],
        controller_state: Optional[dict],
        step_pos: float,
        step_rot: float,
    ) -> Tuple[float, float, float, float]:
        """Compute Cartesian IK deltas (dx, dy, dz, drot)."""
        mult = self.FAST_MULTIPLIER if 'shift' in keys_down else 1.0
        dp = step_pos * mult
        dr = step_rot * mult

        # Keyboard
        kbd_dx = kbd_dy = kbd_dz = kbd_drot = 0.0
        if 'a' in keys_down:
            kbd_dx -= dp
        if 'd' in keys_down:
            kbd_dx += dp
        if 'w' in keys_down:
            kbd_dz += dp
        if 's' in keys_down:
            kbd_dz -= dp
        if 'q' in keys_down:
            kbd_dy -= dp
        if 'e' in keys_down:
            kbd_dy += dp
        if 'r' in keys_down:
            kbd_drot += dr
        if 'f' in keys_down:
            kbd_drot -= dr

        # Gamepad (only when LeftBumper held as safety)
        ctrl_dx = ctrl_dy = ctrl_dz = ctrl_drot = 0.0
        if controller_state is not None and controller_state.get('LeftBumper'):
            fast_dp = step_pos * self.GAMEPAD_POS_MULTIPLIER
            fast_dr = step_rot * self.FAST_MULTIPLIER
            ax, sx = self.GAMEPAD_IK_CARTESIAN['dx']
            az, sz = self.GAMEPAD_IK_CARTESIAN['dz']
            ay, sy = self.GAMEPAD_IK_CARTESIAN['dy']
            ctrl_dx = sx * controller_state[ax] * fast_dp
            ctrl_dz = sz * controller_state[az] * fast_dp
            ctrl_dy = sy * controller_state[ay] * fast_dp
            ctrl_drot = (controller_state['RightTrigger'] - controller_state['LeftTrigger']) * fast_dr

        # Per-axis: pick whichever source has larger abs
        dx = kbd_dx if abs(kbd_dx) >= abs(ctrl_dx) else ctrl_dx
        dy = kbd_dy if abs(kbd_dy) >= abs(ctrl_dy) else ctrl_dy
        dz = kbd_dz if abs(kbd_dz) >= abs(ctrl_dz) else ctrl_dz
        drot = kbd_drot if abs(kbd_drot) >= abs(ctrl_drot) else ctrl_drot

        return dx, dy, dz, drot

    def tick_ik_radial(
        self,
        keys_down: Set[str],
        controller_state: Optional[dict],
        step_pos: float,
        step_rot: float,
        radius: float = 0.6,
    ) -> Tuple[float, float, float, float]:
        """Compute radial IK deltas (dr, d_slew_deg, dz, drot)."""
        mult = self.FAST_MULTIPLIER if 'shift' in keys_down else 1.0
        dp = step_pos * mult
        dr = step_rot * mult
        # Convert linear step to angular step so arc-length matches step_pos
        d_slew = math.degrees(step_pos / max(radius, 0.01)) * mult

        # Keyboard: W/S=radius, A/D=slew, Q/E=vertical, R/F=tool pitch
        kbd_radius = kbd_slew = kbd_dz = kbd_drot = 0.0
        if 'w' in keys_down:
            kbd_radius += dp
        if 's' in keys_down:
            kbd_radius -= dp
        if 'a' in keys_down:
            kbd_slew -= d_slew
        if 'd' in keys_down:
            kbd_slew += d_slew
        if 'q' in keys_down:
            kbd_dz -= dp
        if 'e' in keys_down:
            kbd_dz += dp
        if 'r' in keys_down:
            kbd_drot += dr
        if 'f' in keys_down:
            kbd_drot -= dr

        ctrl_radius = ctrl_slew = ctrl_dz = ctrl_drot = 0.0
        if controller_state is not None and controller_state.get('LeftBumper'):
            fast_dp = step_pos * self.GAMEPAD_POS_MULTIPLIER
            fast_dr = step_rot * self.FAST_MULTIPLIER
            ar, sr = self.GAMEPAD_IK_RADIAL['dr']
            ayaw, syaw = self.GAMEPAD_IK_RADIAL['d_slew']
            az, sz = self.GAMEPAD_IK_RADIAL['dz']
            ctrl_radius = sr * controller_state[ar] * fast_dp
            fast_d_slew = math.degrees(step_pos * self.GAMEPAD_POS_MULTIPLIER / max(radius, 0.01))
            ctrl_slew = syaw * controller_state[ayaw] * fast_d_slew
            ctrl_dz = sz * controller_state[az] * fast_dp
            ctrl_drot = (controller_state['RightTrigger'] - controller_state['LeftTrigger']) * fast_dr

        radius = kbd_radius if abs(kbd_radius) >= abs(ctrl_radius) else ctrl_radius
        slew = kbd_slew if abs(kbd_slew) >= abs(ctrl_slew) else ctrl_slew
        dz = kbd_dz if abs(kbd_dz) >= abs(ctrl_dz) else ctrl_dz
        drot = kbd_drot if abs(kbd_drot) >= abs(ctrl_drot) else ctrl_drot

        return radius, slew, dz, drot

    def tick_direct(
        self,
        keys_down: Set[str],
        controller_state: Optional[dict],
    ) -> Tuple[float, float, float, float]:
        """Compute direct-mode valve commands [slew, boom, arm, bucket] in [-1, 1].

        Returns:
            (slew, boom, arm, bucket) — normalized valve commands.
        """
        mag = 1.0 if 'shift' in keys_down else 0.5
        slew = boom = arm = bucket = 0.0

        # Keyboard: A/D=slew, W/S=boom, Q/E=arm, R/F=bucket
        if 'a' in keys_down:
            slew -= mag
        if 'd' in keys_down:
            slew += mag
        if 'w' in keys_down:
            boom += mag
        if 's' in keys_down:
            boom -= mag
        if 'q' in keys_down:
            arm -= mag
        if 'e' in keys_down:
            arm += mag
        if 'r' in keys_down:
            bucket += mag
        if 'f' in keys_down:
            bucket -= mag

        # Gamepad: LeftBumper held -> analog direct control
        if controller_state is not None and controller_state.get('LeftBumper'):
            a_sl, s_sl = self.GAMEPAD_DIRECT['slew']
            a_bm, s_bm = self.GAMEPAD_DIRECT['boom']
            a_ar, s_ar = self.GAMEPAD_DIRECT['arm']
            a_bk, s_bk = self.GAMEPAD_DIRECT['bucket']
            gp_slew = s_sl * controller_state[a_sl]
            gp_boom = s_bm * controller_state[a_bm]
            gp_arm = s_ar * controller_state[a_ar]
            gp_bucket = s_bk * controller_state[a_bk]
            if abs(gp_slew) > abs(slew):
                slew = gp_slew
            if abs(gp_boom) > abs(boom):
                boom = gp_boom
            if abs(gp_arm) > abs(arm):
                arm = gp_arm
            if abs(gp_bucket) > abs(bucket):
                bucket = gp_bucket

        return (
            max(-1.0, min(1.0, slew)),
            max(-1.0, min(1.0, boom)),
            max(-1.0, min(1.0, arm)),
            max(-1.0, min(1.0, bucket)),
        )
