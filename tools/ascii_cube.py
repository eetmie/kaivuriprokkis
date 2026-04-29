"""
ASCII 3D Cube Renderer for G3 IMU Sensor Demo

Renders a wireframe cube in the terminal that rotates
according to quaternion input from the sensor.

Imported and called with real quaternion data from the sensor.

No dependencies beyond the standard library!
"""

import math
import sys

# ============================================================
# Screen settings
# ============================================================
# Characters are taller than they are wide in a terminal,
# so we stretch horizontally to make the cube look square.

WIDTH = 60
HEIGHT = 30
CHAR_ASPECT = 2.0  # how much wider we draw to compensate for tall chars

# How big the cube appears on screen.
# This is in "screen units" — the cube corners go from -SCALE to +SCALE.
SCALE = 11

# ============================================================
# The 8 corners of a unit cube, centered at origin
# ============================================================
#
#     4 -------- 5          Z (up)        Left-handed coords
#    /|         /|          |             (G3 sensor default)
#   7 -------- 6 |         |
#   | |        | |   X ----+
#   | 0 -------| 1  (left)  \
#   |/         |/             Y (away from you)
#   3 -------- 2
#

VERTICES = [
    (-1, -1, -1),  # 0: back  left  bottom
    ( 1, -1, -1),  # 1: back  right bottom
    ( 1,  1, -1),  # 2: front right bottom
    (-1,  1, -1),  # 3: front left  bottom
    (-1, -1,  1),  # 4: back  left  top
    ( 1, -1,  1),  # 5: back  right top
    ( 1,  1,  1),  # 6: front right top
    (-1,  1,  1),  # 7: front left  top
]

# Each edge connects two vertices by their index
EDGES = [
    # bottom face
    (0, 1), (1, 2), (2, 3), (3, 0),
    # top face
    (4, 5), (5, 6), (6, 7), (7, 4),
    # vertical edges connecting top and bottom
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# We also draw axis arrows from the center of the cube:
#   X = right       (label 'X')
#   Y = toward you  (label 'Y')
#   Z = up          (label 'Z')
# Left-handed coordinate system (G3 sensor default, see manual Figure 3)
AXIS_LENGTH = 1.6  # slightly longer than the cube face


# ============================================================
# Quaternion -> rotation
# ============================================================

def quat_rotate_point(q, point):
    """
    Rotate a 3D point by a quaternion.

    The math: rotated = q * p * q_conjugate
    where p is the point as a pure quaternion (0, x, y, z).

    This is the standard way to apply a quaternion rotation
    to a point in 3D space.
    """
    qw, qx, qy, qz = q
    px, py, pz = point

    # This is the expanded form of q * (0,px,py,pz) * conjugate(q)
    # Saves us from doing two full quaternion multiplications.

    # Common sub-expressions
    qw2 = qw * qw
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz

    # Rotated point
    rx = px * (qw2 + qx2 - qy2 - qz2) + py * 2 * (qx * qy - qw * qz) + pz * 2 * (qx * qz + qw * qy)
    ry = px * 2 * (qx * qy + qw * qz) + py * (qw2 - qx2 + qy2 - qz2) + pz * 2 * (qy * qz - qw * qx)
    rz = px * 2 * (qx * qz - qw * qy) + py * 2 * (qy * qz + qw * qx) + pz * (qw2 - qx2 - qy2 + qz2)

    return (rx, ry, rz)


# ============================================================
# 3D -> 2D projection (orthographic, centered on cube)
# ============================================================

def project(point_3d):
    """
    Project a 3D point onto the 2D screen.

    We use ORTHOGRAPHIC projection (no perspective distortion).
    This means the camera looks straight at the center of the cube
    and things don't get smaller when they're further away.

    This keeps the cube centered and easy to read in ASCII.

    CHAR_ASPECT compensates for terminal characters being tall.

    Returns (screen_x, screen_y) as integers.
    """
    x, y, z = point_3d

    # Z-up, X-left (G3 sensor convention):
    #   X -> screen horizontal (left is positive)
    #   Z -> screen vertical   (up is positive)
    #   Y -> depth             (ignored in orthographic)
    screen_x = int(WIDTH / 2 - x * SCALE * CHAR_ASPECT)
    screen_y = int(HEIGHT / 2 - z * SCALE)  # z is up, screen y goes down

    return (screen_x, screen_y)


# ============================================================
# Drawing
# ============================================================

def draw_line(screen, x0, y0, x1, y1, char='#'):
    """
    Draw a line on the screen buffer using Bresenham's algorithm.

    This is the classic way to rasterize a line onto a grid.
    It figures out which cells to fill between two endpoints.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        # Only draw if within screen bounds
        if 0 <= x0 < WIDTH and 0 <= y0 < HEIGHT:
            screen[y0][x0] = char

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def render_frame(quat, roll_deg=0, pitch_deg=0, yaw_deg=0):
    """
    Render one frame of the cube rotated by the given quaternion.

    Returns a string ready to print to the terminal.
    """
    # Create empty screen buffer (2D array of spaces)
    screen = [[' ' for _ in range(WIDTH)] for _ in range(HEIGHT)]

    # Rotate and project all cube vertices
    projected = []
    for v in VERTICES:
        rotated = quat_rotate_point(quat, v)
        projected.append(project(rotated))

    # Draw all cube edges
    for i, j in EDGES:
        p1 = projected[i]
        p2 = projected[j]
        draw_line(screen, p1[0], p1[1], p2[0], p2[1], '#')

    # Draw axis arrows from center
    origin_2d = project((0, 0, 0))

    axes = [
        ((AXIS_LENGTH, 0, 0), 'X', 'x'),  # X axis
        ((0, AXIS_LENGTH, 0), 'Y', 'y'),  # Y axis
        ((0, 0, AXIS_LENGTH), 'Z', 'z'),  # Z axis
    ]

    for axis_end, label, line_char in axes:
        rotated_end = quat_rotate_point(quat, axis_end)
        p_end = project(rotated_end)
        draw_line(screen, origin_2d[0], origin_2d[1], p_end[0], p_end[1], line_char)
        # Place the label at the tip (overwrites the last line char)
        if 0 <= p_end[0] < WIDTH and 0 <= p_end[1] < HEIGHT:
            screen[p_end[1]][p_end[0]] = label

    # Build the output string
    border = '+' + '-' * WIDTH + '+'
    lines = [border]
    for row in screen:
        lines.append('|' + ''.join(row) + '|')
    lines.append(border)

    # Add angle readout below the cube
    lines.append(f"  Roll(X)={roll_deg:+7.2f}°   Pitch(Y)={pitch_deg:+7.2f}°   Yaw(Z)={yaw_deg:+7.2f}°")
    lines.append(f"  Quat: W={quat[0]:+.4f}  X={quat[1]:+.4f}  Y={quat[2]:+.4f}  Z={quat[3]:+.4f}")
    lines.append("")
    lines.append("  Axes:  x/X = X-axis   y/Y = Y-axis   z/Z = Z-axis   # = cube edge")

    return '\n'.join(lines)


def clear_screen():
    """Move cursor to top-left. Faster than clearing the whole terminal."""
    sys.stdout.write('\033[H')
    sys.stdout.flush()
