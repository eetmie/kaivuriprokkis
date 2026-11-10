#!/usr/bin/env python3
"""Build the boom/arm/bucket surface model that sand_height.py masks depth points with.

  arm_model.py URDF [-o ../config/arm_model.npz]

URDF is the sim excavator (sim-scratchpad ros2/urdf/excavator.urdf, same machine). Each link
mesh is stored in a frame built from two points both sides know: its pivot and the next pivot
(the bucket's: its tip), with the lateral axis of the upper carriage. sand_height.py builds the
same frame from the robot's FK, so URDF and robot joint-zero conventions never meet, and the mesh
is stretched along the link to the robot's FK length (the URDF stick is 1 cm shorter).
Per link: distance (mm, uint8) from each 4 mm voxel to the mesh surface (the bucket: to its
convex hull, so carried sand is masked with it). Needs SciPy (build only).
"""
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

# robot FK link -> URDF links whose meshes ride on it, the URDF joint at its pivot, the next pivot
LINKS = {1: (('lift_boom',), 'revolute_lift', 'revolute_tilt'),
         2: (('tilt_boom',), 'revolute_tilt', 'revolute_tool'),
         3: (('tool_body', 'bucket'), 'revolute_tool', None)}  # None: bucket tip
VOXEL, PAD = .004, .05


def rpy(r):
    x, y, z = r; cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def origin(e):
    o, M = e.find('origin'), np.eye(4)
    if o is not None:
        M[:3, :3] = rpy([float(v) for v in o.get('rpy', '0 0 0').split()])
        M[:3, 3] = [float(v) for v in o.get('xyz', '0 0 0').split()]
    return M


def load_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith('v '):
            V.append(line.split()[1:4])
        elif line.startswith('f '):
            f = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            F += [(f[0], f[i], f[i + 1]) for i in range(1, len(f) - 1)]  # fan-triangulate polygons
    return np.asarray(V, float), np.asarray(F, int)


def surface(V, F, spacing=.0015, seed=0):
    """Vertices plus random points on every triangle, about one per spacing^2."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = .5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.random.default_rng(seed).poisson(area / spacing ** 2)
    i = np.repeat(np.arange(len(F)), n); r = np.random.default_rng(seed + 1).random((len(i), 2))
    flip = r.sum(1) > 1; r[flip] = 1 - r[flip]
    return np.vstack([V, a[i] + r[:, :1] * (b[i] - a[i]) + r[:, 1:] * (c[i] - a[i])])


def upper_frame_meshes(urdf):
    """Every link's surface points and every joint origin in the upper-carriage frame, joints at zero."""
    root = ET.parse(urdf).getroot(); base = Path(urdf).parent
    pose, joints = {'upper_carriage': np.eye(4)}, {}
    pending = [j for j in root.findall('joint')]
    while pending:  # walk the tree down from the upper carriage
        for j in list(pending):
            parent, child = j.find('parent').get('link'), j.find('child').get('link')
            if parent in pose:
                pose[child] = pose[parent] @ origin(j); joints[j.get('name')] = pose[child][:3, 3]; pending.remove(j)
            elif parent in ('world', 'lower_carriage'):
                pending.remove(j)
    points = {}
    for link in root.findall('link'):
        vis = link.find('visual')
        if link.get('name') not in pose or vis is None:
            continue
        mesh = base / 'meshes' / Path(vis.find('geometry/mesh').get('filename')).name
        V, F = load_obj(mesh); M = pose[link.get('name')] @ origin(vis)
        points[link.get('name')] = surface(V, F) @ M[:3, :3].T + M[:3, 3]
    return points, joints


def link_frame(pivot, nxt, lateral):
    """Rows: along the link, lateral, and their cross product."""
    e1 = nxt - pivot; e1 = e1 - (e1 @ lateral) * lateral; e1 /= np.linalg.norm(e1)
    return np.stack([e1, lateral, np.cross(e1, lateral)])


def build(urdf):
    from scipy.ndimage import distance_transform_edt
    points, joints = upper_frame_meshes(urdf)
    lateral, out = np.array([0, 1., 0]), {}
    for link, (names, pivot_joint, next_joint) in LINKS.items():
        P = np.vstack([points[n] for n in names]); pivot = joints[pivot_joint]
        if next_joint is None:  # bucket tip: the lip point farthest from the hinge, mid-width
            mid = np.abs(P[:, 1] - np.median(P[:, 1])) < .004
            d = np.linalg.norm((P - pivot)[:, [0, 2]], axis=1); d[~mid] = 0; nxt = P[np.argmax(d)]
        else:
            nxt = joints[next_joint]
        E = link_frame(pivot, nxt, lateral); Q = (P - pivot) @ E.T; length = float((nxt - pivot) @ E[0])
        lo = Q.min(0) - PAD; shape = tuple(np.ceil((Q.max(0) + PAD - lo) / VOXEL).astype(int) + 1)
        occupied = np.zeros(shape, bool); occupied[tuple(np.floor((Q - lo) / VOXEL).astype(int).T)] = True
        if next_joint is None:  # bucket: fill its convex hull, so carried sand counts as bucket
            from scipy.spatial import Delaunay
            cells = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), -1).reshape(-1, 3)
            occupied |= (Delaunay(Q).find_simplex(lo + (cells + .5) * VOXEL) >= 0).reshape(shape)
        dist = distance_transform_edt(~occupied) * VOXEL
        out[f'link{link}_mm'] = np.minimum(np.rint(dist * 1000), 255).astype(np.uint8)
        out[f'link{link}_lo'] = lo; out[f'link{link}_length'] = length
        print(f'link {link} ({"+".join(names)}): {len(P)} surface points, length {length * 1000:.1f} mm, grid {shape}')
    out['voxel'] = VOXEL
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('urdf', type=Path)
    p.add_argument('-o', '--output', type=Path, default=Path(__file__).resolve().parents[1] / 'config/arm_model.npz')
    a = p.parse_args()
    np.savez_compressed(a.output, **build(a.urdf))
    print(f'wrote {a.output} ({a.output.stat().st_size // 1024} KiB)')


if __name__ == '__main__':
    main()
