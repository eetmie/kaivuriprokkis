"""Regenerate numeric fixtures from the Python waveform classes, without hardware imports.

Run manually from any directory. Native CTest only reads the frozen JSON and
does not need NumPy or the legacy collector. Parameters/directions are explicit:
NumPy and the native version 4 generator intentionally use different RNGs.
"""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    reference = Path(__file__).resolve().parents[2]/'simple_drive.py'
    names = {'SineExcitationGenerator', 'ChirpExcitationGenerator', 'StrokeExcitationGenerator',
             'sine_target_modes'}
    constants = {'SINE_TARGET_MODES', 'SINE_SLEW_MODE'}
    nodes = [node for node in ast.parse(reference.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names or
             isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)]
    namespace = {'np': np, 'JOINT_NAMES': ['slew','boom','arm','bucket']}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reference), 'exec'), namespace)
    limits = {'boom': [-50,30], 'arm': [28,144], 'bucket': [-120,20]}
    cases = []
    for name in ('sine', 'fast_sine', 'chirp', 'fast_chirp', 'constant_chirp', 'stroke'):
        mode = {'fast_sine': 'sine', 'fast_chirp': 'chirp'}.get(name, name)
        amplitude = .65 if mode == 'sine' else .4 if mode == 'stroke' else .35
        cls = namespace[{'sine':'SineExcitationGenerator', 'stroke':'StrokeExcitationGenerator'}.get(mode, 'ChirpExcitationGenerator')]
        options = {'enabled': True, 'seed': 42, 'amplitude': amplitude}
        start, end, sweep = (.5,12,20) if mode == 'stroke' else (.05,.9,60)
        if name == 'fast_chirp': end = 12
        if mode == 'constant_chirp': start = end = .3
        if mode != 'sine': options.update(start_hz=start, end_hz=end, sweep_s=sweep)
        if mode == 'stroke': options['joint_limits_deg'] = limits
        generator = cls(**options)
        parameters = dict(f_env=.025, f_car=.4, f_rate=.04, depth=.8, f_noise=.25,
                          noise=0., amp=amplitude, phi1=.2, phi2=.3)
        if name == 'fast_sine': parameters['f_car'] = 8.
        generator._params = {name: dict(parameters) for name in namespace['JOINT_NAMES']}
        directions = [1,-1,-1,-1]  # Explicit native seed 42 / block 2 inputs.
        if mode == 'stroke':
            generator._direction = dict(zip(('boom','arm','bucket'), directions[1:]))
            times = np.arange(2502)*.01
        else:
            times = [0,.01,.2,.9,1,30,59,60,61,90,119,119.8,120,125,129.99,130,130.2]
        samples = []
        for index, t in enumerate(times):
            fresh = not (mode == 'stroke' and 1000 <= index < 1010)
            q = [0., -39., 29., -119.]
            if mode == 'stroke':
                if index == len(times)-1: q[1] = 31.
                generator.observe(q if fresh else None)
            u = generator.get_all(float(t))
            samples.append([float(t), fresh, q, [u[name] for name in namespace['JOINT_NAMES']], generator.enabled])
        cases.append(dict(name=name, mode=mode, amplitude=amplitude, start_hz=start, end_hz=end,
                          sweep_s=sweep, directions=directions, parameters=parameters, samples=samples))
    fixture = dict(reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                   reference='simple_drive.py waveform classes', numpy_version=np.__version__, cases=cases)
    target = Path(__file__).parent/'fixtures/mlp_excitation.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(fixture, separators=(',', ':'))+'\n')
    print(target)


if __name__ == '__main__':
    main()
