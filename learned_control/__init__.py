"""Run learned and tuned arm controllers exported by Isaac-hydraulic-actuator.

That project trains in simulation and exports a self-contained bundle (TorchScript actor, geometry, collision
grid, contract). Everything here runs on the robot with NumPy, PyTorch and this repository only. The modules
mirror ``hydraulic_controller`` in Isaac-hydraulic-actuator, whose ``training/test_robot_parity.py`` checks this
copy against the training code, and every bundle carries reference inputs and outputs that ``PolicyBundle``
replays at load time.
"""
