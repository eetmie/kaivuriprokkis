"""Observation-only inference scaffolding beside the native services.

Capture has no GPU dependencies. The optional GR00T fixture adapter loads one
verified prebuilt TensorRT cache in a disposable process; it never sends actions.
"""
