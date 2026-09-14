"""X-VLA engine-cache manifest: a boot-cleared cache rebuilds, a mixed one refuses.

clear-build-caches deletes *.engine at every boot and keeps *.timing. The manifest
left behind used to fail validation, so the first run after any reboot (or power
cut) died instead of rebuilding.
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lerobot_vla.runtime.vendor import xvla_split_ort as ort_mod  # noqa: E402

IDENTITY = {"version": 1, "precision": "fp16"}
ENGINE = "TensorrtExecutionProvider_TRTKernel_graph_main_graph_1_0_0_fp16_sm87.engine"
TIMING = "TensorrtExecutionProvider_cache_sm87.timing"


def _built_cache(root: Path) -> Path:
    (root / ENGINE).write_bytes(b"engine")
    (root / TIMING).write_bytes(b"timing")
    (root / ort_mod._ENGINE_CACHE_MANIFEST).write_text(json.dumps(
        {"identity": IDENTITY, "files": ort_mod._engine_cache_files(root)}))
    return root


class EngineCacheManifestTest(unittest.TestCase):
    def test_intact_cache_is_a_hit(self):
        with TemporaryDirectory() as d:
            cache = _built_cache(Path(d))
            self.assertIsNotNone(ort_mod._validate_engine_cache_manifest(cache, IDENTITY))

    def test_engines_cleared_at_boot_rebuilds_and_keeps_timing(self):
        with TemporaryDirectory() as d:
            cache = _built_cache(Path(d))
            (cache / ENGINE).unlink()
            self.assertIsNone(ort_mod._validate_engine_cache_manifest(cache, IDENTITY))
            self.assertFalse((cache / ort_mod._ENGINE_CACHE_MANIFEST).exists())
            self.assertTrue((cache / TIMING).exists())

    def test_truncated_engine_still_refuses(self):
        with TemporaryDirectory() as d:
            cache = _built_cache(Path(d))
            (cache / ENGINE).write_bytes(b"eng")
            with self.assertRaisesRegex(ValueError, "--rebuild"):
                ort_mod._validate_engine_cache_manifest(cache, IDENTITY)

    def test_rebuild_clear_drops_timing_but_not_unrelated_files(self):
        with TemporaryDirectory() as d:
            cache = _built_cache(Path(d))
            (cache / "notes.txt").write_text("keep")
            ort_mod.clear_engine_cache(cache)
            self.assertEqual(sorted(p.name for p in cache.iterdir()), ["notes.txt"])


if __name__ == "__main__":
    unittest.main()
