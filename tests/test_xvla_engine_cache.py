"""Direct TensorRT cache identity and corruption checks without a GPU build."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lerobot_vla.runtime.vendor import trt_engines as runtime


class EngineCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'stage.onnx').write_bytes(b'graph-v1')
        self.bundle=SimpleNamespace(root=self.root,names=['stage'])
        self.cache=self.root/'cache'
        self.calls=[]
        def build(argv,**kwargs):
            self.calls.append(argv)
            Path(argv[4]).write_bytes(b'engine')
        for name,value in [('_trt_version',lambda:'10.16'),('_cudart_version',lambda:13020)]:
            patcher=patch.object(runtime,name,value);patcher.start();self.addCleanup(patcher.stop)
        patcher=patch.object(runtime.subprocess,'run',build);patcher.start();self.addCleanup(patcher.stop)

    def test_cached_bytes_are_reused(self):
        self.assertIn('stage',runtime.prebuild_engines(self.bundle,self.cache))
        self.assertEqual(runtime.prebuild_engines(self.bundle,self.cache),{})
        self.assertEqual(len(self.calls),1)

    def test_changed_graph_invalidates_engine(self):
        runtime.prebuild_engines(self.bundle,self.cache)
        (self.root/'stage.onnx').write_bytes(b'graph-v2')
        self.assertIn('stage',runtime.prebuild_engines(self.bundle,self.cache))
        self.assertEqual(len(self.calls),2)

    def test_changed_builder_precision_invalidates_engine(self):
        runtime.prebuild_engines(self.bundle,self.cache,accumulate='fp32')
        self.assertIn('stage',runtime.prebuild_engines(self.bundle,self.cache,accumulate='auto'))

    def test_engine_removed_after_reboot_is_rebuilt(self):
        runtime.prebuild_engines(self.bundle,self.cache)
        (self.cache/'stage.engine').unlink()
        self.assertIn('stage',runtime.prebuild_engines(self.bundle,self.cache))

    def test_corrupted_engine_refused(self):
        runtime.prebuild_engines(self.bundle,self.cache)
        (self.cache/'stage.engine').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError,'--rebuild'):
            runtime.prebuild_engines(self.bundle,self.cache)


if __name__=='__main__':unittest.main()
