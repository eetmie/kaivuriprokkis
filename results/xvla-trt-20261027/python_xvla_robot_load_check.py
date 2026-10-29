"""Base-model load test beside real robot threads. All valve commands stay zero.
The base fixture needs 3 views: 2 live IR views plus its fixed test image.
This is explicitly not a three-camera robot policy or an excavator demo.
"""
import json,time,sys
from pathlib import Path
import numpy as np
from lerobot_vla.runtime.xvla_native import XVLANativePolicy
from lerobot_vla.excavator_robot import MasiExcavator
from lerobot_vla.camera import check_frames, fresh_observation
from lerobot_vla.tools.benchmark_xvla import rss_mib
p=XVLANativePolicy('/home/joel/vla-clean-install-20261012/bundles/xvla-base',
    cache_dir='/home/joel/vla-clean-install-20261012/runs/xvla-base/baseline-cache')
robot=MasiExcavator()
try:
    robot.connect();robot.set_pump(False);robot.stop_motion()
    robot.controller._perf_tracker.set_enabled(True)
    with np.load(p.bd.root/p.bd.b['fixture']['file']) as r:
        third=r['raw'][2].copy();noise=r['x1'].copy()
    current=robot.camera.get_latest()[1]
    robot.wait_for_next_frame(current,2.0)
    times=[];ages=[];samples=0;last=time.perf_counter();begin=last
    while time.perf_counter()-begin<30:
        keys=['observation.images.cam1','observation.images.cam2']
        obs=fresh_observation(robot.get_observation,keys)
        try:
            check_frames(obs,keys)
        except RuntimeError:
            print("CAMERA DEBUG", {k:obs[k] for k in ["img_ts","cam2_ts","state_ts"]},
                  time.perf_counter(), robot.camera._cam1.error, robot.camera._cam2.error, flush=True)
            raise
        robot.send_action(np.zeros(4,np.float32))
        state=np.zeros(20,np.float32);state[:3]=obs['observation.state']
        t0=time.perf_counter();p.sample_actions([obs[keys[0]],obs[keys[1]],third],
            p.bd.b['fixture']['task'],state,noise)
        times.append((time.perf_counter()-t0)*1000)
        ages.append([(t0-obs['img_ts'])*1000,(t0-obs['cam2_ts'])*1000,(t0-obs['state_ts'])*1000])
        samples+=1
    report={'purpose':__doc__,'status':'passed','samples':samples,'rss_mib':rss_mib(),
       'infer_ms_p50_p95_p99':np.percentile(times,[50,95,99]).tolist(),
       'observation_age_ms_p50_p99_max':np.percentile(ages,[50,99,100],axis=0).tolist(),
       'control_loop':robot.controller.get_performance_stats(),
       'torch_imported':'torch' in sys.modules,'onnxruntime_imported':'onnxruntime' in sys.modules}
    Path('results/xvla-trt-20261027/real-robot-load.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(report,indent=2,default=str),flush=True)
finally:
    robot.disconnect();p.close()
