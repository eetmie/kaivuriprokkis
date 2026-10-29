"""Analytic GPU regression: 2 IR-like inputs, auto action mode, horizons 30/50.
Not a trained policy; proves shape handling and physical processor boundaries.
"""
from pathlib import Path
import json, shutil
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from lerobot_vla.runtime.vendor.xvla_trt import Bundle
from lerobot_vla.runtime.xvla_native import XVLANativePolicy, physical_state, physical_actions

base=Path('results/xvla-trt-20261027/analytic-two-view').resolve();base.mkdir(parents=True,exist_ok=True)
reports=[]

def tensor(name,shape,dtype=T.FLOAT):return h.make_tensor_value_info(name,dtype,shape)
def save(root,name,inputs,outputs,nodes,constants):
    graph=h.make_graph(nodes,name,inputs,outputs,[nh.from_array(np.asarray(a),k) for k,a in constants.items()])
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);model.ir_version=10
    onnx.checker.check_model(model);onnx.save(model,root/f'{name}.onnx')
def node(op,ins,out,**kw):return h.make_node(op,ins,[out],**kw)

def reference(pixels,ids,proprio,noise,steps):
    vis=np.full((2,2,4),pixels.mean(),np.float32)
    full=np.zeros((3,2,4),np.float32);full[:2]=vis
    text=full[0:1]+ids.astype(np.float32).mean()
    cond=text+full[1:].mean()
    act=np.zeros_like(noise)
    for i in range(steps,0,-1):
        t=np.float32(i/steps)
        xt=noise*t+act*np.float32(1-t)
        act=xt*np.float32(.5)+proprio.mean()*np.float32(.1)+cond.mean()*np.float32(.01)+t*np.float32(.2)
    return act

for horizon in [30,50]:
    root=base/f'horizon-{horizon}';root.mkdir(exist_ok=True)
    token=root/'tokenizer'
    if not token.exists():token.symlink_to('/home/joel/vla-clean-install-20261012/bundles/xvla-base/tokenizer')
    graphs=[{'name':n,'file':n+'.onnx'} for n in ['vision_0','text_encoder_0','cond','denoise_0','denoise_1']]
    contract={'version':1,'physical_boundary_complete':True,
        'state':{'dim':3,'model_dim':20,'normalization':{'mode':'MEAN_STD','mean':[10,20,30],'std':[2,4,5]}},
        'action':{'dim':4,'model_dim':20,'normalization':{'mode':'MEAN_STD','mean':[.1,.2,.3,.4],'std':[.2,.3,.4,.5]}}}
    b={'graphs':graphs,'schema_version':1,'tokens_per_view':2,'valid_views':2,'num_image_views':3,
       'lang_len':5,'chunk_size':horizon,'max_state_dim':20,'max_action_dim':20,'num_denoising_steps':2,
       'action_mode':'auto','tokenizer':{'path':'tokenizer'},'fixture':{'file':'fixture.npz','task':'dig',
       'source':'analytic NumPy graph; not trained robot policy'},'processor_contract':contract}
    (root/'bundle.json').write_text(json.dumps(b,indent=2))
    save(root,'vision_0',[tensor('pixel_values',[2,3,224,224])],[tensor('hidden_out',[2,2,4])],
        [node('ReduceMean',['pixel_values'],'m',axes=[0,1,2,3],keepdims=1),node('Reshape',['m','flat'],'r'),
         node('Expand',['r','shape'],'hidden_out')],{'flat':np.array([1,1,1],np.int64),'shape':np.array([2,2,4],np.int64)})
    save(root,'text_encoder_0',[tensor('input_ids',[1,5],T.INT64),tensor('image_tokens',[1,2,4])],
        [tensor('hidden_out',[1,2,4])],[node('Cast',['input_ids'],'f',to=T.FLOAT),
         node('ReduceMean',['f'],'m',axes=[0,1],keepdims=1),node('Reshape',['m','flat'],'r'),
         node('Add',['image_tokens','r'],'hidden_out')],{'flat':np.array([1,1,1],np.int64)})
    save(root,'cond',[tensor('vlm_features',[1,2,4]),tensor('aux_visual',[1,4,4])],
         [tensor('cond_tokens',[1,2,4])],[node('ReduceMean',['aux_visual'],'m',axes=[0,1,2],keepdims=1),
          node('Add',['vlm_features','m'],'cond_tokens')],{})
    shape=[1,horizon,20]
    save(root,'denoise_0',[tensor('x_t',shape),tensor('t',[1]),tensor('proprio',[1,20]),tensor('cond_tokens',[1,2,4])],
         [tensor('hidden_out',shape)], [node('Mul',['x_t','half'],'a'),
          node('ReduceMean',['proprio'],'ps',axes=[0,1],keepdims=1),node('Reshape',['ps','flat'],'p'),
          node('Mul',['p','point1'],'pp'),node('ReduceMean',['cond_tokens'],'c',axes=[0,1,2],keepdims=1),
          node('Mul',['c','point01'],'cc'),node('Reshape',['t','flat'],'tt'),node('Mul',['tt','point2'],'ts'),
          node('Add',['a','pp'],'ap'),node('Add',['ap','cc'],'ac'),node('Add',['ac','ts'],'hidden_out')],
         {'half':np.array(.5,np.float32),'point1':np.array(.1,np.float32),'point01':np.array(.01,np.float32),
          'point2':np.array(.2,np.float32),'flat':np.array([1,1,1],np.int64)})
    save(root,'denoise_1',[tensor('hidden_in',shape)],[tensor('action',shape)],
         [node('Identity',['hidden_in'],'action')],{})
    rng=np.random.default_rng(1)
    raw=rng.integers(0,256,(2,480,640,3),dtype=np.uint8)
    bd=Bundle(root);pixels=np.stack([bd.preprocess(r) for r in raw]);ids=bd.input_ids('dig')
    state=np.array([12,28,45],np.float32);proprio=physical_state(state,contract,20)
    noise=rng.standard_normal(shape).astype(np.float32)
    ref=reference(pixels,ids,proprio,noise,2)
    np.savez(root/'fixture.npz',raw=raw,pixel_values=pixels,input_ids=ids,proprio=proprio,x1=noise,action_pred=ref)
    p=XVLANativePolicy(root,chain='graph')
    try:
        actual=p.sample_actions(list(raw),'dig',state,noise)
        expected=physical_actions(ref,contract)[0]
        np.testing.assert_allclose(actual,expected,rtol=3e-4,atol=3e-4)
        assert actual.shape==(horizon,4)
        reports.append({'views':2,'horizon':horizon,'mode':'auto','chunk':list(actual.shape),
                        'max_physical_error':float(np.abs(actual-expected).max()),'fixture':p.fixture_parity})
    finally:p.close()
(base/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
print('TWO-VIEW AUTO GPU PARITY PASSED',json.dumps(reports,indent=2))
