"""Streaming source alignment and compact-mask contracts on synthetic clips (CPU)."""
from pathlib import Path
import argparse,importlib,importlib.util,json,os,sys,types,tempfile,subprocess
from unittest.mock import patch
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--comfy-root',type=Path,required=True)
parser.add_argument('--relay-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
config=parser.parse_args();repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(config.comfy_root.resolve()));sys.argv=['stream-contract','--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch,nodes
from comfy_api.latest._input_impl.video_types import VideoFromFile
from comfy_extras.nodes_sam3 import SAM3_VideoTrack
from comfy.ldm.sam3.tracker import pack_masks
root=types.ModuleType('stream_contract_relay');root.__path__=[str(config.relay_root.resolve()/'h3_relay')];sys.modules[root.__name__]=root
toolkit=importlib.import_module(root.__name__+'.toolkit');masks=importlib.import_module(root.__name__+'.mask_tools')
nodes.NODE_CLASS_MAPPINGS['H3RelayPrepareSource']=toolkit.H3RelayPrepareSource
nodes.NODE_CLASS_MAPPINGS['H3RelaySelectMask']=masks.SelectSubject
spec=importlib.util.spec_from_file_location('stream_contract_lab',repo/'streaming.py');stream=importlib.util.module_from_spec(spec);sys.modules[spec.name]=stream;spec.loader.exec_module(stream)
spec=importlib.util.spec_from_file_location('stream_contract_plan',repo/'planning.py');planning=importlib.util.module_from_spec(spec);spec.loader.exec_module(planning)
spec=importlib.util.spec_from_file_location('stream_contract_window',repo/'source_window.py');window=importlib.util.module_from_spec(spec);spec.loader.exec_module(window)
torch.set_num_threads(2)
class FakeSAM:
 def __init__(self):self.tracker=types.SimpleNamespace(track_video_with_detection=True);self.calls=0;self.requests=[]
 def forward_video(self,images,initial_masks,pbar=None,**kwargs):
  self.calls+=1;result=[]
  for index in range(images.shape[0]):
   if index+1<len(images):images[index+1:index+2]  # native prefetch/lookback ordering
   frame=images[index:index+1];self.requests.append(index)
   pixels=(frame[:,0]>.25).float()*2-1
   result.append(pack_masks(pixels))
  return {'packed_masks':torch.stack(result),'n_frames':len(images),'scores':[1.]}
class FakeModel:
 def __init__(self,sam):self.model=types.SimpleNamespace(diffusion_model=sam,get_dtype=lambda:torch.float32)
records=[]
with tempfile.TemporaryDirectory() as temp:
 temp=Path(temp)
 source=temp/'cfr30.mp4'
 subprocess.run([stream.tools('ffmpeg'),'-hide_banner','-loglevel','error','-f','lavfi','-i','testsrc2=size=96x64:rate=30:duration=2.4','-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=2.4','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-y',str(source)],check=True)
 variable=temp/'vfr.mp4'
 subprocess.run([stream.tools('ffmpeg'),'-hide_banner','-loglevel','error','-i',str(source),'-vf',"select='not(eq(mod(n,4),1))'",'-fps_mode','vfr','-c:v','libx264','-c:a','copy','-y',str(variable)],check=True)
 for file in [source,variable]:
  for duration in [0.,.5]:
   video=VideoFromFile(str(file));legacy=toolkit.H3RelayPrepareSource().prepare(video,'480',duration,'silence')
   streamed=stream.prepare_source(video,'480',duration,'silence');specification,audio,w,h,length,count=streamed[:6]
   assert (w,h,length,count)==tuple(legacy[2:6]),(file.name,duration,(w,h,length,count),legacy[2:6])
   with stream.FrameSequence(specification,3) as reader:
    # Native prefetch asks for the next frame before reusing frame 0.
    if count>1:reader[1:2]
    for i in range(count):
     actual=reader[i:i+1].movedim(1,-1)
     assert torch.equal(actual,legacy[0][i:i+1]),(file.name,duration,i,(actual-legacy[0][i:i+1]).abs().max().item(),actual.mean((0,1,2)).tolist(),legacy[0][i:i+1].mean((0,1,2)).tolist(),actual[0,0,:3].tolist(),legacy[0][i,0,:3].tolist())
    assert reader.peak_cache_bytes<=3*w*h*4
   assert torch.equal(streamed[8],legacy[0][:1])
   assert audio['sample_rate']==legacy[1]['sample_rate']
   audio_common=min(audio['waveform'].shape[-1],legacy[1]['waveform'].shape[-1])
   assert torch.equal(audio['waveform'][...,:audio_common],legacy[1]['waveform'][...,:audio_common])
   fake=FakeSAM();model=FakeModel(fake);initial=torch.ones(1,h,w)
   with patch('comfy.model_management.load_model_gpu'),patch('comfy.model_management.get_torch_device',return_value=torch.device('cpu')):
    baseline=SAM3_VideoTrack.execute(legacy[0][:count],model,initial_mask=initial,conditioning=None,max_objects=1)
    original=baseline[0];fake.requests=[];fake.calls=0
    tracked,report=stream.track_source(specification,model,initial,None,{'selection':'largest','index':0},recover_empty_frames=False)
   assert fake.calls==1 and fake.requests==list(range(count))
   original['orig_size']=(h,w)
   from comfy_extras.nodes_sam3 import SAM3_TrackToMask
   raw=SAM3_TrackToMask.execute(original)[0]
   assert torch.equal(torch.from_numpy(__import__('numpy').unpackbits(tracked.masks,axis=-1,bitorder='little')).bool(),raw>.5)
   expanded=stream.expand_source(tracked,3)
   expected=masks.expand_masks(raw.numpy(),3)
   assert torch.equal(torch.from_numpy(__import__('numpy').unpackbits(expanded.masks,axis=-1,bitorder='little')).float(),torch.from_numpy(expected))
   for context in [0,1,5,22]:
    plan={'windows':planning.window_plan(count,39,context,7)}
    # This fixture is short; the long native loop/GPU test covers overlap.
    for index in range(len(plan['windows'])):
     a=stream.source_window(expanded,plan,index,audio)
     b=window.H3TextureSourceWindow.execute(legacy[0],torch.from_numpy(expected),plan,index,audio)
     assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]);assert torch.equal(a[2]['waveform'],b[2]['waveform'])
   records.append({'fixture':file.name,'duration':duration,'frames':count,'width':w,'height':h,'rgb_exact':True,'audio_common_samples_exact':True,'stream_audio_samples':audio['waveform'].shape[-1],'legacy_audio_samples':legacy[1]['waveform'].shape[-1],'packed_masks_match_native_threshold':True,'expansion_exact':True,'native_wrapper_contract_calls':fake.calls,'scope':'Synthetic frame provider/model contract; no SAM inference'})
config.output.parent.mkdir(parents=True,exist_ok=True);config.output.write_text(json.dumps({'success':True,'cases':records},indent=2)+'\n');print(json.dumps({'success':True,'cases':records},indent=2))
