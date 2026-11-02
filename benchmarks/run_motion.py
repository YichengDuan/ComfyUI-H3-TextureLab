"""Short real-motion regression, with native reference encoding and per-stage records."""
from pathlib import Path
import argparse,sys,types,importlib,json,time,gc,hashlib,platform,resource
parser=argparse.ArgumentParser()
parser.add_argument('--comfy-root',type=Path,required=True)
parser.add_argument('--motion-context-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--size',type=int,choices=[480,768],default=480)
parser.add_argument('--modes',nargs='+',choices=['raw','refine_join'],default=['raw','refine_join'])
config=parser.parse_args();REPO=Path(__file__).resolve().parents[1];APP=config.comfy_root.resolve();OUT=config.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(APP));sys.argv=['h3-motion-regression','--use-sage-attention','--disable-pinned-memory','--reserve-vram','1']
import torch,numpy as np
from PIL import Image
import comfy.options
comfy.options.enable_args_parsing()
import comfy.sd,comfy.sample,comfy.samplers,comfy.utils,comfy.model_management,comfy.nested_tensor
import nodes,folder_paths
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo,MiniMaxH3SigmaShift,_empty_av_latent
from safetensors.torch import save_file,load_file
def load_package(name,path,initialize=False):
 if initialize:
  spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)]);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 else:module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
 return module
load_package('motion_regression_upstream',config.motion_context_root.resolve())
for name in ['nodes','join_match','refine_blend']:
 module=importlib.import_module('motion_regression_upstream.'+name);nodes.NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
lab=load_package('motion_regression_lab',REPO,True);nodes.NODE_CLASS_MAPPINGS.update(lab.NODE_CLASS_MAPPINGS)
storage=importlib.import_module('motion_regression_lab.storage')
assert torch.cuda.device_count()==1,'Select one GPU with CUDA_VISIBLE_DEVICES'
torch.set_num_threads(4);width,height=(480,736) if config.size==480 else (768,1152)
models=json.loads((REPO/'configs/models.json').read_text());seed=41729
prompt="""subject_definitions:
<Picture 1> is the only reference. Preserve the same adult woman's facial identity, short dark brown bob hair, natural freckles and skin texture, blue collared long-sleeved shirt, beige trousers and white sneakers.
summary:
A continuous realistic full-body studio video. The woman gently walks in place with clear alternating steps, natural arm swing and subtle coordinated torso turns.
detailed_description:
The camera stays fixed at eye level and shows her complete head, both hands and both feet throughout. She repeatedly lifts one heel, takes a small relaxed step in place, transfers her weight and alternates to the other leg. Her arms swing naturally opposite her legs; her shoulders and torso turn gently left and right with the stepping rhythm. The motion is clearly visible, balanced and anatomically natural. Keep the same body proportions, face, clothing weave, buttons, seams and wrinkles. The neutral light gray studio background, soft even lighting, exposure and white balance remain stable. One continuous take with no camera cuts or zooms.
overall_soundscape:
Soft natural sneaker footsteps on the studio floor. No speech or music."""
reference_path=OUT/'reference.png'
if not reference_path.exists():reference_path.write_bytes(config.reference.read_bytes())
reference=torch.from_numpy(np.asarray(Image.open(reference_path).convert('RGB')).copy()).float().unsqueeze(0)/255
fingerprint=hashlib.sha256(reference_path.read_bytes()+prompt.encode()+str((width,height)).encode()).hexdigest()
condition_path=OUT/('conditioning-'+fingerprint[:16]+'.pt')
def cpu(value):
 if isinstance(value,torch.Tensor):return value.detach().cpu()
 if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
 if isinstance(value,list):return [cpu(v) for v in value]
 if isinstance(value,tuple):return tuple(cpu(v) for v in value)
 return value
def write(path,data):path.write_text(json.dumps(data,indent=2,default=str)+'\n')
environment={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'models':models,'reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),'prompt':prompt,'width':width,'height':height,'seed':seed,'steps':8,'sampler':'er_sde','scheduler':'simple','low_vram_policy':'native PREPARE_SAMPLING force_offload=True','window_frames':56,'context_frames':22,'frames':158,'fps':24}
environment['source_sha256']={str(path.relative_to(REPO)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [REPO/'__init__.py',REPO/'storage.py',REPO/'low_vram.py',REPO/'performance.py',Path(__file__)]}
write(OUT/'configuration.json',environment)
with torch.inference_mode():
 vae=comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(APP/'models/vae'/models['video_vae'])))
 if not condition_path.exists():
  print('Encoding one full-body reference with native Ref2V',flush=True);comfy.model_management.unload_all_models();clip=comfy.sd.load_clip([str(APP/'models/text_encoders'/models['text_encoder'])],clip_type=comfy.sd.CLIPType.MINIMAX)
  torch.cuda.reset_peak_memory_stats();started=time.perf_counter();positive=MiniMaxH3ReferenceToVideo.execute(clip,prompt,width,height,56,vae=vae,ref_images={'ref_image_0':reference})[0]
  torch.save(cpu(positive),condition_path);write(OUT/'conditioning-stage.json',{'seconds':time.perf_counter()-started,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20});del clip,positive;comfy.model_management.unload_all_models();comfy.model_management.current_loaded_models.clear();gc.collect();torch.cuda.empty_cache()
 positive=torch.load(condition_path,weights_only=False,map_location='cpu')
 model=comfy.sd.load_diffusion_model(str(APP/'models/diffusion_models'/models['diffusion']));model=nodes.LoraLoaderModelOnly().load_lora_model_only(model,models['lora'],1.)[0];model=MiniMaxH3SigmaShift.execute(model,12.,3.)[0];model=lab.H3TextureLowVRAM.execute(model,True)[0]
 plan=lab.H3TextureWindowPlan.execute(158,'56','22',seed)[3]
 for mode in config.modes:
  target=OUT/mode;target.mkdir(exist_ok=True);folder_paths.set_output_directory(str(target));report_path=target/'progress.json';report=json.loads(report_path.read_text()) if report_path.exists() else {'mode':mode,**environment,'runs':[]}
  if report.get('stage')=='complete':continue
  state=None;completed={r['window'] for r in report['runs']}
  if report.get('cache'):
   c=report['cache'];manifest=json.loads((Path(c['root'])/'manifest.json').read_text());state={k:v for k,v in manifest.items() if k not in ['chunks','shape','fps']};state['frames']=storage.LatentTimeline(c['root'],c['chunks'],c['shape'],vae)
  for entry in plan['windows']:
   index=entry['index']
   if index in completed:continue
   print(f'MOTION {width}x{height} {mode} window {index+1}/4',flush=True);window_started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
   master=_empty_av_latent(width,height,56)[0];guided,latent,window,window_seed,trim,*_=lab.H3TextureMotionGuide.execute(positive,master,vae,reference,plan,index,state)
   guide_seconds=time.perf_counter()-window_started;guide_peak=torch.cuda.max_memory_allocated()/2**20;shared=OUT/'shared-first-av.safetensors';torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
   if index==0 and shared.exists():
    data=load_file(str(shared),device='cpu');sampled={'samples':comfy.nested_tensor.NestedTensor((data['video'],data['audio']))};del data
   else:
    noise=comfy.sample.prepare_noise(latent['samples'],seed);torch.manual_seed(seed);result=comfy.sample.sample(model,noise,8,1.,'er_sde','simple',guided,guided,latent['samples'],seed=seed);sampled={'samples':result};del result,noise
    if index==0:
     video,audio=list(sampled['samples'].unbind());save_file({'video':video.cpu().contiguous(),'audio':audio.cpu().contiguous()},str(shared));del video,audio
   sampling_seconds=time.perf_counter()-started;sampling_peak=torch.cuda.max_memory_allocated()/2**20;refine_seconds=0.;torch.cuda.reset_peak_memory_stats()
   if mode=='refine_join' and index:
    started=time.perf_counter();sigmas=comfy.samplers.calculate_sigmas(model.get_model_object('model_sampling'),'simple',8)[-2:];noise=comfy.sample.prepare_noise(sampled['samples'],seed+1);torch.manual_seed(seed+1)
    refined=comfy.sample.sample_custom(model,noise,1.,comfy.samplers.ksampler('er_sde'),sigmas,positive,positive,sampled['samples'],seed=seed+1)
    sampled=nodes.NODE_CLASS_MAPPINGS['MiniMaxH3RefineBlend']().apply(sampled,22,5,'original',True,clip_index=index,refined={'samples':refined})[0];refine_seconds=time.perf_counter()-started;del refined,noise
   refine_peak=torch.cuda.max_memory_allocated()/2**20;state=lab.H3TextureAppendWindow.execute(sampled,vae,window,state,'join_match' if mode=='refine_join' else 'off')[0]
   record={'window':index,'trim_frames':trim,'delivered_frames':len(state['frames']),'guide_seconds':guide_seconds,'sampling_seconds':sampling_seconds,'refine_seconds':refine_seconds,'total_seconds':time.perf_counter()-window_started,'guide_peak_mib':guide_peak,'sampling_peak_mib':sampling_peak,'refine_peak_mib':refine_peak,'process_max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
   report['runs'].append(record);report['cache']={'root':str(state['frames'].root),'chunks':state['frames'].chunks,'shape':list(state['frames'].shape),'av_latent_file':state['av_latent_file']};write(report_path,report);print(json.dumps(record),flush=True);del sampled,master,latent,guided;gc.collect()
  report['stage']='exporting';write(report_path,report);torch.cuda.reset_peak_memory_stats();started=time.perf_counter();output=lab.H3TextureFinishVideo.execute(state,16);path=target/f'motion-{width}x{height}-{mode}.mp4';output[0].save_to(str(path));report.update(stage='complete',video=str(path),export_seconds=time.perf_counter()-started,export_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20);write(report_path,report);print('MOTION COMPLETE '+str(path),flush=True);del state,output;gc.collect()
