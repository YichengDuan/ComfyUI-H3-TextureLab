from pathlib import Path
import sys,types,importlib,importlib.util,json,tempfile,argparse
parser=argparse.ArgumentParser(description='CPU contract comparing disk/RAM raw context; no H3 inference.')
parser.add_argument('--comfy-root',type=Path,required=True);parser.add_argument('--motion-context-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
config=parser.parse_args();APP=config.comfy_root.resolve();REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP));sys.argv=['ram-context-contract','--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch,nodes,folder_paths
from comfy_extras.nodes_minimax_h3 import _empty_av_latent
up=types.ModuleType('ram_context_upstream');up.__path__=[str(config.motion_context_root.resolve())];sys.modules[up.__name__]=up
module=importlib.import_module(up.__name__+'.nodes');nodes.NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
spec=importlib.util.spec_from_file_location('ram_context_lab',REPO/'__init__.py',submodule_search_locations=[str(REPO)]);lab=importlib.util.module_from_spec(spec);sys.modules[spec.name]=lab;spec.loader.exec_module(lab)
class NoEncoding:
 def encode(self,*a,**k):raise AssertionError('Raw context unexpectedly called VAE.encode')
plan=lab.H3TextureWindowPlan.execute(56,'39','22',7)[3];latent=_empty_av_latent(32,32,39)[0];reference=torch.zeros(1,32,32,3);positive=[[torch.zeros(1,1,1),{'minimax_refs':[{'kind':'image','label':'first'},{'kind':'image','label':'second'}]}]];vae=NoEncoding()
with tempfile.TemporaryDirectory() as directory:
 folder_paths.set_output_directory(directory);outputs=[]
 for mode in ['disk','ram']:
  state=lab.H3TextureAppendWindow.execute(latent,vae,plan['windows'][0],None,'off',mode)[0]
  result=lab.H3TextureMotionGuide.execute(positive,latent,vae,reference,plan,1,state,first_frame_guide=False,preserve_target_latent=True,carry_audio=False)
  outputs.append(result[0]);assert result[4]==22;assert result[0][0][1]['minimax_refs']==positive[0][1]['minimax_refs']
 a,b=outputs[0][0][1]['minimax_keyframes'],outputs[1][0][1]['minimax_keyframes'];assert len(a)==len(b)==7
 for left,right in zip(a,b):
  assert left['resolved_frame_index']==right['resolved_frame_index'];assert torch.equal(left['latent'],right['latent'])
report={'success':True,'raw_context_keyframes':7,'trim_frames':22,'disk_ram_context_video_tensors_exactly_equal':True,'vae_encode_calls':0,'two_reference_metadata_blocks_preserved':True,'scope':'CPU contract of actual upstream Motion Context; no H3 inference or multireference generation.'};config.output.parent.mkdir(parents=True,exist_ok=True);config.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
