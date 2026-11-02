"""CPU single-frame context contract using the actual Motion Context dependency."""
from pathlib import Path
import argparse, importlib, importlib.util, json, sys, tempfile, types
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--comfy-root',type=Path,required=True)
parser.add_argument('--motion-context-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
config=parser.parse_args(); repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(config.comfy_root.resolve()));sys.argv=['single-context-contract','--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch,nodes,folder_paths
from comfy_extras.nodes_minimax_h3 import _empty_av_latent
package=types.ModuleType('single_context_upstream');package.__path__=[str(config.motion_context_root.resolve())];sys.modules[package.__name__]=package
upstream=importlib.import_module(package.__name__+'.nodes');nodes.NODE_CLASS_MAPPINGS.update(upstream.NODE_CLASS_MAPPINGS)
spec=importlib.util.spec_from_file_location('single_context_lab',repo/'__init__.py',submodule_search_locations=[str(repo)])
lab=importlib.util.module_from_spec(spec);sys.modules[spec.name]=lab;spec.loader.exec_module(lab)
class FixtureVAE:
    def __init__(self):self.encoded=[]
    def decode(self,video):
        n=video.shape[2];count=n//5*17+sum([1,4,4,4,4][:n%5])
        values=((float(video[0,0,0,0,0])+torch.arange(count).float())*10).remainder(256)/255
        return values[:,None,None,None].expand(count,32,32,3).clone()
    def encode(self,frames):
        assert len(frames)==1,'Must encode exactly the last frame'
        value=float(frames[0,0,0,0]);self.encoded.append(value)
        return torch.full((1,24,1,2,2),value)
reference=torch.zeros(1,32,32,3)
positive=[[torch.zeros(1,1,1),{'minimax_refs':[{'kind':'image','label':'one'},{'kind':'image','label':'two'}]}]]
reports=[]
with tempfile.TemporaryDirectory() as directory:
    folder_paths.set_output_directory(directory)
    for size in [39,56]:
      for mode in ['disk','ram']:
       for source in ['latent','pixels_reencoded']:
        for carry_audio in [False,True]:
            plan=lab.H3TextureWindowPlan.execute(125,str(size),'1',7)[3];state=None;vae=FixtureVAE();expected=[]
            for index,entry in enumerate(plan['windows']):
                latent=_empty_av_latent(32,32,size)[0]
                video,audio=latent['samples'].unbind();video.fill_(index*1000)
                audio.copy_(torch.arange(audio.shape[-1]).expand_as(audio));latent['noise_mask']='source-mask'
                result=lab.H3TextureMotionGuide.execute(positive,latent,vae,reference,plan,index,state,source,
                    first_frame_guide=False,preserve_target_latent=True,carry_audio=carry_audio)
                assert result[1] is latent
                assert result[4]==(1 if index else 0) and result[5]==bool(index)
                assert result[0][0][1]['minimax_refs']==positive[0][1]['minimax_refs']
                if index:
                    keys=result[0][0][1]['minimax_keyframes'];visual=[k for k in keys if 'latent' in k]
                    assert len(visual)==1 and visual[0]['resolved_frame_index']==0
                    assert visual[0]['latent'].shape[2]==1
                    assert abs(float(visual[0]['latent'][0,0,0,0,0])-(((index-1)*1000+size-1)*10%256)/255)<1.01/255, (size,mode,source,index,float(visual[0]['latent'][0,0,0,0,0]),vae.encoded)
                    sounds=[k for k in keys if 'audio_latent' in k]
                    assert len(sounds)==int(carry_audio)
                    if carry_audio:
                        previous_av=lab.context_latent(state)
                        tail,steps,overhang=upstream._audio_tail_from_latent(previous_av,18)
                        assert torch.equal(sounds[0]['audio_latent'],tail)
                        assert sounds[0]['resolved_frame_index']==round(upstream.FRAME_RESCALE+overhang)/upstream.FRAME_RESCALE-steps/upstream.FRAME_RESCALE
                state=lab.H3TextureAppendWindow.execute(latent,vae,entry,state,'off',mode)[0]
                trim=entry['discarded_overlap'];count=entry['source_end_exclusive']-entry['source_start']-trim
                expected.extend(range(index*1000+trim,index*1000+trim+count))
            actual=torch.stack([frame[0,0,0] for frame in state['frames'].iter_frames()])
            assert torch.allclose(actual,(torch.tensor(expected)*10).remainder(256).float()/255) and len(actual)==125
            assert len(vae.encoded)==len(plan['windows'])-1
            reports.append({'window_frames':size,'cache_mode':mode,'context_source':source,'carry_audio':carry_audio,
                'windows':len(plan['windows']),'delivered_frames':len(actual),'single_frame_encode_calls':len(vae.encoded)})
report={'success':True,'cases':reports,'single_frame_trim_and_last_frame_identity':True,
    'original_references_source_latent_and_mask_preserved':True,'audio_tail_and_grid_alignment_verified':True,
    'scope':'Native CPU contract with synthetic VAE fixtures. Single-frame mode decodes the previous window and re-encodes its last frame; no H3 inference or quality claim.'}
config.output.parent.mkdir(parents=True,exist_ok=True);config.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
