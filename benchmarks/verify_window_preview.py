"""Synthetic half-size MP4 and RAM-preview HTTP range contracts (CPU)."""
from pathlib import Path
import sys,argparse,importlib.util,asyncio,types,json,io
from unittest.mock import patch
parser=argparse.ArgumentParser();parser.add_argument('--comfy-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);config=parser.parse_args()
sys.path.insert(0,str(config.comfy_root.resolve()));sys.argv=['preview-contract','--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch,av
from aiohttp import web
from aiohttp.test_utils import TestClient,TestServer
from server import PromptServer
repo=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('window_preview_contract',repo/'window_preview.py');preview=importlib.util.module_from_spec(spec);spec.loader.exec_module(preview)
torch.set_num_threads(2);torch.manual_seed(91)
from comfy.latent_formats import MiniMaxH3Video
clock=torch.zeros(1,24,6,1,1)
clock[:,0,:,0,0]=torch.arange(6)
frames=preview.latent2rgb_frames(clock,18)
expected_indices=torch.tensor([0]+[1]*4+[2]*4+[3]*4+[4]*4+[5])
fmt=MiniMaxH3Video()
expected=(expected_indices[:,None]*torch.tensor(fmt.latent_rgb_factors[0])+torch.tensor(fmt.latent_rgb_factors_bias)+1)/2
assert torch.allclose(frames[:,0,0],expected.clamp(0,1))
class VAE:
 def decode(self,video):raise AssertionError('Preview must not decode the VAE')
latent={'samples':torch.zeros(1,1,24,12,4,6)}
# unbind()[0] matches the stream shape interface used by the actual AV type.
images=torch.rand(39,64,96,3);mask=torch.zeros(39,64,96);mask[:,10:50,20:70]=1
entry={'index':1,'discarded_overlap':22,'source_start':17,'source_end_exclusive':56}
result=preview.H3TextureWindowPreview().preview(latent,VAE(),images,mask,entry,2,True,True,.5)
assert result['result'][0] is latent
info=result['ui']['h3_texture_window_preview'][0];token=next(iter(preview._PREVIEWS));data=dict(preview._PREVIEWS[token]);assert set(data)=={'source','generated'}
for name,body in data.items():
 with av.open(io.BytesIO(body)) as container:
  frames=list(container.decode(video=0));assert len(frames)==17 and (frames[0].width,frames[0].height)==(48,32)
assert preview.H3TextureWindowPreview().preview(latent,VAE(),images,mask,entry,2,False,False,.5)==(latent,)
short={**entry,'source_end_exclusive':53}
short_result=preview.H3TextureWindowPreview().preview(latent,VAE(),images,mask,short,2,True,True,.5)
assert all(value['frames']==14 for value in short_result['ui']['h3_texture_window_preview'][0]['details'].values())
# Restore the original token for the HTTP checks.
with preview._LOCK:
 preview._PREVIEWS.clear();preview._PREVIEWS[token]=data
class StoredFrames:
 def load_av(self,name):
  assert name=='corrected-window'
  return {'video':torch.full((1,24,12,4,6),.75),'audio':torch.zeros(1,32,2,65)}
class StoredVAE(VAE):pass
state={'frames':StoredFrames(),'av_latent_file':'corrected-window'}
stored=preview.H3TextureStoredWindowPreview().preview(state,StoredVAE(),images,mask,entry,2,True,True,.5)
assert stored['result'][0] is state
with preview._LOCK:
 preview._PREVIEWS.clear();preview._PREVIEWS[token]=data
async def main():
 fake=types.SimpleNamespace(routes=web.RouteTableDef())
 with patch.object(PromptServer,'instance',fake,create=True):preview.register_preview_routes()
 app=web.Application();app.add_routes(fake.routes)
 async with TestClient(TestServer(app)) as client:
  path=info['urls']['source'];response=await client.get(path);assert response.status==200 and await response.read()==data['source'];assert response.headers['Cache-Control']=='no-store'
  response=await client.get(path,headers={'Range':'bytes=0-9'});assert response.status==206 and await response.read()==data['source'][:10]
  response=await client.get(path,headers={'Range':'bytes=-10'});assert response.status==206 and await response.read()==data['source'][-10:]
  response=await client.get(path,headers={'Range':'bytes=999999999-'});assert response.status==416
  response=await client.get('/h3_texture/window_preview/missing/source');assert response.status==404
  preview.publish(data['source'],data['generated']);assert len(preview._PREVIEWS)==1
  response=await client.get(path);assert response.status==404
 report={'success':True,'preview_scale':.5,'source_and_generated_dimensions':[48,32],'new_frames_per_preview':17,'latent_identity_preserved':True,'stored_corrected_latent_preview_and_state_identity':True,'artificial_tail_frames_excluded':True,'latest_window_only':True,'http_full_range_suffix_missing_eviction_checks_passed':True,'preview_bytes':{k:len(v) for k,v in data.items()},'scope':'Synthetic CPU images and native H3 latent2rgb factors; VAE decode forbidden; native MP4 codec and HTTP service. No H3 inference or private imagery.'}
 config.output.parent.mkdir(parents=True,exist_ok=True);config.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
asyncio.run(main())
