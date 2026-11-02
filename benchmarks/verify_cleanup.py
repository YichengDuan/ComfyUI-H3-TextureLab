from pathlib import Path
import sys,asyncio,json,uuid,types,importlib.util,copy,argparse
parser=argparse.ArgumentParser(description='CPU-only native loop/export/cache-cleanup smoke test; no H3 inference.')
parser.add_argument('--comfy-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
config=parser.parse_args();APP=config.comfy_root.resolve();REPO=Path(__file__).resolve().parents[1];OUT=config.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(APP));sys.argv=['texture-cleanup-smoke','--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch,nodes,execution,folder_paths,av
from server import PromptServer
from app.assets.manager import NoAssets
from comfy.cli_args import args
spec=importlib.util.spec_from_file_location('cleanup_native_lab',REPO/'__init__.py',submodule_search_locations=[str(REPO)]);lab=importlib.util.module_from_spec(spec);sys.modules[spec.name]=lab;spec.loader.exec_module(lab)
class VAE:
 def decode(self,video):
  count=sum((1,4,4,4,4)[i%5] for i in range(video.shape[2]));return torch.full((count,32,32,3),.4)
class Fixture:
 @classmethod
 def INPUT_TYPES(cls):return {'required':{'index':('INT',{'default':0})}}
 RETURN_TYPES=('LATENT','VAE','AUDIO');FUNCTION='make'
 def make(self,index=0):return {'samples':[torch.zeros(1,24,12,2,2),torch.zeros(1,128,64)]},VAE(),{'sample_rate':48000,'waveform':torch.zeros(1,2,112000)}
async def main():
 assets=NoAssets(args);server=PromptServer(asyncio.get_running_loop(),assets);execution.PromptQueue(server)
 await nodes.init_extra_nodes(init_custom_nodes=False,init_api_nodes=False);nodes.NODE_CLASS_MAPPINGS.update(lab.NODE_CLASS_MAPPINGS);nodes.NODE_CLASS_MAPPINGS['CleanupFixture']=Fixture;folder_paths.set_output_directory(str(OUT))
 graph={
 'audio_fixture':{'class_type':'CleanupFixture','inputs':{'index':0}},
 'plan':{'class_type':'H3TextureWindowPlan','inputs':{'total_frames':56,'window_frames':'39','context_frames':'22','seed':7}},
 'start':{'class_type':'StartLoop','inputs':{'mode':'simple','mode.num_iterations':['plan',2],'cache_iterations':False}},
 'fixture':{'class_type':'CleanupFixture','inputs':{'index':['start',0]}},
 'window':{'class_type':'CleanupWindow','inputs':{'windows':['plan',3],'index':['start',0]}},
 'store':{'class_type':'H3TextureAppendWindow','inputs':{'av_latent':['fixture',0],'vae':['fixture',1],'window':['window',0],'previous':['start',4],'drift_correction':'off'}},
 'end':{'class_type':'EndLoop','inputs':{'output_value':['store',0],'next_iteration_value':['store',0],'accumulate':False}},
 'finish':{'class_type':'H3TextureFinishVideo','inputs':{'state':['end',0],'crf':16,'source_audio':['audio_fixture',2]}},
 'save':{'class_type':'H3TextureSaveVideo','inputs':{'video':['finish',0],'state':['end',0],'filename_prefix':'delivery/cleanup-smoke','cleanup_cache':False}}}
 class Window:
  @classmethod
  def INPUT_TYPES(cls):return {'required':{'windows':('H3_TEXTURE_WINDOWS',),'index':('INT',)}}
  RETURN_TYPES=('H3_TEXTURE_WINDOW',);FUNCTION='select'
  def select(self,windows,index):return (windows['windows'][index],)
 nodes.NODE_CLASS_MAPPINGS['CleanupWindow']=Window
 executor=execution.PromptExecutor(server,cache_type=execution.CacheType.CLASSIC,cache_args={'lru':0,'ram':0,'ram_inactive':0},asset_manager=assets);records=[]
 template=copy.deepcopy(graph)
 for cache_mode,cleanup in [('disk',False),('disk',True),('disk',True),('ram',False),('ram',True)]:
  graph=copy.deepcopy(template)
  deliveries_before=set((OUT/'delivery').glob('*.mp4'))
  graph['store']['inputs']['cache_mode']=cache_mode
  graph['save']['inputs']['cleanup_cache']=cleanup;pid=str(uuid.uuid4());valid=await execution.validate_prompt(pid,graph,None);assert valid[0],valid
  before=set((OUT/'H3-TextureLab-Cache').glob('*'));await executor.execute_async(graph,pid,{},valid[2]);assert executor.success,executor.status_messages
  after=set((OUT/'H3-TextureLab-Cache').glob('*'));assert len(set((OUT/'delivery').glob('*.mp4'))-deliveries_before)==1,'Expected a newly saved delivery';assert len(after-before)==(1 if cache_mode=='disk' and not cleanup else 0),(cleanup,before,after)
  image=executor.history_result['outputs']['save']['images'][0];video=OUT/image['subfolder']/image['filename']
  with av.open(str(video)) as c:assert sum(1 for _ in c.decode(video=0))==56
  with av.open(str(video)) as c:assert sum(f.samples for f in c.decode(audio=0))>0
  records.append({'cache_mode':cache_mode,'cleanup_cache':cleanup,'success':True,'frames':56,'audio_decodes':True,'new_cache_directories_retained':len(after-before),'saved_file':video.name})
 (OUT/'native-validation.json').write_text(json.dumps({'runs':records,'repeat_with_same_executor':True,'fixture':'Synthetic CPU latents and VAE, actual native loop/export/save nodes; no H3 inference.'},indent=2)+'\n');print('NATIVE CLEANUP SMOKE PASSED',json.dumps(records),flush=True)
asyncio.run(main())
