"""Isolated H3 texture experiments: native latent carry and disk-backed windows."""
import nodes
from .storage import append_window,H3TextureFinishVideo
from .planning import window_plan
from .source_window import H3TextureSourceWindow
from .save_video import H3TextureSaveVideo
from .low_vram import H3TextureLowVRAM
from .performance import NODE_CLASS_MAPPINGS as PERFORMANCE_NODES
from comfy_api.latest import io


def context_latent(state):
    data = state['frames'].load_av(state['av_latent_file'])
    return {'samples': [data['video'], data['audio']]}


class H3TextureWindowPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3TextureWindowPlan', display_name='H3 Texture Lab · Window Plan', category='H3 Texture Lab',
            inputs=[io.Int.Input('total_frames', default=158, min=5, max=24000),
                io.Combo.Input('window_frames', options=['39','56','73','90','107','124'], default='56'),
                io.Combo.Input('context_frames', options=['0','1','5','22','39','56'], default='22', tooltip='0: independent windows. 1: last decoded frame re-encoded as a guide. 5/22/39/56: aligned raw latent context.'),
                io.Int.Input('seed', default=28461, min=0, max=0xffffffffffffffff, control_after_generate=io.ControlAfterGenerate.fixed)],
            outputs=[io.Int.Output('frames'),io.Int.Output('stride'),io.Int.Output('iterations'),io.Custom('H3_TEXTURE_WINDOWS').Output('windows')])

    @classmethod
    def execute(cls, total_frames=158, window_frames='56', context_frames='22', seed=28461):
        entries=window_plan(total_frames,window_frames,context_frames,seed)
        return io.NodeOutput(total_frames,int(window_frames)-int(context_frames),len(entries),{'windows':entries})


class H3TextureMotionGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3TextureMotionGuide', display_name='H3 Texture Lab · Native Latent Context', category='H3 Texture Lab',
            inputs=[io.Conditioning.Input('positive'),io.Latent.Input('latent'),io.Vae.Input('vae'),io.Image.Input('reference'),
                io.Custom('H3_TEXTURE_WINDOWS').Input('windows'),io.Int.Input('iteration_index',default=0,min=0),
                io.Custom('H3_TEXTURE_STATE').Input('previous',optional=True),
                io.Combo.Input('context_source',options=['latent','pixels_reencoded'],default='latent',optional=True),
                io.Boolean.Input('first_frame_guide',default=True,optional=True),
                io.Boolean.Input('preserve_target_latent',default=False,optional=True),
                io.Boolean.Input('carry_audio',default=True,optional=True)],
            outputs=[io.Conditioning.Output('positive'),io.Latent.Output('latent'),io.Custom('H3_TEXTURE_WINDOW').Output('window'),
                io.Int.Output('seed'),io.Int.Output('trim_frames'),io.Boolean.Output('has_context')])

    @classmethod
    def execute(cls,positive,latent,vae,reference,windows,iteration_index=0,previous=None,context_source='latent',first_frame_guide=True,preserve_target_latent=False,carry_audio=True):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide,_empty_av_latent
        entry=windows['windows'][iteration_index]
        shape=list(latent['samples'].unbind())[0].shape
        target=latent if preserve_target_latent else _empty_av_latent(shape[-1]*16,shape[-2]*16,entry['window_frames'])[0]
        has_context=previous is not None and entry['history_frames']>0
        if not has_context:
            if first_frame_guide:
                positive=MiniMaxH3AddGuide.execute(positive=positive,latent=target,vae=vae,image=reference[:1],frame_idx=0)[0]
            trim=0
        else:
            motion=nodes.NODE_CLASS_MAPPINGS['MiniMaxH3MotionContext']()
            if entry['history_frames']==1:
                # A generated tail's final token covers four frames. Decode the
                # true last pixel frame instead of mislabelling that token as one.
                positive,trim=motion.apply(positive,vae,target,'1',18,
                    context_frames=previous['frames'][-1:])
                if carry_audio:
                    import importlib
                    upstream=importlib.import_module(type(motion).__module__)
                    if upstream.AUDIO_MODE!='timeline' or upstream.ANCHOR_MODE!='head':
                        raise ValueError('Single-frame audio carry requires upstream timeline/head modes')
                    audio,steps,overhang=upstream._audio_tail_from_latent(context_latent(previous),18)
                    scale=upstream.FRAME_RESCALE
                    end_frame=round(scale+overhang)/scale
                    keyframe={'resolved_frame_index':end_frame-steps/scale,'audio_latent':audio}
                    positive=[[embedding,{**metadata,'minimax_keyframes':[
                        *metadata.get('minimax_keyframes',[]),dict(keyframe)]}] for embedding,metadata in positive]
            else:
                positive,trim=motion.apply(positive,vae,target,str(entry['history_frames']),18,
                    context_latent=context_latent(previous))
            if context_source=='pixels_reencoded' and entry['history_frames']!=1:
                encoded=vae.encode(previous['frames'][-trim:])
                changed=[]
                for embedding,metadata in positive:
                    metadata=metadata.copy();keyframes=[dict(k) for k in metadata.get('minimax_keyframes',[])]
                    indices=[i for i,k in enumerate(keyframes) if 'latent' in k and 0<=k['resolved_frame_index']<trim]
                    if len(indices)!=encoded.shape[2]:
                        raise ValueError('Re-encoded H3 history does not match its native latent grid')
                    for step,index in enumerate(indices):keyframes[index]['latent']=encoded[:,:,step:step+1].clone()
                    metadata['minimax_keyframes']=keyframes;changed.append([embedding,metadata])
                positive=changed
        if not carry_audio:
            positive=[[embedding,{**metadata,'minimax_keyframes':[k for k in metadata.get('minimax_keyframes',[]) if 'audio_latent' not in k]}] for embedding,metadata in positive]
        if trim != entry['discarded_overlap']:
            raise ValueError('Native H3 context trim does not match the disk window plan')
        return io.NodeOutput(positive,target,entry,entry['seed'],trim,has_context)


class H3TextureAppendWindow(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3TextureAppendWindow', display_name='H3 Texture Lab · Store AV Latent · Disk / RAM', category='H3 Texture Lab',
            inputs=[io.Latent.Input('av_latent'),io.Vae.Input('vae'),io.Custom('H3_TEXTURE_WINDOW').Input('window'),
                io.Custom('H3_TEXTURE_STATE').Input('previous',optional=True),
                io.Combo.Input('drift_correction',options=['off','join_match'],default='off'),
                io.Combo.Input('cache_mode',options=['disk','ram'],default='disk',optional=True,tooltip='disk: save per-window AV files. ram: keep CPU tensors and assemble MP4 in memory; final delivery still saves to output.')],
            outputs=[io.Custom('H3_TEXTURE_STATE').Output('state')])

    @classmethod
    def execute(cls,av_latent,vae,window,previous=None,drift_correction='off',cache_mode='disk'):
        if drift_correction=='join_match' and previous is not None and window['discarded_overlap']>0:
            av_latent=nodes.NODE_CLASS_MAPPINGS['MiniMaxH3JoinMatch']().apply(
                av_latent,1.,window['discarded_overlap'],1.5,match_spread=True,match_detail=True,
                tuning='auto',context_latent=context_latent(previous))[0]
        return io.NodeOutput(append_window(av_latent,vae,window,previous,cache_mode))


NODE_CLASS_MAPPINGS={c.__name__:c for c in (H3TextureWindowPlan,H3TextureMotionGuide,H3TextureAppendWindow,H3TextureSourceWindow)}

NODE_CLASS_MAPPINGS.update({c.__name__:c for c in (H3TextureFinishVideo,H3TextureLowVRAM,H3TextureSaveVideo)})
NODE_CLASS_MAPPINGS.update(PERFORMANCE_NODES)
