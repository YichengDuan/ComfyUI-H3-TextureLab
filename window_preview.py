"""Latest-window source/mask and generated-video previews in bounded RAM."""
from collections import OrderedDict
from fractions import Fraction
import io
import threading
import uuid
import av
import torch
import torch.nn.functional as F

_PREVIEWS = OrderedDict()
_LOCK = threading.Lock()
MAX_PREVIEW_BYTES = 64 * 1024 ** 2

def publish(source, generated):
    if sum(len(x) for x in (source,generated) if x is not None) > MAX_PREVIEW_BYTES:
        raise ValueError('Current-window previews exceed the 64 MiB RAM limit; reduce preview_scale')
    token=uuid.uuid4().hex
    with _LOCK:
        _PREVIEWS.clear()  # Latest window only; never retain old RGB or latent tensors.
        _PREVIEWS[token]={'source':source,'generated':generated}
    return token

def register_preview_routes():
    from server import PromptServer
    from aiohttp import web
    server=getattr(PromptServer,'instance',None)
    if server is None or getattr(server,'_h3_texture_preview_routes',False):return
    server._h3_texture_preview_routes=True
    @server.routes.get('/h3_texture/window_preview/{token}/{kind}')
    async def preview(request):
        with _LOCK:
            data=_PREVIEWS.get(request.match_info['token'],{}).get(request.match_info['kind'])
        if data is None:raise web.HTTPNotFound()
        length=len(data);start,end=0,length-1;status=200
        headers={'Accept-Ranges':'bytes','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}
        ranges=request.headers.get('Range')
        if ranges:
            try:
                if not ranges.startswith('bytes=') or ',' in ranges:raise ValueError()
                first,last=ranges[6:].split('-',1)
                if first:
                    start=int(first);end=min(int(last),length-1) if last else length-1
                else:
                    suffix=int(last)
                    if suffix<=0:raise ValueError()
                    start=max(0,length-suffix)
                if not 0<=start<=end<length:raise ValueError()
            except (ValueError,TypeError):
                return web.Response(status=416,headers={'Content-Range':f'bytes */{length}'})
            status=206;headers['Content-Range']=f'bytes {start}-{end}/{length}'
        return web.Response(body=data[start:end+1],status=status,content_type='video/mp4',headers=headers)

def encode_preview(frames, scale=.5, mask=None, first=0, display_size=None):
    import comfy.model_management as mm
    if not .125<=scale<=1:raise ValueError('Preview scale must be between 0.125 and 1')
    count=len(frames)-first
    if count<=0:raise ValueError('Preview needs at least one new frame')
    h,w=display_size or frames.shape[1:3];out_h=max(2,round(h*scale)//2*2);out_w=max(2,round(w*scale)//2*2)
    buffer=io.BytesIO()
    with av.open(buffer,mode='w',format='mp4') as output:
        stream=output.add_stream('libx264',rate=24);stream.width=out_w;stream.height=out_h;stream.pix_fmt='yuv420p'
        stream.options={'crf':'28','preset':'veryfast','threads':'2'}
        for index in range(first,len(frames)):
            mm.throw_exception_if_processing_interrupted()
            image=frames[index:index+1].detach().cpu().movedim(-1,1)
            image=F.interpolate(image,size=(out_h,out_w),mode='bilinear',align_corners=False,antialias=True)[0].movedim(0,-1)
            if mask is not None:
                selected=F.interpolate(mask[index:index+1,None].cpu().float(),size=(out_h,out_w),mode='nearest')[0,0]>0
                image=torch.where(selected[...,None],image*.65+torch.tensor([.1,.9,.4])*.35,image)
            array=image.mul(255).round().clamp(0,255).byte().contiguous().numpy()
            frame=av.VideoFrame.from_ndarray(array,format='rgb24');frame.pts=index-first;frame.time_base=Fraction(1,24)
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    data=buffer.getvalue();buffer.close()
    return data,{'width':out_w,'height':out_h,'frames':count,'bytes':len(data)}

def latent2rgb_frames(video, frame_count):
    """Native H3 linear RGB approximation, with its 1/4/4/4/4 frame clock."""
    from comfy.latent_formats import MiniMaxH3Video
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError('H3 preview expects one [1,24,T,H,W] video latent')
    fmt = MiniMaxH3Video()
    latent = video.detach().to(device='cpu', dtype=torch.float32)[0].movedim(0, -1)
    factors = torch.tensor(fmt.latent_rgb_factors, dtype=torch.float32).T
    bias = torch.tensor(fmt.latent_rgb_factors_bias, dtype=torch.float32)
    rgb = F.linear(latent, factors, bias).add_(1).mul_(.5).clamp_(0, 1)
    repeats = torch.tensor([1 if i % 5 == 0 else 4 for i in range(len(rgb))])
    rgb = torch.repeat_interleave(rgb, repeats, dim=0)
    if len(rgb) < frame_count:
        rgb = torch.cat((rgb, rgb[-1:].expand(frame_count-len(rgb), -1, -1, -1)))
    return rgb[:frame_count]

class H3TextureWindowPreview:
    DESCRIPTION = "Current source and generated window; excludes carried overlap and model-only tail padding. Generated previews use approximate latent2rgb without VAE decoding; final output is unchanged."
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'av_latent':('LATENT',),'vae':('VAE',),'source_frames':('IMAGE',),'mask':('MASK',),
            'window':('H3_TEXTURE_WINDOW',),'total_windows':('INT',{'default':1,'min':1}),
            'preview_source':('BOOLEAN',{'default':True}),'preview_generated':('BOOLEAN',{'default':True}),
            'preview_scale':('FLOAT',{'default':.5,'min':.125,'max':1.,'step':.125})}}
    RETURN_TYPES=('LATENT',);RETURN_NAMES=('av_latent',)
    FUNCTION='preview';CATEGORY='H3 Texture Lab/streaming'
    def preview(self,av_latent,vae,source_frames,mask,window,total_windows,preview_source=True,preview_generated=True,preview_scale=.5):
        if not preview_source and not preview_generated:return (av_latent,)
        first=window['discarded_overlap'];source=generated=None;details={}
        valid=min(len(source_frames),window['source_end_exclusive']-window['source_start'])
        if preview_source:
            source,details['source']=encode_preview(source_frames[:valid],preview_scale,mask[:valid],first)
        if preview_generated:
            samples=av_latent['samples']
            video=list(samples.unbind() if hasattr(samples,'unbind') else samples)[0]
            decoded=latent2rgb_frames(video, len(source_frames))
            generated,details['generated']=encode_preview(decoded[:valid],preview_scale,None,first,
                                                           display_size=source_frames.shape[1:3])
            details['generated']['method']='latent2rgb'
            del decoded
        token=publish(source,generated)
        urls={name:f'/h3_texture/window_preview/{token}/{name}' for name,data in [('source',source),('generated',generated)] if data is not None}
        info={'window_index':window['index'],'total_windows':total_windows,'source_start':window['source_start'],
              'source_end_exclusive':window['source_end_exclusive'],'scale':preview_scale,'urls':urls,'details':details}
        return {'result':(av_latent,),'ui':{'h3_texture_window_preview':[info]}}

NODE_CLASS_MAPPINGS={'H3TextureWindowPreview':H3TextureWindowPreview}


class H3TextureStoredWindowPreview(H3TextureWindowPreview):
    DESCRIPTION = "Preview the stored window after Join Match; passes its cache state through unchanged."
    @classmethod
    def INPUT_TYPES(cls):
        schema=super().INPUT_TYPES()
        schema['required']={'state':('H3_TEXTURE_STATE',),**{k:v for k,v in schema['required'].items() if k!='av_latent'}}
        return schema
    RETURN_TYPES=('H3_TEXTURE_STATE',)
    RETURN_NAMES=('state',)
    def preview(self,state,vae,source_frames,mask,window,total_windows,preview_source=True,preview_generated=True,preview_scale=.5):
        if not preview_source and not preview_generated:return (state,)
        data=state['frames'].load_av(state['av_latent_file']) if preview_generated else {'video':None,'audio':None}
        result=super().preview({'samples':[data['video'],data['audio']]},vae,source_frames,mask,window,total_windows,preview_source,preview_generated,preview_scale)
        return {'result':(state,),'ui':result['ui']}

NODE_CLASS_MAPPINGS['H3TextureStoredWindowPreview']=H3TextureStoredWindowPreview
