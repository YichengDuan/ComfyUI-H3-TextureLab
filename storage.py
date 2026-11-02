"""Bounded-memory AV latent storage and final streaming video export."""
from pathlib import Path
import json
import subprocess
import uuid
import wave
import numpy as np
import torch
from safetensors.torch import load_file, save_file
import imageio_ffmpeg
from comfy_api.latest import io

CONTRACT='h3_texture_disk_v1'


class LatentTimeline:
    def __init__(self,root,chunks,shape,vae):
        self.root=Path(root);self.chunks=tuple(chunks);self.shape=torch.Size(shape);self.vae=vae

    def __len__(self):return self.shape[0]

    def path(self,name):
        path=(self.root/name).resolve()
        if not path.is_relative_to(self.root.resolve()):raise ValueError('Latent file escaped its run directory')
        return path

    def decode_chunk(self,chunk):
        name,first,count,trim=chunk
        data=load_file(str(self.path(name)),device='cpu');frames=self.vae.decode(data['video']).detach().cpu()
        if frames.ndim==5:frames=frames.reshape(-1,*frames.shape[-3:])
        if len(frames)<trim+count:raise ValueError('Decoded latent does not cover the requested window')
        return frames[trim:trim+count]

    def __getitem__(self,key):
        if not isinstance(key,slice):raise TypeError('Window history requires a contiguous frame slice')
        start,stop,step=key.indices(len(self))
        if step!=1:raise ValueError('Window history requires contiguous frames')
        result=torch.empty((max(0,stop-start),*self.shape[1:]),dtype=torch.float32);offset=0
        for chunk in self.chunks:
            _,first,count,_=chunk;lo,hi=max(start,first),min(stop,first+count)
            if hi<=lo:continue
            decoded=self.decode_chunk(chunk);result[offset:offset+hi-lo].copy_(decoded[lo-first:hi-first]);offset+=hi-lo;del decoded
        if offset!=len(result):raise ValueError('Saved windows do not cover the requested history')
        return result

    def iter_frames(self):
        for chunk in self.chunks:
            decoded=self.decode_chunk(chunk)
            for frame in decoded:yield frame.clone()
            del frame,decoded


def append_window(av_latent,vae,window,previous=None):
    import folder_paths
    samples=av_latent['samples'];video,audio=list(samples.unbind()) if hasattr(samples,'unbind') else samples
    if video.ndim!=5 or video.shape[0]!=1 or video.shape[1]!=24:
        raise ValueError('Expected native H3 video/audio sampler output')
    prefix=previous['frames'] if previous is not None else None;retained=len(prefix) if prefix is not None else 0
    trim=window['discarded_overlap'];end=window['source_end_exclusive']
    if retained!=window['source_start']+trim:raise ValueError('Window plan would create a gap or duplicate')
    if prefix is None:
        root=Path(folder_paths.get_output_directory())/'H3-TextureLab-Cache'/uuid.uuid4().hex;root.mkdir(parents=True);chunks=()
    else:root,chunks=prefix.root,prefix.chunks
    name=f'window_{len(chunks):05d}.safetensors';temporary=root/(name+'.part')
    save_file({'video':video.detach().to(device='cpu',dtype=torch.float32).contiguous(),
        'audio':audio.detach().to(device='cpu',dtype=torch.float32).contiguous()},str(temporary));temporary.replace(root/name)
    timeline=LatentTimeline(root,(*chunks,(name,retained,end-retained,trim)),(end,video.shape[-2]*16,video.shape[-1]*16,3),vae)
    state={'contract':CONTRACT,'frames':timeline,'windows':[*previous['windows'],window] if previous else [window],
        'source_frames':window['source_frames'],'av_latent_file':name}
    manifest={k:v for k,v in state.items() if k!='frames'};manifest.update(chunks=timeline.chunks,shape=list(timeline.shape),fps=24)
    temporary=root/'manifest.json.part';temporary.write_text(json.dumps(manifest,indent=2)+'\n');temporary.replace(root/'manifest.json')
    return state


class H3TextureFinishVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3TextureFinishVideo',display_name='H3 Texture Lab · Stream Final Video',category='H3 Texture Lab',
            description='Decode one saved window at a time; optionally retain the original source soundtrack.',
            inputs=[io.Custom('H3_TEXTURE_STATE').Input('state'),io.Int.Input('crf',default=16,min=0,max=51),io.Audio.Input('source_audio',optional=True)],
            outputs=[io.Video.Output('video'),io.String.Output('report')])

    @classmethod
    def execute(cls,state,crf=16,source_audio=None):
        from comfy_api.input_impl import VideoFromFile
        timeline=state['frames']
        if state['contract']!=CONTRACT or len(timeline)!=state['source_frames']:
            raise ValueError('Saved windows do not cover the completed timeline')
        partial=timeline.root/'assembling.mp4';output=timeline.root/'assembled.mp4'
        audio_args=['-an']
        audio_file=None
        if source_audio is not None:
            rate=int(source_audio['sample_rate']);count=round(len(timeline)*rate/24)
            waveform=source_audio['waveform'][0].detach().cpu().clamp(-1,1)
            waveform=torch.nn.functional.pad(waveform[...,:count],(0,max(0,count-waveform.shape[-1])))
            audio_file=timeline.root/'source-audio.wav'
            with wave.open(str(audio_file),'wb') as stream:
                stream.setnchannels(waveform.shape[0]);stream.setsampwidth(2);stream.setframerate(rate)
                stream.writeframes(waveform.T.mul(32767).round().numpy().astype('<i2').tobytes())
            audio_args=['-i',str(audio_file),'-map','0:v:0','-map','1:a:0','-c:a','aac','-b:a','192k']
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-threads','2','-f','rawvideo','-pix_fmt','rgb48le',
            '-s',f'{timeline.shape[2]}x{timeline.shape[1]}','-r','24','-i','pipe:0',*audio_args,'-c:v','libx264','-threads','2',
            '-preset','fast','-crf',str(crf),'-pix_fmt','yuv420p','-movflags','+faststart',str(partial)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            with process.stdin:
                for frame in timeline.iter_frames():
                    pixels=frame.clamp(0,1).mul(65535).round().numpy().astype('<u2');process.stdin.write(pixels.tobytes())
            error=process.stderr.read().decode(errors='replace')
            if process.wait()!=0:raise RuntimeError('Video export failed: '+error[-1500:])
        except BaseException:
            process.kill();process.wait();raise
        finally:
            process.stderr.close()
            if audio_file is not None:audio_file.unlink(missing_ok=True)
        partial.replace(output)
        report={'frames':len(timeline),'fps':24,'canvas':list(timeline.shape[1:3]),'audio_export':'source' if source_audio is not None else 'none','cache_directory':str(timeline.root)}
        return io.NodeOutput(VideoFromFile(str(output)),json.dumps(report,indent=2))
