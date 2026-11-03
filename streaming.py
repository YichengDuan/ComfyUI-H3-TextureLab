"""File-backed RGB reads and compact masks for continuous native SAM3 tracking.

RGB tensors are materialized only for the first frame or the requested H3 window.
The native tracker receives a bounded, tensor-like frame sequence in a single call.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import importlib
import io as python_io
import av
import json
import math
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
import numpy as np
import torch
import torch.nn.functional as F

FPS = 24

def interrupted():
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()

def tools(name):
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f'{name} is required on PATH for streaming video input')
    return found

def capture(command, timeout=180):
    """Small metadata/audio subprocess with cancellation and bounded duration."""
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    started = time.monotonic()
    try:
        while True:
            interrupted()
            if time.monotonic() - started > timeout:
                raise TimeoutError('Video input subprocess timed out')
            try:
                output, error = process.communicate(timeout=.25)
                break
            except subprocess.TimeoutExpired:
                pass
        if process.returncode:
            raise RuntimeError('Video input could not be decoded: ' + error.decode(errors='replace')[-1200:])
        return output
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

def prep_helpers():
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get('H3RelayPrepareSource')
    if cls is None:
        raise RuntimeError('Install H3 Relay before using streaming Character Swap')
    return importlib.import_module(cls.__mro__[1].__module__)

def file_stamp(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns

@dataclass
class StreamSource:
    path: str
    width: int
    height: int
    frames: int
    filters: str
    stamp: tuple
    def verify(self):
        if file_stamp(self.path) != self.stamp:
            raise ValueError('Source video changed during the run; prepare/track it again')
    def command(self):
        self.verify()
        return [tools('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin',
                '-threads', '2', '-i', self.path, '-map', '0:v:0', '-an', '-vf', self.filters,
                '-frames:v', str(self.frames), '-threads', '2']

class InterruptiblePipe(python_io.RawIOBase):
    def __init__(self, pipe):
        self.pipe = pipe
    def readable(self):return True
    def seekable(self):return False
    def read(self, size=-1):
        while True:
            interrupted()
            if select.select([self.pipe], [], [], .5)[0]:
                return os.read(self.pipe.fileno(), min(size if size > 0 else 32768, 1024*1024))

class FrameSequence:
    """NCHW indexing for native SAM3; cache holds native pixel planes, never a full RGB float timeline."""
    def __init__(self, source, capacity=3):
        self.source = source
        self.capacity = int(capacity)
        if not 1 <= self.capacity <= 126:
            raise ValueError('Frame cache must be bounded to 1..126 frames')
        self.shape = (source.frames, 3, source.height, source.width)
        self.device, self.dtype = torch.device('cpu'), torch.float32
        self.cache = OrderedDict()
        self.process = None
        self.container = None
        self.decoded = None
        self.error = None
        self.next_index = 0
        self.peak_cache_bytes = 0
        self.restarts = 0
        self.lock = threading.RLock()
    def __len__(self):
        return self.source.frames
    def _open(self):
        self.source.verify()
        self.error = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(self.source.command() + ['-c:v', 'ffv1',
                '-cluster_time_limit', '100', '-f', 'matroska', 'pipe:1'],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=self.error, bufsize=0)
            self.container = av.open(InterruptiblePipe(self.process.stdout), format='matroska')
            self.container.streams.video[0].codec_context.thread_count = 2
            self.decoded = self.container.decode(video=0)
        except BaseException:
            self.close()
            raise
        self.next_index = 0
        self.restarts += 1
    def _one(self):
        interrupted()
        try:
            return next(self.decoded)
        except StopIteration:
            self.error.seek(0)
            raise ValueError('Source ended before its counted frame: ' + self.error.read()[-1200:].decode(errors='replace'))
    def _get(self, index):
        if not 0 <= index < len(self):
            raise IndexError('Source frame is outside the timeline')
        if index in self.cache:
            return self.cache[index]
        if self.process is None or index < self.next_index:
            self.close()
            self._open()
        while self.next_index <= index:
            frame = self._one()
            self.cache[self.next_index] = frame
            self.next_index += 1
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
            self.peak_cache_bytes = max(self.peak_cache_bytes, sum(sum(p.buffer_size for p in x.planes) for x in self.cache.values()))
        return self.cache[index]
    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError('Streaming frames support only bounded contiguous slices')
        first, end, step = key.indices(len(self))
        if step != 1 or not 0 < end-first <= self.capacity:
            raise ValueError('A streaming read must fit the bounded frame cache')
        with self.lock:
            self.source.verify()
            try:
                output = torch.empty((end-first, self.source.height, self.source.width, 3), dtype=torch.float32)
                for offset, index in enumerate(range(first, end)):
                    frame = self._get(index)
                    # Match VideoFromFile's precision/chroma conversion, rather
                    # than silently quantizing the new source path to RGB8.
                    if frame.format.name in ('yuvj420p','yuvj422p','yuvj444p','rgb24','rgba','pal8'):
                        pixels = torch.from_numpy(frame.to_ndarray(format='rgb24')).float().div_(255)
                    else:
                        pixels = torch.from_numpy(frame.to_ndarray(format='gbrpf32le'))
                    output[offset].copy_(pixels)
                return output.movedim(-1, 1)
            except BaseException:
                self.close()
                raise
    def close(self):
        self.decoded = None
        if self.container is not None:
            self.container.close(); self.container = None
        if self.process is not None:
            process, self.process = self.process, None
            if process.stdout:
                process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
        if self.error:
            self.error.close(); self.error = None
        self.cache.clear()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

@dataclass
class TrackedSource:
    source: StreamSource
    masks: np.ndarray
    report: dict
    reader: object = None
    lock: object = field(default_factory=threading.RLock)
    def close(self):
        if self.reader is not None:
            self.reader.close(); self.reader = None
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

def prepare_source(video, resolution, duration_seconds=0., missing_audio='silence'):
    if os.name != "posix":
        raise RuntimeError("Streaming source currently requires Linux/POSIX")
    helpers = prep_helpers()
    # Materialized/trimmed/cropped wrappers need a separate explicit adapter;
    # never silently read an underlying file while ignoring its active edits.
    source = video.get_stream_source()
    if not isinstance(source, (str, os.PathLike)) or video.get_active_trim_window() != (0., 0.) or getattr(video, '_VideoFromFile__crop', None):
        raise ValueError('Streaming input requires an untrimmed file-backed Load Video. Use duration_seconds here for a prefix trim.')
    source = str(Path(source).resolve())
    stamp = file_stamp(source)
    helpers.requested_window(duration_seconds)
    requested = max(1, math.floor(duration_seconds*FPS)) if duration_seconds else None
    probe = json.loads(capture([tools('ffprobe'), '-v', 'error', '-show_streams', '-of', 'json', source], 30))
    visual = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    if visual is None:
        raise ValueError('Source does not have a video stream')
    w, h = helpers.canvas_dimensions(*helpers.display_dimensions(visual), resolution, preserve_aligned=True)
    filters = (f'setpts=PTS-STARTPTS,fps=fps=24:round=near:eof_action=round,'
        f"scale=w='ceil(max({w},{h}*dar)-1e-6)':h='ceil(max({h},{w}/dar)-1e-6)':flags=lanczos,"
        f'setsar=1,crop=w={w}:h={h}')
    base = [tools('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin', '-threads', '2',
            '-i', source, '-map', '0:v:0', '-an', '-vf', filters, '-frames:v', str(requested or helpers.MAX_FRAMES+1), '-threads', '2']
    progress = capture(base + ['-progress', 'pipe:1', '-nostats', '-f', 'null', '-']).decode()
    count = max((int(line.split('=',1)[1]) for line in progress.splitlines() if line.startswith('frame=')), default=0)
    length = helpers.window_length(count)
    stream = StreamSource(source, w, h, count, filters, stamp)
    with FrameSequence(stream, 1) as reader:
        first = reader[:1].movedim(1,-1).contiguous()
    sound = next((s for s in probe['streams'] if s['codec_type']=='audio'), None)
    audio = None
    if sound:
        rate, channels = int(sound['sample_rate']), int(sound['channels'])
        if not 1 <= channels <= 8 or not 1 <= rate <= 192000:
            raise ValueError('Streaming audio supports up to 8 channels and 192 kHz')
        raw = capture([tools('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin', '-threads', '2',
            '-i', source, '-map', '0:a:0', '-vn', '-af', f'asetpts=PTS-STARTPTS,atrim=end={count/FPS:.12f}',
            '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'])
        values = np.frombuffer(raw, '<i2')
        if len(values) % channels:
            raise ValueError('Audio decoder returned an incomplete sample')
        wave = torch.from_numpy(values.reshape(-1, channels).T.copy()).float().div_(32768)[None]
        audio = {'sample_rate':rate,'waveform':wave[..., :round(count*rate/FPS)]}
    elif missing_audio == 'silence':
        audio = {'sample_rate':48000,'waveform':torch.zeros(1,2,round(count*48000/FPS))}
    stream.verify()
    summary = f'{w} × {h} · {count} source frames · 24 fps · streaming RGB; compact per-frame masks; no prepared-video cache file'
    return stream, audio, w, h, length, count, float(FPS), summary, first

def track_source(source, model, initial_mask, conditioning, selection, detection_threshold=.5, max_objects=1, detect_interval=1, recover_empty_frames=True):
    import comfy.model_management as mm
    import comfy.utils
    import nodes
    from comfy_extras.nodes_sam3 import _extract_text_prompts, SAM3_Detect
    from comfy.ldm.sam3.tracker import unpack_masks
    source.verify()
    mm.load_model_gpu(model)
    device, dtype = mm.get_torch_device(), model.model.get_dtype()
    sam = model.model.diffusion_model
    if not hasattr(sam.tracker, 'track_video_with_detection'):
        raise ValueError('Streaming Character Swap requires SAM3.1 multiplex tracking')
    prompts = None
    if conditioning:
        prompts = [(emb, mask) for emb,mask,_ in _extract_text_prompts(conditioning, device, dtype)]
    with FrameSequence(source, 3) as frames:
        result = sam.forward_video(images=frames, initial_masks=initial_mask.unsqueeze(1).to(device=device,dtype=dtype),
            pbar=comfy.utils.ProgressBar(source.frames), text_prompts=prompts, new_det_thresh=detection_threshold,
            max_objects=max_objects, detect_interval=detect_interval, target_device=device, target_dtype=dtype)
        peak = frames.peak_cache_bytes
    packed = result['packed_masks']
    if packed is not None:
        packed = packed.cpu()
    masks = np.empty((source.frames, source.height, source.width//8), np.uint8)
    empty, recovered = [], []
    with FrameSequence(source, 3) as frames:
        for index in range(source.frames):
            interrupted()
            if packed is None or packed.shape[1] == 0:
                binary = np.zeros((source.height,source.width), bool)
            else:
                union = packed[index,0].clone()
                for obj in range(1, packed.shape[1]):
                    union |= packed[index,obj]
                image_mask = F.interpolate(unpack_masks(union)[None,None].float(),
                    size=(source.height,source.width),mode='bilinear',align_corners=False)[0,0]
                binary = (image_mask > .5).numpy()
            if not binary.any():
                empty.append(index)
                if recover_empty_frames:
                    image = frames[index:index+1].movedim(1,-1)
                    detection = SAM3_Detect.execute(model=model,image=image,conditioning=conditioning,
                        threshold=.35,refine_iterations=2,individual_masks=True)[0]
                    chosen = nodes.NODE_CLASS_MAPPINGS['H3RelaySelectMask']().select(detection,**selection)[0][0]
                    binary = (chosen.detach().cpu() > .5).numpy()
                    recovered.append(index)
            masks[index] = np.packbits(binary,axis=-1,bitorder='little')
    source.verify()
    report={'source_frames':source.frames,'width':source.width,'height':source.height,
            'empty_tracking_frames':empty,'recovered_frames':recovered,'packed_mask_bytes':masks.nbytes,
            'rgb_cache_limit_frames':3,'peak_tracking_native_frame_cache_bytes':peak,'native_tracker_calls':1,
            'tracker_state_reset_between_frames':False}
    return TrackedSource(source,masks,report), json.dumps(report)

def expand_source(tracked, expansion_px):
    from scipy.ndimage import distance_transform_edt
    if not 0 <= expansion_px <= 256:
        raise ValueError('Expansion must be between 0 and 256 pixels')
    tracked.source.verify()
    output = np.empty_like(tracked.masks)
    missing=[]
    for index,packed in enumerate(tracked.masks):
        interrupted()
        binary = np.unpackbits(packed,axis=-1,bitorder='little').astype(bool)
        if not binary.any():
            missing.append(index)
        elif expansion_px:
            binary = distance_transform_edt(~binary) <= expansion_px
        output[index] = np.packbits(binary,axis=-1,bitorder='little')
    if missing:
        raise ValueError(f'SAM lost the person at frames {missing[:20]}; correct the subject or split the source shot')
    return TrackedSource(tracked.source,output,{**tracked.report,'expansion_px':expansion_px})

def source_window(tracked, windows, iteration_index=0, source_audio=None):
    source=tracked.source
    source.verify()
    entry=windows['windows'][iteration_index]
    start,size,count=entry['source_start'],entry['window_frames'],entry['source_frames']
    if count!=source.frames or not 1 <= size <= 124 or not 0 <= start < count:
        raise ValueError('Window plan does not match its streaming source')
    end=min(start+size,count)
    with tracked.lock:
        if tracked.reader is None or tracked.reader.capacity<size+2:
            tracked.close();tracked.reader=FrameSequence(source,size+2)
        try:
            frames=tracked.reader[start:end].movedim(1,-1).contiguous()
            if end==count:tracked.close()
        except BaseException:
            tracked.close();raise
    mask=torch.from_numpy(np.unpackbits(tracked.masks[start:end],axis=-1,bitorder='little').copy()).float()
    if end-start<size:
        frames=torch.cat((frames,frames[-1:].expand(size-len(frames),-1,-1,-1)))
        mask=torch.cat((mask,mask[-1:].expand(size-len(mask),-1,-1)))
    audio=None
    if source_audio is not None:
        rate=source_audio['sample_rate'];first,last=round(start*rate/FPS),round((start+size)*rate/FPS)
        wave=source_audio['waveform'][...,first:last]
        audio={'sample_rate':rate,'waveform':F.pad(wave,(0,last-first-wave.shape[-1]))}
    return frames,mask,audio,size

class H3TextureStreamPrepareSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'video':('VIDEO',),'resolution':(['480','768'],{'default':'480'})},'optional':{
            'duration_seconds':('FLOAT',{'default':0.,'min':0.,'max':149.66,'step':.1}),
            'missing_audio':(['silence','none'],{'default':'silence'})}}
    RETURN_TYPES=('H3_TEXTURE_STREAM_SOURCE','AUDIO','INT','INT','INT','INT','FLOAT','STRING','IMAGE')
    RETURN_NAMES=('stream_source','source_audio','width','height','h3_length','source_frames','fps','summary','first_frame')
    FUNCTION='prepare';CATEGORY='H3 Texture Lab/streaming'
    def prepare(self,video,resolution,duration_seconds=0.,missing_audio='silence'):
        return prepare_source(video,resolution,duration_seconds,missing_audio)

class H3TextureStreamSAM3Track:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'source':('H3_TEXTURE_STREAM_SOURCE',),'model':('MODEL',),'initial_mask':('MASK',),
            'conditioning':('CONDITIONING',),'selection':('H3_SUBJECT_SELECTION',),
            'detection_threshold':('FLOAT',{'default':.5,'min':0.,'max':1.,'step':.01}),
            'max_objects':('INT',{'default':1,'min':1,'max':64}),
            'detect_interval':('INT',{'default':1,'min':1}),
            'recover_empty_frames':('BOOLEAN',{'default':True})}}
    RETURN_TYPES=('H3_TEXTURE_TRACKED_SOURCE','STRING');RETURN_NAMES=('tracked_source','report')
    FUNCTION='track';CATEGORY='H3 Texture Lab/streaming'
    def track(self,**kwargs):return track_source(**kwargs)

class H3TextureStreamExpandMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'tracked':('H3_TEXTURE_TRACKED_SOURCE',),'expansion_px':('INT',{'default':32,'min':0,'max':256})}}
    RETURN_TYPES=('H3_TEXTURE_TRACKED_SOURCE',);RETURN_NAMES=('expanded_source',)
    FUNCTION='expand';CATEGORY='H3 Texture Lab/streaming'
    def expand(self,tracked,expansion_px):return (expand_source(tracked,expansion_px),)

class H3TextureStreamSourceWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'source':('H3_TEXTURE_TRACKED_SOURCE',),'windows':('H3_TEXTURE_WINDOWS',),
            'iteration_index':('INT',{'default':0,'min':0})},'optional':{'source_audio':('AUDIO',)}}
    RETURN_TYPES=('IMAGE','MASK','AUDIO','INT');RETURN_NAMES=('frames','mask','audio','length')
    FUNCTION='window';CATEGORY='H3 Texture Lab/streaming'
    def window(self,source,windows,iteration_index=0,source_audio=None):
        return source_window(source,windows,iteration_index,source_audio)

NODE_CLASS_MAPPINGS={cls.__name__:cls for cls in (H3TextureStreamPrepareSource,H3TextureStreamSAM3Track,H3TextureStreamExpandMasks,H3TextureStreamSourceWindow)}
NODE_DISPLAY_NAME_MAPPINGS={
    'H3TextureStreamPrepareSource':'H3 Texture Lab · Streaming Source · No Full RGB',
    'H3TextureStreamSAM3Track':'H3 Texture Lab · Streaming SAM3 · Continuous State',
    'H3TextureStreamExpandMasks':'H3 Texture Lab · Compact Mask Expansion',
    'H3TextureStreamSourceWindow':'H3 Texture Lab · Stream Source Window',
}
