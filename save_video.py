"""Save a verified delivery before optionally removing this run's disk cache."""
import json
import logging
from pathlib import Path
import re
import uuid

import av
from comfy_api.latest import io

from .storage import CONTRACT


def verify_video(path, timeline, interrupt_check):
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate != 24:
            raise ValueError("Saved video has an unexpected frame rate")
        for frame in container.decode(stream):
            interrupt_check()
            if (frame.width, frame.height) != (timeline.shape[2], timeline.shape[1]):
                raise ValueError("Saved video has an unexpected canvas")
            count += 1
    if count != len(timeline):
        raise ValueError("Saved video does not contain the complete timeline")


def cleanup_run(state, output_directory):
    """Remove only the completed, recognized files belonging to this state."""
    timeline = state['frames']
    root = timeline.root
    cache = (Path(output_directory) / 'H3-TextureLab-Cache').resolve()
    if root.is_symlink() or root.resolve().parent != cache or not re.fullmatch(r'[0-9a-f]{32}', root.name):
        return 'skipped: run directory is outside the expected cache layout'
    names = {chunk[0] for chunk in timeline.chunks} | {'manifest.json', 'assembled.mp4'}
    if any(Path(name).name != name for name in names):
        return 'skipped: cache contains an unexpected filename'
    entries = list(root.iterdir())
    if any(p.name not in names or p.is_symlink() or not p.is_file() for p in entries):
        return 'skipped: cache contains unrecognized files or links'
    manifest = json.loads((root / 'manifest.json').read_text())
    if (manifest.get('contract') != CONTRACT
            or manifest.get('source_frames') != len(timeline)
            or manifest.get('chunks') != [list(chunk) for chunk in timeline.chunks]
            or manifest.get('windows') != state['windows']):
        return 'skipped: cache manifest does not match the completed state'
    for path in entries:
        path.unlink()
    root.rmdir()
    return 'deleted'


def save_delivery(video, state, filename_prefix, cleanup_cache, interrupt_check, metadata=None):
    import folder_paths
    from comfy_api.latest import Types

    timeline = state['frames']
    if state['contract'] != CONTRACT or len(timeline) != state['source_frames']:
        raise ValueError("Cannot save an incomplete H3 timeline")
    output_directory = folder_paths.get_output_directory()
    folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, output_directory, timeline.shape[2], timeline.shape[1])
    filename = f'{name}_{counter:05}_.mp4'
    destination = Path(folder) / filename
    cache = (Path(output_directory) / 'H3-TextureLab-Cache').resolve()
    if destination.resolve().is_relative_to(cache) or destination.resolve().is_relative_to(timeline.root.resolve()):
        raise ValueError("Final video must be saved outside the cache directory")
    if cleanup_cache:
        source = video.get_stream_source()
        if not isinstance(source, (str, Path)) or Path(source).resolve() != (timeline.root / 'assembled.mp4').resolve():
            raise ValueError("Cleanup requires the assembled video from this run's Finish Video node")
    temporary = destination.with_name(f'.{destination.stem}.{uuid.uuid4().hex}.part.mp4')
    try:
        interrupt_check()
        video.save_to(str(temporary), format=Types.VideoContainer.MP4,
                      codec=Types.VideoCodec.AUTO, metadata=metadata)
        verify_video(temporary, timeline, interrupt_check)
        # Match source audio presence; video verification above remains streaming.
        with av.open(video.get_stream_source()) as original, av.open(str(temporary)) as saved:
            if len(original.streams.audio) != len(saved.streams.audio):
                raise ValueError("Saved video did not preserve the source audio stream count")
            audio_streams = len(saved.streams.audio)
        for index in range(audio_streams):
            samples = 0
            with av.open(str(temporary)) as saved:
                for frame in saved.decode(audio=index):
                    interrupt_check()
                    samples += frame.samples
            if not samples:
                raise ValueError("Saved audio stream could not be decoded")
        interrupt_check()
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    cleanup = 'disabled'
    if cleanup_cache:
        interrupt_check()
        try:
            cleanup = cleanup_run(state, output_directory)
        except (OSError, ValueError, KeyError) as error:
            cleanup = f'incomplete: {error}'
            logging.warning('H3 Texture Lab cache cleanup: %s', error)
    return destination, filename, subfolder, cleanup


class H3TextureSaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3TextureSaveVideo',
            display_name='H3 Texture Lab · Save Video & Optional Cache Cleanup',
            category='H3 Texture Lab', is_output_node=True,
            description='Save and fully decode-check the delivered MP4 before optional per-run cache cleanup. Default: keep cache.',
            inputs=[io.Video.Input('video'), io.Custom('H3_TEXTURE_STATE').Input('state'),
                    io.String.Input('filename_prefix', default='h3-relay/texture-lab/ComfyUI'),
                    io.Boolean.Input('cleanup_cache', default=False)],
            outputs=[io.Video.Output('video'), io.String.Output('report')],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, video, state, filename_prefix='h3-relay/texture-lab/ComfyUI', cleanup_cache=False):
        import comfy.model_management
        from comfy.cli_args import args
        from comfy_api.latest import ui
        from comfy_api.input_impl import VideoFromFile

        metadata = {}
        if not args.disable_metadata:
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata['prompt'] = cls.hidden.prompt
        path, filename, subfolder, cleanup = save_delivery(
            video, state, filename_prefix, cleanup_cache,
            comfy.model_management.throw_exception_if_processing_interrupted, metadata or None)
        report = json.dumps({'saved_video': str(path), 'frames': len(state['frames']),
                             'cleanup_cache': cleanup}, indent=2)
        return io.NodeOutput(VideoFromFile(str(path)), report,
                             ui=ui.PreviewVideo([ui.SavedResult(filename, subfolder, io.FolderType.output)]))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')
