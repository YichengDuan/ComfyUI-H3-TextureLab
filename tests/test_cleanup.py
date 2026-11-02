import importlib
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import json

import av
import numpy as np
import torch

# Unit IO definitions only; a separate smoke test uses real ComfyUI node schemas.
api = types.ModuleType('comfy_api.latest')
api.io = types.SimpleNamespace(ComfyNode=object, NodeOutput=lambda *items: items)
sys.modules.setdefault('comfy_api.latest', api)
active_api = sys.modules['comfy_api.latest']
active_api.Types = types.SimpleNamespace(VideoContainer=types.SimpleNamespace(MP4='mp4'), VideoCodec=types.SimpleNamespace(AUTO='auto'))
package = types.ModuleType('texture_cleanup_test')
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
saving = importlib.import_module(package.__name__ + '.save_video')
storage = importlib.import_module(package.__name__ + '.storage')


def fixture_video(path, frames=5, audio=False):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        sound = container.add_stream('aac', rate=48000) if audio else None
        if sound is not None:
            sound.layout = 'stereo'
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 20, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if sound is not None:
            frame = av.AudioFrame.from_ndarray(np.zeros((2, frames * 2000), dtype=np.float32), format='fltp', layout='stereo')
            frame.sample_rate = 48000
            for packet in sound.encode(frame):
                container.mux(packet)
            for packet in sound.encode():
                container.mux(packet)


class FileVideo:
    def __init__(self, path, save_source=None, error=None):
        self.path, self.save_source, self.error = path, save_source, error

    def get_stream_source(self):
        return str(self.path)

    def save_to(self, path, **kwargs):
        if self.error:
            Path(path).write_bytes(b'partial')
            raise self.error
        shutil.copyfile(self.save_source or self.path, path)


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        folders = types.ModuleType('folder_paths')
        folders.get_output_directory = lambda: str(self.output)
        def save_path(prefix, output, *dimensions):
            path = Path(output) / prefix
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path.parent), path.name, 1, str(path.parent.relative_to(output)), prefix
        folders.get_save_image_path = save_path
        self.folder_patch = patch.dict(sys.modules, {'folder_paths': folders})
        self.folder_patch.start()
        self.addCleanup(self.folder_patch.stop)
        window = {'index': 0, 'source_start': 0, 'source_end_exclusive': 5,
                  'source_frames': 5, 'discarded_overlap': 0}
        self.state = storage.append_window(
            {'samples': [torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 128, 9)]}, None, window)
        self.root = self.state['frames'].root
        self.source = self.root / 'assembled.mp4'
        fixture_video(self.source, audio=True)
        self.video = FileVideo(self.source)

    def save(self, video=None, cleanup=True, prefix='delivery/result', check=lambda: None):
        return saving.save_delivery(video or self.video, self.state, prefix, cleanup, check)

    def test_cleanup_default_off_keeps_cache(self):
        path, _, _, result = self.save(cleanup=False)
        self.assertEqual(result, 'disabled')
        self.assertTrue(self.root.exists())
        self.assertTrue(path.exists())

    def test_success_deletes_only_this_run_and_video_still_decodes(self):
        neighbor = self.root.parent / ('b' * 32)
        neighbor.mkdir()
        (neighbor / 'keep').write_text('another run')
        path, _, _, result = self.save()
        self.assertEqual(result, 'deleted')
        self.assertFalse(self.root.exists())
        self.assertTrue((neighbor / 'keep').exists())
        with av.open(str(path)) as video:
            self.assertEqual(sum(1 for _ in video.decode(video=0)), 5)
        with av.open(str(path)) as video:
            self.assertGreater(sum(f.samples for f in video.decode(audio=0)), 0)

    def test_save_failure_and_partial_file_keep_cache(self):
        with self.assertRaises(OSError):
            self.save(FileVideo(self.source, error=OSError('write failure')))
        self.assertTrue((self.root / 'manifest.json').exists())
        self.assertEqual(list((self.output / 'delivery').glob('*.mp4')), [])

    def test_truncated_frame_sequence_keeps_cache(self):
        short = self.output / 'short.mp4'
        fixture_video(short, frames=3, audio=True)
        with self.assertRaisesRegex(ValueError, 'complete timeline'):
            self.save(FileVideo(self.source, save_source=short))
        self.assertTrue(self.root.exists())

    def test_lost_audio_keeps_cache(self):
        silent = self.output / 'no-audio.mp4'
        fixture_video(silent)
        with self.assertRaisesRegex(ValueError, 'audio stream'):
            self.save(FileVideo(self.source, save_source=silent))
        self.assertTrue(self.root.exists())

    def test_cancel_during_verification_keeps_cache(self):
        checks = 0
        def interrupt():
            nonlocal checks
            checks += 1
            if checks == 3:
                raise InterruptedError('cancelled')
        with self.assertRaises(InterruptedError):
            self.save(check=interrupt)
        self.assertTrue(self.root.exists())

    def test_destination_inside_cache_is_rejected(self):
        prefix = str(self.source.relative_to(self.output).with_suffix(''))
        with self.assertRaisesRegex(ValueError, 'outside the cache'):
            self.save(prefix=prefix)
        self.assertTrue(self.root.exists())

    def test_other_video_cannot_authorize_deletion(self):
        unrelated = self.output / 'other.mp4'
        fixture_video(unrelated, audio=True)
        with self.assertRaisesRegex(ValueError, 'from this run'):
            self.save(FileVideo(unrelated))
        self.assertTrue(self.root.exists())

    def test_unknown_files_or_links_skip_entire_cleanup(self):
        extra = self.root / 'personal.txt'
        extra.write_text('keep')
        _, _, _, result = self.save()
        self.assertTrue(result.startswith('skipped'))
        self.assertTrue(extra.exists())
        self.assertTrue((self.root / 'manifest.json').exists())
        extra.unlink()
        (self.root / 'link').symlink_to(self.output)
        _, _, _, result = self.save()
        self.assertTrue(result.startswith('skipped'))
        self.assertTrue((self.root / 'manifest.json').exists())

    def test_unrelated_directory_is_not_deleted(self):
        unrelated = self.output / 'unrelated'
        self.root.rename(unrelated)
        self.state['frames'].root = unrelated
        self.video = FileVideo(unrelated / 'assembled.mp4')
        _, _, _, result = self.save()
        self.assertTrue(result.startswith('skipped'))
        self.assertTrue((unrelated / 'manifest.json').exists())

    def test_symlinked_run_directory_is_not_deleted(self):
        target = self.output / 'linked-target'
        self.root.rename(target)
        self.root.symlink_to(target, target_is_directory=True)
        _, _, _, result = self.save()
        self.assertTrue(result.startswith('skipped'))
        self.assertTrue((target / 'manifest.json').exists())

    def test_mismatched_manifest_skips_cleanup(self):
        manifest = self.root / 'manifest.json'
        data = json.loads(manifest.read_text())
        data['source_frames'] = 99
        manifest.write_text(json.dumps(data))
        _, _, _, result = self.save()
        self.assertTrue(result.startswith('skipped'))
        self.assertTrue(manifest.exists())


if __name__ == '__main__':
    unittest.main()
