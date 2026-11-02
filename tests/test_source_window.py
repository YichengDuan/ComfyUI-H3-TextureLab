import importlib.util
from pathlib import Path
import sys
import types
import unittest
import torch

api = types.ModuleType('comfy_api.latest')
api.io = types.SimpleNamespace(ComfyNode=object, NodeOutput=lambda *items: items)
sys.modules.setdefault('comfy_api.latest', api)
repo = Path(__file__).resolve().parents[1]
def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, repo / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
source = module('test_source_adapter', 'source_window.py')
planning = module('test_source_planning', 'planning.py')

class SourceWindowTests(unittest.TestCase):
    def test_overlap_uses_exact_source_clock_and_source_audio(self):
        images = torch.arange(56).float()[:, None, None, None].expand(56, 32, 32, 3)
        masks = torch.ones(56, 32, 32)
        windows = {'windows': planning.window_plan(56, 39, 22, 7)}
        wave = torch.arange(112000).float()[None, None, :]
        frames, mask, audio, length = source.H3TextureSourceWindow.execute(
            images, masks, windows, 1, {'sample_rate': 48000, 'waveform': wave})
        self.assertEqual(length, 39)
        self.assertEqual(frames[:, 0, 0, 0].tolist(), list(range(17, 56)))
        self.assertEqual(audio['waveform'].shape[-1], 78000)
        self.assertEqual(audio['waveform'][0, 0, 0].item(), 34000)
        self.assertEqual(audio['waveform'][0, 0, -1].item(), 111999)
        self.assertEqual(tuple(mask.shape), (39, 32, 32))

    def test_short_tail_pads_video_mask_and_audio_without_extending_delivery(self):
        windows = {'windows': planning.window_plan(57, 39, 22, 7)}
        images = torch.arange(57).float()[:, None, None, None].expand(57, 32, 32, 3)
        masks = torch.ones(57, 32, 32)
        frames, _, audio, _ = source.H3TextureSourceWindow.execute(
            images, masks, windows, 2,
            {'sample_rate': 44100, 'waveform': torch.ones(1, 2, round(57 * 44100 / 24))})
        self.assertEqual(frames[:, 0, 0, 0].tolist(), list(range(34, 57)) + [56] * 16)
        self.assertEqual(audio['waveform'].shape[-1], round(73 * 44100 / 24) - round(34 * 44100 / 24))
        self.assertTrue(torch.equal(audio['waveform'][..., -29400:], torch.zeros(1, 2, 29400)))
        self.assertEqual(windows['windows'][-1]['source_end_exclusive'], 57)

if __name__ == '__main__':
    unittest.main()
