import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
import torch
from safetensors.torch import load_file

# Unit-only IO stub; real ComfyUI schemas/execution are validated separately.
api = types.ModuleType('comfy_api.latest')
api.io = types.SimpleNamespace(ComfyNode=object)
sys.modules.setdefault('comfy_api.latest', api)
spec = importlib.util.spec_from_file_location('texture_storage', Path(__file__).resolve().parents[1] / 'storage.py')
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)
plan_spec = importlib.util.spec_from_file_location('texture_storage_planning', Path(__file__).resolve().parents[1] / 'planning.py')
planning = importlib.util.module_from_spec(plan_spec)
plan_spec.loader.exec_module(planning)

class VAE:
    def decode(self, video):
        count = sum((1, 4, 4, 4, 4)[k % 5] for k in range(video.shape[2]))
        values = float(video[0, 0, 0, 0, 0]) + torch.arange(count, dtype=torch.float32)
        return values[:, None, None, None].expand(count, 32, 32, 3).clone()

class StorageTests(unittest.TestCase):
    def test_exact_av_storage_trim_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            folders = types.ModuleType('folder_paths')
            folders.get_output_directory = lambda: directory
            sys.modules['folder_paths'] = folders
            state = None
            for entry in planning.window_plan(158, 56, 22, 7):
                video = torch.full((1, 24, 17, 2, 2), float(entry['index'] * 1000))
                audio = torch.randn(1, 128, 93)
                state = storage.append_window({'samples': [video, audio]}, VAE(), entry, state)
                saved = load_file(str(state['frames'].root / state['av_latent_file']))
                self.assertTrue(torch.equal(saved['video'], video))
                self.assertTrue(torch.equal(saved['audio'], audio))
            timeline = state['frames']
            self.assertEqual(len(timeline), 158)
            self.assertEqual(timeline[50:70][:, 0, 0, 0].tolist(), list(range(50, 56)) + list(range(1022, 1036)))
            self.assertEqual(timeline[-1:][0, 0, 0, 0].item(), 3055)
            first = next(timeline.iter_frames())
            self.assertEqual(first.untyped_storage().nbytes(), first.numel() * first.element_size())
            with self.assertRaises(ValueError):
                timeline.path('../outside.safetensors')

if __name__ == '__main__':
    unittest.main()
