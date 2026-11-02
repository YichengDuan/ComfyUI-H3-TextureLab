import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch

api = types.ModuleType('comfy_api.latest')
api.io = types.SimpleNamespace(ComfyNode=object, NodeOutput=lambda *items: items)
sys.modules.setdefault('comfy_api.latest', api)
package = types.ModuleType('texture_ram_test')
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
storage = importlib.import_module(package.__name__ + '.storage')
planning = importlib.import_module(package.__name__ + '.planning')

class VAE:
    def decode(self, video):
        count = sum((1, 4, 4, 4, 4)[i % 5] for i in range(video.shape[2]))
        values = float(video[0, 0, 0, 0, 0]) + torch.arange(count).float()
        return values[:, None, None, None].expand(count, 32, 32, 3).clone()

class RamTests(unittest.TestCase):
    def make(self, index=0):
        return {'samples': [torch.full((1, 24, 17, 2, 2), float(index * 1000)), torch.randn(1, 32, 2, 93)]}

    def test_ram_cache_has_no_files_and_keeps_owned_exact_cpu_tensors(self):
        state = None
        prior = None
        entries = planning.window_plan(158, 56, 22, 7)
        with patch.object(storage, 'save_file', side_effect=AssertionError('Unexpected disk write')), \
             patch.object(Path, 'mkdir', side_effect=AssertionError('Unexpected cache folder')), \
             patch.object(Path, 'write_text', side_effect=AssertionError('Unexpected manifest')):
            for entry in entries:
                av = self.make(entry['index'])
                before = [t.clone() for t in av['samples']]
                prior = state
                state = storage.append_window(av, VAE(), entry, state, 'ram')
                stored = state['frames'].load_av(state['av_latent_file'])
                self.assertTrue(torch.equal(stored['video'], before[0]))
                self.assertTrue(torch.equal(stored['audio'], before[1]))
                self.assertEqual(stored['video'].device.type, 'cpu')
                av['samples'][0].add_(99)
                stored['video'].zero_()
                self.assertTrue(torch.equal(state['frames'].load_av(state['av_latent_file'])['video'], before[0]))
        timeline = state['frames']
        self.assertIsNone(timeline.root)
        self.assertEqual(len(timeline), 158)
        self.assertEqual(timeline[50:70][:, 0, 0, 0].tolist(), list(range(50, 56)) + list(range(1022, 1036)))
        frame = next(timeline.iter_frames())
        self.assertEqual(frame.untyped_storage().nbytes(), frame.numel() * frame.element_size())
        # The prefix and latest states share cache ownership; cleanup releases this run only.
        other = storage.append_window(self.make(), VAE(), entries[0], cache_mode='ram')
        timeline.release()
        with self.assertRaisesRegex(ValueError, 'released'):
            prior['frames'].load_av(prior['av_latent_file'])
        self.assertTrue(other['frames'].load_av(other['av_latent_file']))

    def test_cache_mode_cannot_change_during_a_run(self):
        entries = planning.window_plan(158, 56, 22, 7)
        state = storage.append_window(self.make(), VAE(), entries[0], cache_mode='ram')
        with self.assertRaisesRegex(ValueError, 'partway'):
            storage.append_window(self.make(1), VAE(), entries[1], state, 'disk')
        with self.assertRaisesRegex(ValueError, 'disk or ram'):
            storage.append_window(self.make(), VAE(), entries[0], cache_mode='invalid')

if __name__ == '__main__':
    unittest.main()
