"""CPU contract for independent windows; no model or private media required."""
from pathlib import Path
import argparse, importlib.util, json, sys, tempfile
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--comfy-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
config = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(config.comfy_root.resolve()))
sys.argv = ['zero-context-contract', '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch, nodes, folder_paths
from comfy_extras.nodes_minimax_h3 import _empty_av_latent, MiniMaxH3AddGuide
spec = importlib.util.spec_from_file_location('zero_context_lab', repo / '__init__.py', submodule_search_locations=[str(repo)])
lab = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lab
spec.loader.exec_module(lab)

class FixtureVAE:
    def encode(self, *args, **kwargs):
        raise AssertionError('Zero context must not re-encode history')
    def decode(self, video):
        blocks = video.shape[2]
        count = blocks // 5 * 17 + sum([1, 4, 4, 4, 4][:blocks % 5])
        return (float(video[0, 0, 0, 0, 0]) + torch.arange(count).float())[:, None, None, None].expand(count, 32, 32, 3).clone()

class ForbiddenHistory:
    def __init__(self):
        raise AssertionError('Zero context must not invoke Motion Context or Join Match')

vae = FixtureVAE()
reference = torch.zeros(1, 32, 32, 3)
positive = [[torch.zeros(1, 1, 1), {'minimax_refs': [{'kind': 'image', 'label': 'one'}, {'kind': 'image', 'label': 'two'}]}]]
plan = lab.H3TextureWindowPlan.execute(83, '39', '0', 7)[3]
results = []
with tempfile.TemporaryDirectory() as directory, patch.dict(nodes.NODE_CLASS_MAPPINGS, {'MiniMaxH3MotionContext': ForbiddenHistory, 'MiniMaxH3JoinMatch': ForbiddenHistory}), patch.object(lab, 'context_latent', side_effect=AssertionError('History must not be loaded')):
    folder_paths.set_output_directory(directory)
    for mode in ['disk', 'ram']:
        state = None
        for index in range(3):
            latent = _empty_av_latent(32, 32, 39)[0]
            latent['samples'].unbind()[0].fill_(index * 1000)
            latent['noise_mask'] = 'source-mask-sentinel'
            output = lab.H3TextureMotionGuide.execute(positive, latent, vae, reference, plan, index, state,
                context_source='pixels_reencoded', first_frame_guide=False, preserve_target_latent=True)
            assert output[0] is positive, 'Reference conditioning changed'
            assert output[1] is latent, 'Source latent/mask changed'
            assert output[4] == 0 and output[5] is False
            state = lab.H3TextureAppendWindow.execute(latent, vae, output[2], state, 'join_match', mode)[0]
        actual = torch.stack([frame[0, 0, 0] for frame in state['frames'].iter_frames()])
        expected = torch.tensor(list(range(39)) + list(range(1000, 1039)) + list(range(2000, 2005)))
        assert torch.equal(actual, expected), 'Frames duplicated, dropped or reordered'
        results.append({'cache_mode': mode, 'windows': 3, 'delivered_frames': len(actual), 'zero_trim': True})
    # Optional first-frame anchoring still applies to each independently started window.
    with patch.object(MiniMaxH3AddGuide, 'execute', return_value=(positive,)) as add_guide:
        for index in range(3):
            output = lab.H3TextureMotionGuide.execute(positive, latent, vae, reference, plan, index, state,
                first_frame_guide=True, preserve_target_latent=True)
            assert output[5] is False
        assert add_guide.call_count == 3
        assert all(call.kwargs['frame_idx'] == 0 for call in add_guide.call_args_list)
report = {'success': True, 'cases': results, 'motion_context_calls': 0, 'join_match_calls': 0,
    'history_loads': 0, 'history_vae_encode_calls': 0, 'reference_conditioning_and_source_latent_preserved': True,
    'optional_first_frame_anchor_on_each_independent_window': True,
    'scope': 'Native ComfyUI CPU node contract with synthetic latent/frame fixtures; no H3 inference or quality claim.'}
config.output.parent.mkdir(parents=True, exist_ok=True)
config.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
