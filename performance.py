"""Opt-in H3 FFN token chunks and experimental native Kitchen modulation."""
import torch
import comfy.ops
from comfy_api.latest import io
from comfy.ldm.minimax.model import MiniMaxH3Model, MLP, _mod_gate, _mod_row
from comfy.quant_ops import QuantizedTensor


class ChunkedMLP:
    def __init__(self, forward, rows):
        self.forward = forward
        self.rows = rows
        self.module = forward.__self__ if getattr(forward, '__func__', None) is MLP.forward else None

    def __call__(self, x):
        if x.ndim != 2 or x.shape[0] <= self.rows:
            return self.forward(x)
        output = torch.empty_like(x)
        first = self.module.fc1 if self.module is not None else None
        second = self.module.fc2 if self.module is not None else None
        prepared = self.module is not None and all(isinstance(layer.weight, QuantizedTensor)
            and not layer.comfy_force_forward and not layer._full_precision_mm
            and getattr(layer, 'pre_quant_scale', None) is None
            and getattr(layer, 'input_scale', None) is None for layer in (first, second))
        if prepared:
            with comfy.ops.CastBiasWeightContext(first, x, offloadable=True, compute_dtype=x.dtype, want_requant=True) as (w1, b1), \
                    comfy.ops.CastBiasWeightContext(second, x, offloadable=True, compute_dtype=x.dtype, want_requant=True) as (w2, b2):
                for start in range(0, x.shape[0], self.rows):
                    h = comfy.ops.linear_input_act_(x[start:start+self.rows], w1, b1)
                    output[start:start+self.rows].copy_(comfy.ops.linear_input_act_(h, w2, b2, 'swiglu'))
                    del h
            return output
        for start in range(0, x.shape[0], self.rows):
            output[start:start+self.rows].copy_(self.forward(x[start:start+self.rows]))
        return output


def fused_norm_modulation(norm, x, shift, scale, segments):
    import comfy_kitchen as ck
    chunks = []
    with comfy.ops.CastBiasWeightContext(norm, x, offloadable=True) as (weight, _):
        for start, stop, row in segments:
            factor = 1.0 + _mod_row(scale, row, x.dtype)
            folded = (weight.float() * factor.float() - 1.0).to(x.dtype)
            chunks.append(ck.rms_adaln(x[start:stop], folded, _mod_row(shift, row, x.dtype), norm.eps))
    return torch.cat(chunks, dim=0)


class FusedDiTForward:
    def __init__(self, block):
        self.block = block

    def __call__(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        block = self.block
        attention = block.attn if attention is None else attention
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        h = fused_norm_modulation(block.norm1, x, shift_msa, scale_msa, mod_segments)
        x = _mod_gate(x, gate_msa, attention(h, rope_freqs=rope_freqs, transformer_options=transformer_options), mod_segments)
        h = fused_norm_modulation(block.norm2, x, shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, block.mlp(h), mod_segments)


class H3TexturePerformance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3TexturePerformance', display_name='H3 Texture Lab · FFN / Experimental Modulation',
            category='H3 Texture Lab',
            description='FFN chunks reduce activation memory. CPU weight offloading can make chunks slower. Fused modulation changes BF16 rounding; experimental and off by default.',
            inputs=[io.Model.Input('model'), io.Int.Input('ffn_chunk_tokens', default=0, min=0, max=32768, step=256),
                io.Boolean.Input('fused_modulation', default=False)], outputs=[io.Model.Output('model')])

    @classmethod
    def execute(cls, model, ffn_chunk_tokens=0, fused_modulation=False):
        diffusion = model.get_model_object('diffusion_model')
        if not isinstance(diffusion, MiniMaxH3Model):
            raise ValueError('H3 performance options require a MiniMax H3 model')
        result = model.clone()
        if ffn_chunk_tokens:
            for name, module in diffusion.named_modules():
                if isinstance(module, MLP):
                    path = f'diffusion_model.{name}.forward'
                    forward = model.get_model_object(path)
                    if isinstance(forward, ChunkedMLP):
                        forward = forward.forward
                    result.add_object_patch(path, ChunkedMLP(forward, ffn_chunk_tokens))
        if fused_modulation:
            for index, block in enumerate(diffusion.blocks):
                result.add_object_patch(f'diffusion_model.blocks.{index}.forward', FusedDiTForward(block))
        return io.NodeOutput(result)


NODE_CLASS_MAPPINGS={'H3TexturePerformance':H3TexturePerformance}
