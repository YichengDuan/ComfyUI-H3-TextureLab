"""Source-aligned video, mask and audio slices for character replacement."""
import torch
from comfy_api.latest import io


class H3TextureSourceWindow(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3TextureSourceWindow",
            display_name="H3 Texture Lab · Character Swap Source Window",
            category="H3 Texture Lab",
            inputs=[
                io.Image.Input("source"), io.Mask.Input("mask"),
                io.Custom("H3_TEXTURE_WINDOWS").Input("windows"),
                io.Int.Input("iteration_index", default=0, min=0),
                io.Audio.Input("source_audio", optional=True),
            ],
            outputs=[io.Image.Output("frames"), io.Mask.Output("mask"),
                     io.Audio.Output("audio"), io.Int.Output("length")],
        )

    @classmethod
    def execute(cls, source, mask, windows, iteration_index=0, source_audio=None):
        entry = windows["windows"][iteration_index]
        start, size = entry["source_start"], entry["window_frames"]
        count = entry["source_frames"]
        if len(source) < count or len(mask) < count or source.shape[1:3] != mask.shape[1:]:
            raise ValueError("Source and mask must cover the same planned timeline and canvas")

        def padded(value):
            selected = value[start:min(start + size, count)]
            missing = size - len(selected)
            return torch.cat((selected, selected[-1:].expand(missing, *selected.shape[1:]))) if missing else selected

        audio = None
        if source_audio is not None:
            rate = source_audio["sample_rate"]
            first, end = round(start * rate / 24), round((start + size) * rate / 24)
            wave = source_audio["waveform"][..., first:end]
            audio = {"sample_rate": rate, "waveform": torch.nn.functional.pad(wave, (0, end - first - wave.shape[-1]))}
        return io.NodeOutput(padded(source), padded(mask), audio, size)
