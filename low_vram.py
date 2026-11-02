"""Use the native sampler's weight-offload policy without clearing model caches."""
from comfy_api.latest import io


class H3TextureLowVRAM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3TextureLowVRAM',display_name='H3 Texture Lab · Native Weight Offload',category='H3 Texture Lab',
            inputs=[io.Model.Input('model'),io.Boolean.Input('enabled',default=True)],outputs=[io.Model.Output('model')])

    @classmethod
    def execute(cls,model,enabled=True):
        if not enabled:return io.NodeOutput(model)
        from comfy.patcher_extension import WrappersMP
        result=model.clone()
        def prepare(executor,model,noise_shape,conds,model_options=None,force_full_load=False,force_offload=False):
            return executor(model,noise_shape,conds,model_options=model_options,force_full_load=force_full_load,force_offload=True)
        result.add_wrapper(WrappersMP.PREPARE_SAMPLING,prepare)
        return io.NodeOutput(result)
