import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

try:
    from mamba_ssm import Mamba

    print("successfully imported Mamba from mamba-ssm")
except ImportError:
    Mamba = None


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MambaOriginal(nn.Module):
    def __init__(self, config):
        super().__init__()

        VALID_NORMALIZATION_LAYERS = {
            "RMSNorm": RMSNorm,
            "LayerNorm": LayerNorm,
            "LearnableLayerScaling": LearnableLayerScaling,
            "BatchNorm": BatchNorm,
            "InstanceNorm": InstanceNorm,
            "GroupNorm": GroupNorm,
        }

        # Get normalization layer from config
        norm = config.norm
        if isinstance(norm, str) and norm in VALID_NORMALIZATION_LAYERS:
            self.norm_f = VALID_NORMALIZATION_LAYERS[norm](
                config.d_model, eps=config.layer_norm_eps
            )
        else:
            raise ValueError(
                f"Invalid normalization layer: {norm}. "
                f"Valid options are: {', '.join(VALID_NORMALIZATION_LAYERS.keys())}"
            )

        # Initialize Mamba layers based on the configuration
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand_factor,
                    dt_rank=config.dt_rank,
                    dt_min=config.dt_min,
                    dt_max=config.dt_max,
                    dt_init=config.dt_init,
                    dt_scale=config.dt_scale,
                    dt_init_floor=config.dt_init_floor,
                    conv_bias=config.conv_bias,
                    bias=config.bias,
                    use_fast_path=True,  # Fused kernel options
                    layer_idx=i,
                )
                for i in range(config.n_layers)
            ]
        )

        # Apply weight initialization
        self.apply(
            lambda m: _init_weights(
                m,
                n_layer=config.n_layers,
                n_residuals_per_layer=1 if config.d_state == 0 else 2,
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                inference_params=inference_params,
                **mixer_kwargs,
            )

        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )

        return hidden_states
