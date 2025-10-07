import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
# from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
# from transformers.models.mistral.modeling_mistral import (
#     MistralDecoderLayer,
#     MistralRMSNorm,
# )
# from transformers.models.mixtral.modeling_mixtral import (
#     MixtralDecoderLayer,
#     MixtralRMSNorm,
# )
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer

def get_act_masks(act_scales, threshold=1, perc=0.1):

    # act_masks = {} 

    # keys = list(act_scales.keys())   

    # for key in keys: 
    x = act_scales.float()
    mean_val = x.mean().item()
    median_val = x.median().item()
    q1 = x.quantile(0.25).item()  # 25th percentile
    q3 = x.quantile(0.75).item()
    iqr = q3 - q1 
    lower_fence = q1 - threshold * iqr
    upper_fence = q3 + threshold * iqr

    q11 = x.quantile(0.45).item()
    q33 = x.quantile(0.55).item()
    iqr2 = q33 - q11 
    # lower_fence_2 = q1 - threshold * iqr
    # upper_fence_2 = q33 #+ iqr2
    lower_fence_2 = (median_val) - (perc *median_val)
    upper_fence_2 = (median_val) + (perc *median_val)

    # Count outliers
    below_fence = (x < lower_fence).sum().item()
    above_fence = (x > upper_fence).sum().item()

    # mask = torch.where((x >= lower_fence) & (x <= upper_fence), 
    #                torch.ones_like(x), 
    #                x)
    # mask = torch.where((x >= lower_fence) & (x <= upper_fence),
    #                     torch.zeros_like(x),   # inside → 0
    #                     torch.ones_like(x))    # outside → 1
    
    mask = torch.where((x >= lower_fence_2) & (x <= upper_fence_2),
                        torch.zeros_like(x),   # inside → 0
                        torch.ones_like(x))    # outside → 1

        # act_masks[key] = mask

    return mask

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5, masking=False, r=1, perc=0.1):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    
    # masks = masks.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    
    if masking:
        masks = get_act_masks(act_scales, threshold=r, perc=perc) 
        act_scales_masked = torch.where(masks == 0, torch.ones_like(act_scales), act_scales)
        weight_scales_masked = torch.where(masks == 0, torch.ones_like(weight_scales), weight_scales)
        masked_scales = (
            (act_scales_masked.pow(alpha) / weight_scales_masked.pow(1 - alpha))
            .clamp(min=1e-5)
            .to(device)
            .to(dtype)
        )
        # masked_scales = torch.where(masks == 0, torch.full_like(masked_scales, 1e-5), masked_scales)
        ln.weight.div_(masked_scales)
        ln.bias.div_(masked_scales)

        for fc in fcs:
            fc.weight.mul_(masked_scales.view(1, -1))
    else:
        ln.weight.div_(scales)
        ln.bias.div_(scales)

        for fc in fcs:
            fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5, masking=False, r=1):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, (LlamaRMSNorm, MistralRMSNorm, MixtralRMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    
    # masks = masks.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    
    if masking: 
        masks = get_act_masks(act_scales, threshold=r) 
        act_scales_masked = torch.where(masks == 0, torch.ones_like(act_scales), act_scales)
        weight_scales_masked = torch.where(masks == 0, torch.ones_like(weight_scales), weight_scales) 
        
        masked_scales = (
            (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
            .clamp(min=1e-5)
            .to(device)
            .to(dtype)
        )

        ln.weight.div_(masked_scales)
        for fc in fcs:
            fc.weight.mul_(masked_scales.view(1, -1))
    else:
        ln.weight.div_(scales)
        for fc in fcs:
            fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5, masking=False, r=1, perc=0.1):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            # qkv_input_masks = masks[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha, masking, r, perc)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            # fc1_input_masks = masks[name + ".fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha, masking, r, perc)
        # elif isinstance(module, BloomBlock):
        #     attn_ln = module.input_layernorm
        #     qkv = module.self_attention.query_key_value
        #     qkv_input_scales = scales[name + ".self_attention.query_key_value"]
        #     smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

        #     ffn_ln = module.post_attention_layernorm
        #     fc1 = module.mlp.dense_h_to_4h
        #     fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
        #     smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, FalconDecoderLayer):
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            # qkv_input_masks = masks[name + ".self_attention.query_key_value"]
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            # fc1_input_masks = masks[name + ".mlp.dense_h_to_4h"]
            fc1 = module.mlp.dense_h_to_4h

            if (
                not module.config.new_decoder_architecture
                and module.config.parallel_attn
            ):
                attn_ln = module.input_layernorm
                smooth_ln_fcs(attn_ln, [qkv, fc1], qkv_input_scales, alpha, masking, r)
            else:
                attn_ln = (
                    module.ln_attn
                    if module.config.new_decoder_architecture
                    else module.input_layernorm
                )
                ffn_ln = (
                    module.ln_mlp
                    if module.config.new_decoder_architecture
                    else module.post_attention_layernorm
                )
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha, masking, r)
                smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha, masking, r)
        elif isinstance(module, LlamaDecoderLayer): # (LlamaDecoderLayer, MistralDecoderLayer)
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            # qkv_input_masks = masks[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha, masking, r)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]
            # fcs_input_masks = masks[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha, masking, r)
        # elif isinstance(module, MixtralDecoderLayer):
        #     attn_ln = module.input_layernorm  # attention forward norm
        #     qkv = [
        #         module.self_attn.q_proj,
        #         module.self_attn.k_proj,
        #         module.self_attn.v_proj,
        #     ]

        #     qkv_input_scales = scales[name + ".self_attn.q_proj"]
        #     smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

        #     ffn_ln = module.post_attention_layernorm  # feed forward norm
        #     fcs = [module.block_sparse_moe.gate]
        #     for expert in module.block_sparse_moe.experts:
        #         fcs.append(expert.w1)
        #         fcs.append(expert.w3)
        #     fcs_input_scales = scales[name + ".block_sparse_moe.gate"]

        #     smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
