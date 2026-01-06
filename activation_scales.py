import torch
import torch.nn as nn

import os
# from activation_scale_utils import * 

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from datasets import load_dataset
import functools

from tqdm import tqdm


# MODEL_NAME = 'facebook/opt-125m'
# MODEL_NAME = 'facebook/opt-1.3b'
# MODEL_NAME = 'facebook/opt-2.7b'
MODEL_NAME = 'facebook/opt-6.7b'
# OUTPUT_PATH_SCALES = 'act_scales/opt-350m.pt'
# OUTPUT_PATH_MASKS = 'act_scales/masked_opt-350m.pt'
DATASET_PATH = 'dataset/val.jsonl.zst' 
NUM_SAMPLES = 512 
SEQ_LEN = 512 
MODEL_MAX_LENGTH = 512  


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=MODEL_MAX_LENGTH)
kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)


def get_act_scales(model, tokenizer, dataset_path, num_samples, seq_len):
    
    model.eval()
    device = next(model.parameters()).device

    act_scales = {}

    def stat_input_hook(m, x, y, name):

        if isinstance(x, tuple):
            x = x[0]

        hidden_dim = x.shape[-1] # embedding/channel dimension
        x = x.view(-1, hidden_dim).abs().detach() # [batch x token, channel]
        comming_max = torch.max(x, dim=0)[0].float().cpu()

        if name in act_scales: 
            act_scales[name] = torch.max(act_scales[name], comming_max) 
        
        else: 
            act_scales[name] = comming_max 

    hooks = []

    for name, m in model.named_modules():
        
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(num_samples)): 
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)

        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales

def get_act_masks(act_scales, range=1):

    act_masks = {} 

    keys = list(act_scales.keys())   

    for key in keys: 
        x = act_scales[key] 
        mean_val = x.mean().item()
        median_val = x.median().item()
        q1 = x.quantile(0.25).item()  # 25th percentile
        q3 = x.quantile(0.75).item()
        iqr = q3 - q1 
        lower_fence = q1 - range * iqr
        upper_fence = q3 + range * iqr

        # Count outliers
        below_fence = (x < lower_fence).sum().item()
        above_fence = (x > upper_fence).sum().item()

        # mask = torch.where((x >= lower_fence) & (x <= upper_fence), 
        #                torch.ones_like(x), 
        #                x)
        mask = torch.where((x >= lower_fence) & (x <= upper_fence),
                         torch.zeros_like(x),   # inside → 0
                         torch.ones_like(x))    # outside → 1

        act_masks[key] = mask

    return act_masks

########## KIBME STAT MAX ##########
def get_act_scales_2(model, tokenizer, dataset_path, num_samples, seq_len):
    """
    Compute per-example max activations for each Linear layer in the model.

    Returns:
        act_scales: dict[layer_name] = tensor of shape [num_samples, hidden_dim]
    """
    model.eval()
    device = next(model.parameters()).device

    # Initialize storage for each layer
    act_scales = {}
    hooks = []

    # Hook function
    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]  # unpack if input is a tuple

        hidden_dim = x.shape[-1]  # embedding/channel dimension
        x = x.view(-1, hidden_dim).abs().detach()  # flatten batch and seq
        example_max = torch.max(x, dim=0)[0].float().cpu()  # max per channel

        # Append per-example max to the list
        act_scales[name].append(example_max)

    # Attach hooks to all Linear layers
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            act_scales[name] = []  # initialize empty list for each layer
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    # Load and shuffle dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    # Run forward passes and collect per-example max activations
    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["text"],
            return_tensors="pt",
            max_length=seq_len,
            truncation=True
        ).input_ids.to(device)

        model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Convert lists to tensors [num_samples, hidden_dim]
    for name in act_scales:
        act_scales[name] = torch.stack(act_scales[name], dim=0)

    return act_scales
print("MAX VALUE PER EXAMPLE DONE")
act_scales = get_act_scales_2(model, tokenizer, DATASET_PATH, NUM_SAMPLES, SEQ_LEN)    
# act_masks = get_act_masks(act_scales, range=0.9)

# out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_stat_max/opt125m_stat_max_512.pt'
# out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_stat_max/opt1b_stat_max_512.pt'
# out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_stat_max/opt2b_stat_max_512.pt'
out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_stat_max/opt6b_stat_max_512.pt'
os.makedirs(os.path.dirname(out_p), exist_ok=True)
torch.save(act_scales, out_p)

from scipy.stats import genpareto
import numpy as np
def evt_scale_channel(values, threshold_quantile=0.95, target_p=0.0001):
    """
    values: Tensor of shape [512] for a single channel
    threshold_quantile: the quantile to define the tail
    target_p: desired exceedance probability (e.g., 0.0001 = 99.99%)

    Returns:
        EVT-based scale estimate (float)
    """

    v = values.cpu().numpy()

    # Step 1: threshold
    u = np.quantile(v, threshold_quantile)

    # Step 2: exceedances
    y = v[v > u] - u
    if len(y) < 5:
        # not enough tail data — fallback to raw max
        return float(v.max())

    # Step 3: fit GPD (xi = shape, beta = scale)
    params = genpareto.fit(y)   # fits shape, loc, scale
    xi, loc, beta = params

    # Step 4: EVT quantile
    if abs(xi) < 1e-6:
        # limit as xi → 0
        q = u + beta * np.log(1/target_p)
    else:
        q = u + (beta / xi) * ((target_p ** (-xi)) - 1)

    return float(q)

# print("CHANNEL WISE STATS")
# act_scale_final = {}
# for name, X in act_scales.items(): 
#     # print(f"{k}: {tuple(v.shape)}")
#     # print(type(v))
#     Xc = X.transpose(0,1)
    
#     scales = [] 
#     print(f"name: {name}")
#     for c in tqdm(range(Xc.shape[0])):
#         scales.append(evt_scale_channel(
#             Xc[c],
#             threshold_quantile=0.95,
#             target_p=0.0001
#         ))
    
#     act_scale_final[name] = torch.tensor(scales)


# out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_stat_max/opt125m_stat_max.pt'
# out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_stat_max/opt1b_stat_max.pt'
# out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_stat_max/opt2b_stat_max.pt'
# # out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_stat_max/opt6bm_max.pt'
# os.makedirs(os.path.dirname(out_p), exist_ok=True)
# torch.save(act_scale_final, out_p)


def evt_thresholds(x, k=20, tail='right'):
    """
    x: tensor of shape (N, C) = (512, 768)
    k: number of top samples for tail fitting
    returns: thresholds of shape (C,)
    """

    # ---- STEP 1: Get top-k values per channel (parallel) ----
    # topk_vals: (C, k)
    topk_vals, _ = torch.topk(x, k, dim=0)
    
    # reshape to (C, k)
    topk_vals = topk_vals.transpose(0, 1)

    # ---- STEP 2: Compute GPD parameters using PWM estimator (parallel) ----
    # PWM method is very stable and fully vectorizable.

    # u = threshold (smallest of top-k)
    u = topk_vals[:, -1]   # shape (C,)

    # excess = x - u
    excess = topk_vals - u.unsqueeze(1)  # (C, k)

    # PWM estimates
    b0 = excess.mean(dim=1)                                  # (C,)
    i = torch.arange(1, k+1, device=x.device).float()
    b1 = (excess * (i / (k + 1))).mean(dim=1)                # (C,)

    # shape parameter (ξ)
    xi = (2 * b1 - b0) / (b0 - 2 * b1 + 1e-9)               # (C,)

    # scale parameter (σ)
    sigma = (2 * b0 * b1) / (b0 - 2 * b1 + 1e-9)            # (C,)

    # ---- STEP 3: Choose a high quantile threshold ----
    # Typically q = 0.999 to get "max_plus_margin"
    q = 0.999

    # EVT quantile per channel
    evt_thresh = u + (sigma / xi) * ((1 - q) ** (-xi) - 1)

    return evt_thresh

# print("CHANNEL WISE PARALLEL STATS")
# act_scale_final = {}
# for name, X in tqdm(act_scales.items()): 
#     thresholds = evt_thresholds(X, k=20)
    
#     act_scale_final[name] = thresholds

# out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_stat_max/opt125m_stat_max_parallel.pt'
# out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_stat_max/opt1b_stat_max_parallel.pt'
# out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_stat_max/opt2b_stat_max_parallel.pt'
# out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_stat_max/opt6bm_stat_max_parallel.pt'
# os.makedirs(os.path.dirname(out_p), exist_ok=True)
# torch.save(act_scale_final, out_p)

########## KIBME STAT MAX ##########


# act_scales = get_act_scales(model, tokenizer, DATASET_PATH, NUM_SAMPLES, SEQ_LEN)    
# # act_masks = get_act_masks(act_scales, range=0.9)
# out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_logsum/1b/opt1b_max.pt'
# os.makedirs(os.path.dirname(out_p), exist_ok=True)
# torch.save(act_scales, out_p)

# os.makedirs(os.path.dirname(OUTPUT_PATH_SCALES), exist_ok=True)
# torch.save(act_scales, OUTPUT_PATH_SCALES)

# os.makedirs(os.path.dirname(OUTPUT_PATH_MASKS), exist_ok=True)
# torch.save(act_masks, OUTPUT_PATH_MASKS)