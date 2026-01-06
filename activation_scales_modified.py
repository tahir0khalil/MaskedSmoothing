import torch
import torch.nn as nn

import os
import activation_scale_utils

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from datasets import load_dataset
import functools

from tqdm import tqdm


MODEL_NAME = 'facebook/opt-125m'
# MODEL_NAME = 'facebook/opt-350m' #400 examples
# MODEL_NAME = 'facebook/opt-1.3b' # 180
# MODEL_NAME = 'facebook/opt-2.7b' # 90
# MODEL_NAME = 'facebook/opt-6.7b' #30

# OUTPUT_PATH_SCALES = 'act_scales_modified/opt-125m.pt'
# OUTPUT_PATH_SCALES = 'act_scales_modified/opt-7b.pt'
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
        # comming_max = torch.max(x, dim=0)[0].float().cpu()

        if name in act_scales: 
            # act_scales[name] = torch.max(act_scales[name], comming_max)
            act_scales[name] = torch.cat([act_scales[name], x], dim=0) 
        
        else: 
            # act_scales[name] = comming_max
            act_scales[name] = x 

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



act_scales = get_act_scales(model, tokenizer, DATASET_PATH, NUM_SAMPLES, SEQ_LEN)    
torch.save(act_scales, '/home/tahir/workspace2/MaskedSmoothing/act_scales_logsum/opt-125m_reference.pt') 

methods = {
    # "1_max": activation_scale_utils.scale_max,
    # "2_mean": activation_scale_utils.scale_mean,
    # "3_median": activation_scale_utils.scale_median,
    # "4_percentile": activation_scale_utils.scale_percentile,
    # "5_top_k_mean": activation_scale_utils.scale_topk_mean,
    # "6_scale_trimmed_mean": activation_scale_utils.scale_trimmed_mean,
    # "7_huber_estimator": activation_scale_utils.huber_estimator,
    # "8_generalized_mean": activation_scale_utils.generalized_mean,
    "9_logsumexp_magnitude": activation_scale_utils.logsumexp_magnitude,
    # "10_high_quantile_gaussian": activation_scale_utils.high_quantile_gaussian,
    # "11_high_quantile_gpd": activation_scale_utils.high_quantile_gpd,
#     "12_empirical_bayes_per_channel": activation_scale_utils.empirical_bayes_per_channel,
#     "13_evt_per_channel_scale": activation_scale_utils.evt_per_channel_scale,
#     "14_robust_per_channel_scale": activation_scale_utils.robust_per_channel_scale,
}

# for name, func in methods.items(): 
#     print(f"Running the function: {name}")
#     results = {} 
#     OUTPUT_PATH_SCALES = 'act_scales_modified/opt-125m_b_30_' + name + '.pt'
#     for key, val in tqdm(act_scales.items()):
#         result = func(val) 
#         results[key] = result
#     print("==============")
#     torch.save(results, OUTPUT_PATH_SCALES)
    


# os.makedirs(os.path.dirname(OUTPUT_PATH_SCALES), exist_ok=True)
# torch.save(act_scales, OUTPUT_PATH_SCALES)

