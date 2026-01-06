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


MODEL_NAME = 'facebook/opt-125m'
# MODEL_NAME = 'facebook/opt-350m'
# MODEL_NAME = 'facebook/opt-1.3b'
# MODEL_NAME = 'facebook/opt-2.7b'
# MODEL_NAME = 'facebook/opt-6.7b'
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
        comming_max = torch.max(x, dim=0, keepdim=True)[0].float().cpu()

        if name in act_scales: 
            act_scales[name] = torch.cat([act_scales[name], comming_max], dim=0) 
        
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


act_scales = get_act_scales(model, tokenizer, DATASET_PATH, NUM_SAMPLES, SEQ_LEN)    
# act_masks = get_act_masks(act_scales, range=0.9)
out_p = '/home/tahir/workspace2/MaskedSmoothing/act_scales_modified_3/opt125m_max_per_exp.pt'
os.makedirs(os.path.dirname(out_p), exist_ok=True)
torch.save(act_scales, out_p)

# os.makedirs(os.path.dirname(OUTPUT_PATH_SCALES), exist_ok=True)
# torch.save(act_scales, OUTPUT_PATH_SCALES)

# os.makedirs(os.path.dirname(OUTPUT_PATH_MASKS), exist_ok=True)
# torch.save(act_masks, OUTPUT_PATH_MASKS)