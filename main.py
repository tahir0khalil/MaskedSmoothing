from quant import * 
from evals import * 
from smoothing import smooth_lm

import torch
from torch import nn
import numpy as np 
import pandas as pd 

from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoderLayer,
    OPTForCausalLM,
)
from transformers import GPT2Tokenizer 
from datasets import load_dataset 
import tqdm 

from functools import partial 

# MODEL_NAME = 'facebook/opt-125m'
# MODEL_NAME = 'facebook/opt-350m'
# MODEL_NAME = 'facebook/opt-1.3b'
# MODEL_NAME = 'facebook/opt-2.7b'
MODEL_NAME = 'facebook/opt-6.7b'

W_BITS = 8
A_BITS = 8

W_QUANT = 'per_tensor' # [per_channel, per_tensor]
A_QUANT = 'per_tensor' # [per_token, per_tensor] 


print(f"initializing tokenizer")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
print(f"initializing dataset")
dataset = load_dataset("cimec/lambada", split="test")

evaluator_last_token = Evaluator(dataset, tokenizer, "cuda")
evaluator_perplexity = EvaluatorPerplexity(dataset, tokenizer, "cuda")
#################### FP16 ####################
# print("loading model")
# model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto") 
# model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=None).to("cuda") 

# print("running eval of FP16 model")
# fp16_acc = evaluator_last_token.evaluate(model_fp16) 
# fp16_prep = evaluator_perplexity.evaluate(model_fp16) 

#################### Naive #################### 
# print("quantizing model")
# model_w8a8 = quantize_opt(model_fp16, weight_quant=W_QUANT, act_quant=A_QUANT, w_bits=W_BITS, a_bits=A_BITS) 

# print("running eval of quantized model")
# w8a8_acc = evaluator_last_token.evaluate(model_w8a8) 
# w8a8_prep = evaluator_perplexity.evaluate(model_w8a8) 

#################### smoothing ####################
# print("loading model for smoothing")
# model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto") 
# model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=None).to("cuda") 

# print("loading activation scales")
# # act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-125m.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-350m.pt")
# # act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-1b.pt")
# # act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-2b.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-7b.pt")
# # print("loading activation masks")
# # act_masks = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/masked_opt-125m.pt")

# smooth_lm(model_fp16, act_scales, 0.5)

# print("quantizing the smoothed model")
# model_smoothquant_w8a8 = quantize_opt(model_fp16, weight_quant=W_QUANT, act_quant=A_QUANT, w_bits=W_BITS, a_bits=A_BITS)

# print("running eval of smoothed quantized model")
# smooth_w8a8_acc = evaluator_last_token.evaluate(model_smoothquant_w8a8) 
# smooth_w8a8_prep = evaluator_perplexity.evaluate(model_smoothquant_w8a8) 

#################### masked smoothing ####################
print("loading model for masked smoothing")
# model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto") 
model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=None).to("cuda")  

print("loading activation scales")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-125m.pt")
act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-7b.pt")
# # print("loading activation masks")
# # act_masks = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/masked_opt-125m.pt")
# act_masks = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/masked_opt-7b.pt")

smooth_lm(model_fp16, act_scales, 0.5, True, 1, 0.015)

print("quantizing the masked smoothed model")
model_masked_smoothquant_w8a8 = quantize_opt(model_fp16, weight_quant=W_QUANT, act_quant=A_QUANT, w_bits=W_BITS, a_bits=A_BITS)

print("running eval of smoothed quantized model")
masked_smooth_w8a8_acc = evaluator_last_token.evaluate(model_masked_smoothquant_w8a8) 
masked_smooth_w8a8_prep = evaluator_perplexity.evaluate(model_masked_smoothquant_w8a8) 
#######
# perc = [] 
# acc = [] 
# per = [] 

# for ii in np.arange(0.5, 0.6, 0.05): 
# for ii in np.arange(0.005, 0.03, 0.005): # A
# for ii in np.arange(0.03, 0.055, 0.005): # B
# for ii in np.arange(0.055, 0.08, 0.005): # C
# for ii in np.arange(0.08, 0.11, 0.005): # D
# for ii in np.arange(0.005, 0.1, 0.005): 
    # threshold_val = round(ii, 2)
    # perc_val = round(ii, 3)

    # print(f"loading model for masked smoothing for perc: {perc_val}")
    # model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto") 

    # print("loading activation scales")
    # act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-125m.pt")
    # act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-350m.pt")
    # act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-1b.pt")
    # act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-2b.pt")
    # act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-7b.pt")
    # print("loading activation masks")
    # act_masks = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/masked_opt-125m.pt")

#     smooth_lm(model_fp16, act_scales, 0.5, True, 1, perc_val)

#     print("quantizing the masked smoothed model")
#     model_masked_smoothquant_w8a8 = quantize_opt(model_fp16, weight_quant=W_QUANT, act_quant=A_QUANT, w_bits=W_BITS, a_bits=A_BITS)

#     print("running eval of smoothed quantized model")
#     masked_smooth_w8a8_acc = evaluator_last_token.evaluate(model_masked_smoothquant_w8a8) 
#     masked_smooth_w8a8_prep = evaluator_perplexity.evaluate(model_masked_smoothquant_w8a8) 

#     perc.append(perc_val)
#     acc.append(masked_smooth_w8a8_acc)
#     per.append(masked_smooth_w8a8_prep)

#     print(f"perc: {perc_val}")
#     print(f"masked_smooth_w8a8_acc: {masked_smooth_w8a8_acc}")
#     print(f"masked_smooth_w8a8_prep: {masked_smooth_w8a8_prep}")

# df = pd.DataFrame() 
# df['perc'] = perc
# df['acc'] = acc
# df['per'] = np.array([p.item() for p in per], dtype=float)

# df.to_csv('res_opt_350m_perc_med_5_.csv')

'''
('res_opt_125_perc_med_4.cv') -> np.arange(0.01, 0.11, 0.01) (0.9 0.9)

('res_opt_125_perc_med_5.cv') -> np.arange(0.005, 0.1, 0.005)
lower_fence_2 = (median_val) - (perc *median_val)
upper_fence_2 = (median_val) + (perc *median_val)
'''
###################################################

# print(f"original model (fp16) accuracy: {fp16_acc}")
# print(f"original model (fp16) perplexity: {fp16_prep}")

# print(f"naive quantized model (fp16) accuracy: {w8a8_acc}")
# print(f"naive quantized model (fp16) perplexity: {w8a8_prep}")

# print(f"smooth quantized model (fp16) accuracy: {smooth_w8a8_acc}")
# print(f"smooth quantized model (fp16) perplexity: {smooth_w8a8_prep}")

print(f"masked smooth quantized model (fp16) accuracy: {masked_smooth_w8a8_acc}")
print(f"masked smooth quantized model (fp16) perplexity: {masked_smooth_w8a8_prep}")

