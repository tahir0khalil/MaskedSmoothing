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
MODEL_NAME = 'facebook/opt-1.3b'
# MODEL_NAME = 'facebook/opt-2.7b'
# MODEL_NAME = 'facebook/opt-6.7b'

if '125m' in MODEL_NAME:
    act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-125m.pt")
if '1.3b' in MODEL_NAME:
    act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-1b.pt")
if '2.7b' in MODEL_NAME:
    act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-2b.pt")
if '6.7b' in MODEL_NAME:
    act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-7b.pt")




W_BITS = 6
A_BITS = 6

W_QUANT = 'per_tensor' # [per_channel, per_tensor]
A_QUANT = 'per_tensor' # [per_token, per_tensor] 


print(f"initializing tokenizer")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
print(f"initializing dataset")
dataset = load_dataset("cimec/lambada", split="test")
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# dataset = load_dataset("EleutherAI/pile", name="hacker_news", split="validation[:5000]", trust_remote_code=True)
# dataset = load_dataset("openwebtext", split="train[:10000]")
# dataset = load_dataset("stas/c4-en-10k", split="train")

evaluator_last_token = Evaluator(dataset, tokenizer, "cuda")
evaluator_perplexity = EvaluatorPerplexity(dataset, tokenizer, "cuda")
#################### FP16 ####################
print("loading model")
model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto") 
model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=None).to("cuda") 

print("running eval of FP16 model")
fp16_acc = evaluator_last_token.evaluate(model_fp16) 
fp16_prep = evaluator_perplexity.evaluate(model_fp16) 

#################### Naive #################### 
print("quantizing model")
model_w8a8 = quantize_opt(model_fp16, weight_quant=W_QUANT, act_quant=A_QUANT, w_bits=W_BITS, a_bits=A_BITS) 

print("running eval of quantized model")
w8a8_acc = evaluator_last_token.evaluate(model_w8a8) 
w8a8_prep = evaluator_perplexity.evaluate(model_w8a8) 

#################### smoothing ####################
print("loading model for smoothing")
# model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto") 
model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=None).to("cuda") 

# print("loading activation scales")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-125m.pt")

# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-350m.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-1b.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-2b.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-7b.pt")
# ===
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_1_max.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_2_mean.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_3_median.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_4_percentile.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_p990_4_percentile.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_5_top_k_mean.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_6_scale_trimmed_mean.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_7_huber_estimator.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_8_generalized_mean.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_9_logsumexp_magnitude.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_b3_9_logsumexp_magnitude.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_b15_9_logsumexp_magnitude.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_b_30_9_logsumexp_magnitude.pt")
# bb = 18
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_logsum/opt-125m_b_"+str(bb)+ ".pt")
# bb = 18.5 # [10, 10.5, 11, 11.5, 17.5, 18, 18.5]
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_logsum/1b/opt-1b_b_"+str(bb)+ ".pt")

# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_logsum/1b/opt1b_max.pt") 

# for k,v in act_scales.items(): 
#     act_scales[k] = 1.5 * v

# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_10_high_quantile_gaussian.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_90_10_high_quantile_gaussian.pt")

# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales_modified/opt-125m_12_empirical_bayes_per_channel.pt")
# print("loading activation masks")
# act_masks = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/masked_opt-125m.pt")

smooth_lm(model_fp16, act_scales, alpha=0.5)

print("quantizing the smoothed model")
model_smoothquant_w8a8 = quantize_opt(model_fp16, weight_quant=W_QUANT, act_quant=A_QUANT, w_bits=W_BITS, a_bits=A_BITS)

print("running eval of smoothed quantized model")
smooth_w8a8_acc = evaluator_last_token.evaluate(model_smoothquant_w8a8) 
smooth_w8a8_prep = evaluator_perplexity.evaluate(model_smoothquant_w8a8) 

#################### masked smoothing ####################
print("loading model for masked smoothing")
# # model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto") 
model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=None).to("cuda")  

# print("loading activation scales")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-125m.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-1b.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-2b.pt")
# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-7b.pt")

# act_scales = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/opt-7b.pt")
# # # print("loading activation masks")
# # # act_masks = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/masked_opt-125m.pt")
# # act_masks = torch.load("/home/tahir/workspace2/MaskedSmoothing/act_scales/masked_opt-7b.pt")

smooth_lm(model_fp16, act_scales, 0.5, True, 1, 0.02)

print("quantizing the masked smoothed model")
model_masked_smoothquant_w8a8 = quantize_opt(model_fp16, weight_quant=W_QUANT, act_quant=A_QUANT, w_bits=W_BITS, a_bits=A_BITS)

print("running eval of smoothed quantized model")
masked_smooth_w8a8_acc = evaluator_last_token.evaluate(model_masked_smoothquant_w8a8) 
masked_smooth_w8a8_prep = evaluator_perplexity.evaluate(model_masked_smoothquant_w8a8) 

# percs = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.085]
# ppl = [] 
# acc = [] 
# ind = [] 
# for ii in percs: 
#     print("loading model for masked smoothing")
#     model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=None).to("cuda")   
#     print("loading activation scales")
#     smooth_lm(model_fp16, act_scales, 0.5, True, 1, ii) 
#     print("quantizing the masked smoothed model")
#     model_masked_smoothquant_w8a8 = quantize_opt(model_fp16, weight_quant=W_QUANT, act_quant=A_QUANT, w_bits=W_BITS, a_bits=A_BITS)
#     print("running eval of smoothed quantized model")
#     masked_smooth_w8a8_acc = evaluator_last_token.evaluate(model_masked_smoothquant_w8a8) 
#     masked_smooth_w8a8_prep = evaluator_perplexity.evaluate(model_masked_smoothquant_w8a8)  
#     ppl.append(masked_smooth_w8a8_prep)
#     acc.append(masked_smooth_w8a8_acc) 
#     ind.append(ii)
#     df = pd.DataFrame() 
#     df['perc'] = ind
#     df['ppl'] = np.array([p.item() for p in ppl], dtype=float)
#     df['acc'] = acc 
#     df.to_csv('opt_1b_20251113' + str(ii) +'.csv')

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

print(f"original model (fp16) accuracy: {fp16_acc}")
print(f"original model (fp16) perplexity: {fp16_prep}")

print(f"naive quantized model (fp16) accuracy: {w8a8_acc}")
print(f"naive quantized model (fp16) perplexity: {w8a8_prep}")

print(f"smooth quantized model (fp16) accuracy: {smooth_w8a8_acc}")
print(f"smooth quantized model (fp16) perplexity: {smooth_w8a8_prep}")

print(f"masked smooth quantized model (fp16) accuracy: {masked_smooth_w8a8_acc}")
print(f"masked smooth quantized model (fp16) perplexity: {masked_smooth_w8a8_prep}")

