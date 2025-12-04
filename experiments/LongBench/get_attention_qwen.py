import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import gc
import time
import math
import torch.nn.functional as F
import torch.nn as nn

from adaptive_snapkv.monkeypatch.monkeypatch import replace_mistral_adaptive, replace_llama_adaptive, replace_mistral_fixed, replace_llama_fixed, replace_mistral_slm, replace_llama_slm
from adaptive_snapkv.qwen3_hack.patch import replace_qwen3_adaptive

# Global storage for attention scores
GLOBAL_ATTN_SCORES = {}

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def manual_calcul_attn_score(key_states, query_states, window_size, kernel_size, pooling, gqa_support=False, num_key_value_groups=1, gqa_func="mean"):
    """
    计算最近 window_size 个 Query Token 对所有 Key Token 的注意力分数。
    Ref: adaptive_snapkv/monkeypatch/snapkv_utils.py
    """
    # Handle GQA: Repeat KV heads to match Query heads
    # Mistral-7B uses GQA (32 Q heads, 8 KV heads). We MUST repeat KV heads to 32 to perform matmul.
    # This check should be independent of the gqa_support flag if the dimensions don't match.
    if query_states.shape[1] != key_states.shape[1]:
        # Calculate n_rep dynamically if needed, or use provided group size
        n_rep = query_states.shape[1] // key_states.shape[1]
        if n_rep != num_key_value_groups:
             # Fallback or warning if group size doesn't match calculated ratio
             # But usually trusting the ratio is safer for pure matrix math
             pass
        key_states = repeat_kv(key_states, n_rep)
        # We typically don't use value_states here, but if we did, it would need repeat too

    bsz, num_heads, q_len, head_dim = query_states.shape
    
    # 确保只计算最后 window_size 个 query 的注意力
    valid_window = min(window_size, q_len)
    
    # [bsz, num_heads, valid_window, head_dim] x [bsz, num_heads, head_dim, k_len] -> [bsz, num_heads, valid_window, k_len]
    attn_weights = torch.matmul(query_states[..., -valid_window:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
    
    # Masking (Causal Mask for the diagonal part within the window)
    mask = torch.full((valid_window, valid_window), torch.finfo(attn_weights.dtype).min,
                        device=attn_weights.device)
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(attn_weights.device)
    attention_mask = mask[None, None, :, :]

    k_len = key_states.shape[-2]
    # Apply mask to the local window attention
    if k_len >= valid_window:
        attn_weights[:, :, -valid_window:, -valid_window:] += attention_mask
    
    # Softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Mean aggregation over Query dimension (excluding the local window from the "history" perspective if needed)
    # Following snapkv_utils logic:
    if k_len > valid_window:
        # 原始逻辑：只看历史部分的注意力分布
        attn_weights_mean = attn_weights[:, :, -valid_window:, : -valid_window].mean(dim=-2)
    else:
        # 序列太短，直接平均
        attn_weights_mean = attn_weights.mean(dim=-2)

    # # GQA handling
    # if gqa_support:
    #     attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0], num_heads//num_key_value_groups, num_key_value_groups, -1)
    #     if gqa_func == 'max':
    #         attn_weights_mean = attn_weights_mean.max(dim=-2).values
    #     elif gqa_func == 'mean':
    #         attn_weights_mean = attn_weights_mean.mean(dim=-2)
    #     else:
    #         raise ValueError('gqa_func not supported')

    # # Pooling
    # if pooling == 'avgpool':
    #     attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=kernel_size,
    #                                                 padding=kernel_size // 2,
    #                                                 stride=1)
    # elif pooling == 'maxpool':
    #     attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=kernel_size,
    #                                                 padding=kernel_size // 2,
    #                                                 stride=1)
    # else:
    #     raise ValueError('Pooling method not supported')
        
    return attn_weights_mean

class AttnScoreCapturer:
    """
    Injects into the model to capture attention scores during forward pass without compressing KV cache.
    """
    def __init__(self, layer_idx, config):
        self.layer_idx = layer_idx
        self.config = config
        self.window_size = getattr(config, 'window_size', 32)
        self.kernel_size = getattr(config, 'kernel_size', 7)
        self.pooling = getattr(config, 'pooling', 'maxpool')
        self.gqa_support = getattr(config, 'gqa_support', False)
        self.gqa_func = getattr(config, 'gqa_func', 'mean')
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        
        # Mock attributes to satisfy potential checks in monkeypatch code
        # Use INT32_MAX to ensure it covers any possible sequence length
        self.base_capacity = 2147483647 
        self.max_capacity_prompt = self.base_capacity
        self.pyram_mode = getattr(config, 'pyram_mode', False)

    def update_kv(self, key_states, query_states, value_states):
        # Debug: check if called
        # print(f"DEBUG: update_kv called. q_len={query_states.shape[-2]}, k_len={key_states.shape[-2]}")
        
        # Capture scores only during Prefill (q_len > 1)
        if query_states.shape[-2] > 1:
            # Only calculate if we have enough history
            if key_states.shape[-2] > self.window_size:
                score = manual_calcul_attn_score(
                    key_states, query_states, 
                    self.window_size, self.kernel_size, self.pooling,
                    self.gqa_support, self.num_key_value_groups, self.gqa_func
                )
                
                if self.layer_idx not in GLOBAL_ATTN_SCORES:
                    GLOBAL_ATTN_SCORES[self.layer_idx] = []
                # Save detached score to global storage
                GLOBAL_ATTN_SCORES[self.layer_idx].append(score.detach().cpu())
            else:
                # Debug info
                print(f"DEBUG: k_len ({key_states.shape[-2]}) <= window_size ({self.window_size}), skipping score calc")
                pass
        
        # Always return original states (No Compression)
        return key_states, value_states
        
    def reset(self, *args, **kwargs):
        pass

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model_name_or_path', type=str, required=True)
    parser.add_argument('--max_length', type=int, required=True)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("-d", '--dataset', type=str, default="THUDM/LongBench")
    parser.add_argument("--out_name", type=str, required=True)
    
    # Compress args (used for calculating scores logic)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--pooling', type=str, default='maxpool')
    
    parser.add_argument('--gqa_support', action='store_true', default=False, help="init gqa_support")
    parser.add_argument('--gqa_func', type=str, default="mean", help="gqa operation:optional max mean")
    
    # Legacy args to prevent errors if config object is shared
    parser.add_argument('--budget', default=1024, type=int, help="budget size for kv cache (not used for compression here)")
    parser.add_argument('--pyram', action='store_true', help="using pyram mode")
    parser.add_argument('--floor_alpha', type=float, default=0.2, help="floor_alpha budgets for each head")

    return parser.parse_args(args)

def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif ("llama-3" in model_name.lower() or "qwen" in model_name.lower()) and "instruct" in model_name.lower():
        # General support for models with chat templates (Llama-3, Qwen2/3-Instruct)
        prompt =  [{ "role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
        )
    return prompt

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",
                                             attn_implementation="flash_attention_2",
                                             trust_remote_code=True)
    model = model.eval()
    return model, tokenizer

def get_attn_scores(model, tokenizer, data, max_length, prompt_format, dataset, device, model_name_or_path, out_path_prefix):
    print(f"Processing dataset: {dataset}")
    
    # Ensure output directory exists
    save_dir = os.path.dirname(out_path_prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, json_obj in enumerate(tqdm(data)):
        # Clear previous scores
        GLOBAL_ATTN_SCORES.clear()
        
        prompt = prompt_format.format(**json_obj)
        
        # Tokenize & Truncate
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name_or_path:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: 
            prompt = build_chat(tokenizer, prompt, model_name_or_path)
            
        # To Device
        if "chatglm3" in model_name_or_path:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input_tokens = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input_tokens = prompt.to(device)
        else:
            input_tokens = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            
        # Forward Pass (Prefill Only via generate)
        # We use generate with max_new_tokens=1 to trigger the prefill phase exactly as pred.py does.
        # The monkey patch will capture the attention scores during the prefill step.
        # Subsequent decode step (1 token) will be ignored by update_kv because q_len=1.
        
        context_length = input_tokens.input_ids.shape[-1]
        
        if dataset == "samsum": 
            _ = model.generate(
                **input_tokens,
                max_new_tokens=1,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )
        else:
            _ = model.generate(
                **input_tokens,
                max_new_tokens=1,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=[tokenizer.eos_token_id],
            )
            
        # Save Scores
        sample_scores = {}
        # GLOBAL_ATTN_SCORES structure: {layer_idx: [tensor_chunk1, tensor_chunk2...]}
        for layer_idx, scores_list in GLOBAL_ATTN_SCORES.items():
            if scores_list:
                # Convert to list for JSON serialization
                # Taking the first element assuming single-chunk prefill for simplicity
                # The shape is [bsz=1, num_heads, compressed_len]
                # NOTE: Convert to float32 before numpy conversion because bfloat16 is not supported by numpy
                sample_scores[layer_idx] = scores_list[0].squeeze(0).float().cpu().numpy().tolist() 
        
        # Save to individual file per sample
        save_file = f"{out_path_prefix}_sample_{i}.json"
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump({
                "sample_idx": i,
                "length": json_obj["length"],
                "real_length": input_tokens.input_ids.shape[-1],
                "scores": sample_scores
            }, f)
            
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    mp.set_start_method('spawn', force=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_or_path = args.model_name_or_path
    
    # 1. Apply Monkey Patch to enable hook injection
    # This replaces the forward pass with one that calls our custom update_kv
    print("Applying Monkey Patch for Attention Capture...")
    if "qwen" in model_name_or_path.lower():
        replace_qwen3_adaptive()
    else:
        replace_mistral_adaptive()
        replace_llama_adaptive()
    
    # 2. Load Model
    print(f"Loading model from {model_name_or_path}...")
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)
    
    # 3. Inject Config into Model (needed for AttnScoreCapturer init)
    model.config.window_size = args.window_size
    model.config.kernel_size = args.kernel_size
    model.config.pooling = args.pooling
    model.config.gqa_support = args.gqa_support
    model.config.gqa_func = args.gqa_func
    # Dummy values
    model.config.base_capacity = args.budget 
    model.config.pyram_mode = args.pyram
    model.config.floor_alpha = args.floor_alpha
    model.config.skip = 0
    model.config.normalize = None
    model.config.pyram_beta = 20

    # 4. Inject AttnScoreCapturer into each layer
    print("Injecting AttnScoreCapturer into model layers...")
    
    # Debug: Check Attention Type
    print(f"Attention Class Type: {type(model.model.layers[0].self_attn)}")

    for i, layer in enumerate(model.model.layers):
        # 直接注入，不检查是否存在。
        # 因为 AdaKV 通常是 Lazy Initialization (在第一次 forward 时创建)，所以现在肯定没有。
        # 我们预先注入，这样当 Patch 过的 forward 运行时，它会发现已经有 kv_cluster 了，从而使用我们的 Capturer。
        layer.self_attn.kv_cluster = AttnScoreCapturer(i, model.config)
        print(f"Layer {i}: Injected AttnScoreCapturer")

    # 5. Prepare Datasets
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    # Load Prompt Configs
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    
    # 6. Run Extraction
    if not os.path.exists("pred_attn"):
        os.makedirs("pred_attn")
        
    for dataset in datasets:
        if args.e:
            ds_name = f"{dataset}_e"
        else:
            ds_name = f"{dataset}"

        # If args.dataset is a directory, use it as the base path for data files
        if os.path.isdir(args.dataset):
            base_data_dir = os.path.join(args.dataset, "data") # Assuming structure is args.dataset/data/*.jsonl
            # Check if "data" subdirectory exists, if not, assume files are directly in args.dataset or adjust logic
            if not os.path.exists(base_data_dir):
                 # Fallback: maybe the jsonl files are directly in args.dataset? 
                 # Or maybe the structure is different. Based on user query, it is /root/cyx/dataset/LongBench/data/narrativeqa.jsonl
                 # so args.dataset is /root/cyx/dataset/LongBench.
                 pass 
        else:
             # Default local assumption relative to script
             base_data_dir = "data"

        json_file = os.path.join(base_data_dir, f"{ds_name}.jsonl")
        
        print(f"Loading local dataset from: {json_file}")
        data = load_dataset("json", data_files={'test': json_file}, split='test')

        out_path_prefix = f"pred_attn/{args.out_name}/{dataset}"

        # Check if already done (simple check, maybe improved later)
        if os.path.exists(f"{out_path_prefix}_sample_0.json"):
            print(f"Output for {dataset} seems to exist, skipping...")
            continue

        prompt_format = dataset2prompt[dataset]
        data_all = [data_sample for data_sample in data]
        
        get_attn_scores(model, tokenizer, data_all, args.max_length, prompt_format, dataset, device, model_name_or_path, out_path_prefix)
        
    print("Done!")

