import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.multiprocessing as mp
import gc
import time
import sys

# Import Llama adaptive hijack from AdaKV
# Adjust sys.path to find adaptive_snapkv if running from subdirectory
sys.path.append("../../") 
from adaptive_snapkv.monkeypatch.monkeypatch import replace_llama_adaptive

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model_name_or_path', type=str, required=True)
    parser.add_argument('--max_length', type=int, required=True)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("-d", '--dataset', type=str, default="THUDM/LongBench")
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['ada', 'base'], help="Ada mode or base mode")
    parser.add_argument('--floor_alpha',type=float,default=0.2,help="floor_alpha budgets for each head")
    parser.add_argument('--pyram',action='store_true',help="using pyram mode")
    parser.add_argument('--pyram_beta',default=20,type=int, help="hyper parameter for pyram")
    parser.add_argument('--budget',default=1024, type=int, help="budget size for kv cache")
    parser.add_argument('--window_size', default=32, type=int, help="window size")
    parser.add_argument('--tasks', type=str, default=None, help="Comma-separated list of tasks to evaluate")
    return parser.parse_args(args)

def build_chat(tokenizer, prompt, model_name):
    if "llama-3" in model_name.lower():
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ], tokenize=False, add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    if "llama-3" in model_name.lower():
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .strip()
        )
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name_or_path, out_path):
    device = "cuda"
    preds = []
    times = []
    
    # Enable performance tracking
    peak_mems = []

    with open(f"{out_path}_tmp", "w", encoding="utf-8") as f:
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt, model_name_or_path)
            
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]

            # Clear stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_time = time.time()

            generate_output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            )
            
            torch.cuda.synchronize()
            end_time = time.time()
            current_peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            
            t = end_time - start_time
            times.append(t)
            peak_mems.append(current_peak_mem)
            
            output = generate_output[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = post_process(pred, model_name_or_path)
            preds.append(pred)

            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"], "time": t, "peak_mem": current_peak_mem}, f, ensure_ascii=False )
            f.write('\n')
            f.flush()

            gc.collect()
            torch.cuda.empty_cache()

    with open(out_path, "w", encoding="utf-8") as f:
        for json_obj, pred, t, mem in zip(data, preds, times, peak_mems):
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"], "time": t, "peak_mem": mem}, f, ensure_ascii=False )
            f.write('\n')
            
    print(f"Average Time: {np.mean(times):.2f}s, Max Memory: {np.max(peak_mems):.2f}GB")

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
                                             trust_remote_code=True,
                                             )
    model = model.eval()
    return model, tokenizer

# Configuration helper for AdaKV
def config_compress(model, window_size=32, base_capacity=512, kernel_size=7, pooling="maxpool", floor_alpha=0.2, pyram_mode=False, beta=20):
    model.model.config.window_size = window_size
    model.model.config.base_capacity = base_capacity
    model.model.config.kernel_size = kernel_size
    model.model.config.pooling = pooling
    model.model.config.floor_alpha = floor_alpha
    model.model.config.pyram_mode = pyram_mode
    model.model.config.pyram_beta = beta
    # Standard settings for AdaKV Llama
    model.model.config.skip = 0
    model.model.config.normalize = False
    model.model.config.num_hidden_layers = model.config.num_hidden_layers
    model.model.config.gqa_support = False 
    model.model.config.gqa_func = "mean"
    return model

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    mp.set_start_method('spawn', force=True)

    model_name_or_path = args.model_name_or_path
    
    # Dataset selection
    if args.tasks:
        datasets = args.tasks.split(",")
    else:
        datasets = ["narrativeqa"] # Default to narrativeqa

    print(f"Running datasets: {datasets}")
    
    # Load configs
    config_path = "config"
    print(f"Loading config from {config_path}")

    dataset2prompt = json.load(open(f"{config_path}/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(f"{config_path}/dataset2maxlen.json", "r"))

    # Prepare output directory
    if not os.path.exists(f"pred/{args.out_name}"):
        os.makedirs(f"pred/{args.out_name}", exist_ok=True)

    # 1. Apply Hijack (Monkey Patch) BEFORE loading model
    if args.mode == "ada":
        print("🔧 Ada mode: Applying Adaptive SnapKV for Llama3")
        replace_llama_adaptive()

    # 2. Load Model
    print(f"Loading model from: {model_name_or_path}")
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    # 3. Config Compress Args
    if args.mode == "ada":
        config_compress(
            model,
            window_size=args.window_size,
            base_capacity=args.budget,
            floor_alpha=args.floor_alpha,
            pyram_mode=args.pyram,
            beta=args.pyram_beta
        )

    for dataset in datasets:
        # Load Dataset
        print(f"Loading dataset: {dataset} from {args.dataset}")
        
        # Support loading from local folder structure: /users/cyx/LongBench/data/narrativeqa.jsonl
        local_file_path = os.path.join(args.dataset, "data", f"{dataset}.jsonl")
        
        if os.path.exists(local_file_path):
            print(f"Found local file: {local_file_path}")
            data = load_dataset("json", data_files={"test": local_file_path}, split='test')
        elif os.path.exists(os.path.join(args.dataset, dataset)): # Try direct folder
             data = load_dataset(args.dataset, dataset, split='test')
        else:
             print(f"Warning: Local file not found at {local_file_path}, trying huggingface load...")
             try:
                data = load_dataset(args.dataset, dataset, split='test')
             except Exception as e:
                 print(f"Error loading dataset: {e}")
                 continue

        out_path = f"pred/{args.out_name}/{dataset}.jsonl"
        
        if os.path.exists(out_path):
            print(f"Skipping {dataset}, output exists.")
            continue

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        
        # Run prediction
        get_pred(model, tokenizer, data, args.max_length, max_gen, prompt_format, dataset, "cuda", model_name_or_path, out_path)
