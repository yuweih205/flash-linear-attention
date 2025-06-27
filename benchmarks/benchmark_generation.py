# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024, Songlin Yang, Yu Zhang.

import argparse
import statistics
import time
import warnings

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import fla  # noqa: import only for dependency check

_UNITS = ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei")


def human_bytes(num: int, suffix: str = "B") -> str:
    for unit in _UNITS:
        if abs(num) < 1024:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def print_stats(values, label):
    if not values:
        return 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    print(
        f"{label:<7}  avg {mean*1e3:7.1f} ms | "
        f"min {min(values)*1e3:7.1f} | "
        f"max {max(values)*1e3:7.1f} | "
        f"std {std*1e3:5.1f}"
    )
    return mean


def main():
    parser = argparse.ArgumentParser("Generation benchmark (prefill / decode)")
    parser.add_argument("--path", type=str, default="fla-hub/transformer-1.3B-100B")
    parser.add_argument("--data", type=str, default="fla-hub/pg19")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--topp", type=float, default=0.2)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--num-runs", type=int, default=6, help="包含 1 次 warm-up")
    parser.add_argument("--no-sample", action="store_true", help="关闭采样，测确定性路径")
    args = parser.parse_args()

    # Torch全局
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    warnings.filterwarnings("ignore")

    # 加载tokenizer/model
    print(f"[Load] {args.path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.path, trust_remote_code=True, add_eos_token=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        device_map={"": device},
        torch_dtype=dtype,
        use_cache=not args.no_cache,
    )
    if args.compile:
        print("[torch.compile] optimizing…")
        model = torch.compile(model, dynamic=True, fullgraph=False)
    model.eval()

    # 构造 Prompt
    dataset = load_dataset(args.data, split="train", trust_remote_code=True)
    prompt_text = dataset[0][list(dataset[0].keys())[0]]
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[:, :args.length].to(device)
    max_new_tokens = args.maxlen

    # 生成配置
    gen_cfg = dict(
        do_sample=not args.no_sample,
        temperature=args.temperature,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    print(model.config)      # 打印 TransformerConfig
    print(model)    

    prefill_times, decode_times = [], []

    for run in range(args.num_runs):
        torch.cuda.reset_peak_memory_stats(device)

        # Prefill+首token
        torch.cuda.synchronize()
        t0 = time.time()
        outputs = model(input_ids, use_cache=True)
        pkv = outputs.past_key_values
        first_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()
        t1 = time.time()

        # Decode剩余tokens
        torch.cuda.synchronize()
        t2 = time.time()
        model.generate(
            first_token,
            past_key_values=pkv,
            max_new_tokens=max_new_tokens - 1,
            **gen_cfg,
        )
        torch.cuda.synchronize()
        t3 = time.time()

        pre_ms = (t1 - t0) * 1e3
        dec_ms = (t3 - t2) * 1e3

        print(
            f"Run {run:<2}  prefill {pre_ms:7.1f} ms | "
            f"decode {dec_ms:7.1f} ms | "
            f"peak {torch.cuda.max_memory_allocated()/1e6:,.0f} MB"
        )
        if run:                       # 0 = warm-up
            prefill_times.append(t1 - t0)
            decode_times.append(t3 - t2)

    # 汇总
    print("\n===== SUMMARY (exclude warm-up) =====")
    avg_pf = print_stats(prefill_times, "Prefill")
    avg_dc = print_stats(decode_times, "Decode")

    if avg_pf and avg_dc:
        tp_pf = args.length / avg_pf
        tp_dc = args.maxlen / avg_dc
        print(
            f"\nToken/s  prefill {tp_pf:7.1f} | "
            f"decode {tp_dc:7.1f} | "
            f"ratio {tp_pf/tp_dc:4.2f}"
        )

    print(f"Peak CUDA memory: {human_bytes(torch.cuda.max_memory_allocated())}")


if __name__ == "__main__":
    main()
