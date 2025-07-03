#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import warnings
from typing import Dict, List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. CSV output will be limited.")

import numpy as np
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.testing
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: triton not available.")

try:
    from fla.utils import device
    HAS_FLA = True
except ImportError as e:
    print(f"Error importing FLA: {e}")
    HAS_FLA = False

warnings.filterwarnings("ignore")

# Configuration
dtype = torch.bfloat16
REPEATS = 10
QUANTILES = [0.5, 0.2, 0.8]
SEQ_LENGTHS = [128 * 2 ** i for i in range(8)]
BATCH_SIZE = 4
NUM_HEADS = 8
HEAD_DIM = 128

# 在此处新增算子名 需和真实调用算子名一致 并在后续create_tensors_for_operator匹配输入 
STABLE_OPERATORS = [
    # ABC Family
    'chunk_abc',
    
    # Attention Family
    'parallel_attn',
    
    # Based Family
    'fused_chunk_based',
    'parallel_based',
    
    # Comba Family
    'chunk_comba',
    
    # Delta Rule Family
    'chunk_delta_rule',
#    'fused_chunk_delta_rule',
    'fused_recurrent_delta_rule',
    
    # Gated Delta Rule Family
    'chunk_gated_delta_rule',
#    'fused_recurrent_gated_delta_rule',
    
    # GLA Family
    'chunk_gla',
    'fused_chunk_gla',
#    'fused_recurrent_gla',
    
    # GSA Family
    'chunk_gsa',
#    'fused_recurrent_gsa',
    
    # HGRN Family
    'fused_recurrent_hgrn',
    
    # Lightning Attention Family
    'chunk_lightning_attn',
    
    # Linear Attention Family
    'chunk_linear_attn',
    'fused_chunk_linear_attn',
    'fused_recurrent_linear_attn',
    
    # NSA Family
    'parallel_nsa',
    
    # Retention Family
    'chunk_retention',
    'fused_chunk_retention',
    'fused_recurrent_retention',
    'parallel_retention',
    
    # RWKV6 Family
    'chunk_rwkv6',
    'fused_recurrent_rwkv6',
    
    # RWKV7 Family
    'chunk_rwkv7',
    
    # Simple GLA Family
    'chunk_simple_gla',
]

def create_tensors_for_operator(operator_name: str, seq_len: int) -> Tuple[torch.Tensor, ...]:
    """为指定算子创建正确的张量"""
    q = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
    
    if operator_name == 'chunk_gla' or operator_name.startswith('fused_chunk_gla') or operator_name.startswith('fused_recurrent_gla'):
        g = F.logsigmoid(torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)).clamp_min(-5).requires_grad_(True)
        return (q, k, v, g)
    
    elif operator_name == 'chunk_rwkv6' or operator_name.startswith('fused_recurrent_rwkv6'):
        w = F.logsigmoid(torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)).clamp_min(-5).requires_grad_(True)
        u = torch.randn(NUM_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        return (q, k, v, w, u)
    
    elif operator_name == 'chunk_rwkv7':
        w = F.logsigmoid(torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)).clamp_min(-5).requires_grad_(True)
        a = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        return (q, w, k, v, a, b)
    
    elif operator_name == 'chunk_abc':
        s = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        return (q, k, v, s)
    
    elif operator_name == 'chunk_comba':
        p = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        g = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, device=device, dtype=dtype, requires_grad=True)
        beta = torch.rand(BATCH_SIZE, seq_len, NUM_HEADS, device=device, dtype=dtype).sigmoid().requires_grad_(True)
        return (q, k, v, p, g, beta)
    
    elif operator_name == 'chunk_delta_rule' or operator_name.startswith('fused_chunk_delta_rule') or operator_name.startswith('fused_recurrent_delta_rule'):
        beta = torch.rand(BATCH_SIZE, seq_len, NUM_HEADS, device=device, dtype=dtype).sigmoid().requires_grad_(True)
        return (q, k, v, beta)
    
    elif operator_name == 'chunk_gated_delta_rule' or operator_name.startswith('fused_recurrent_gated_delta_rule'):
        gk = F.logsigmoid(torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)).clamp_min(-5).requires_grad_(True)
        beta = torch.rand(BATCH_SIZE, seq_len, NUM_HEADS, device=device, dtype=dtype).sigmoid().requires_grad_(True)
        return (q, k, v, gk, beta)
    
    elif operator_name == 'chunk_simple_gla':
        gk = F.logsigmoid(torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)).clamp_min(-5).requires_grad_(True)
        return (q, k, v, gk)
    
    elif operator_name == 'chunk_gsa' or operator_name.startswith('fused_recurrent_gsa'):
        M = 64
        f = F.logsigmoid(torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, M, device=device, dtype=dtype))
        s = (1 - f.exp()).to(f.dtype)
        return (q, k, v, s, f)
    
    # elif operator_name.startswith('fused_recurrent_gsa'):
    #     s = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, device=device, dtype=dtype, requires_grad=True)
    #     f = F.logsigmoid(torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, M, device=device, dtype=dtype))
    #     f = F.logsigmoid(torch.randn(B, T, H, M, device=device, dtype=dtype))
    #     s = (1 - f.exp()).to(f.dtype)
    #     return (q, k, v, s, f)
    
    elif operator_name == 'fused_recurrent_hgrn':
        x = torch.randn((BATCH_SIZE, seq_len, HEAD_DIM), dtype=dtype, device=device)
        g = torch.randn((BATCH_SIZE, seq_len, HEAD_DIM), dtype=dtype, device=device).sigmoid()
        x = (1 - g) * x
        x, g = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g))
        g = F.logsigmoid(torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)).clamp_min(-5).requires_grad_(True)
        return (x, g)
    
    elif operator_name == 'chunk_lightning_attn':
        return (q, k, v, 0, 12)  # layer_idx=0, num_layers=12
    
    elif operator_name == 'parallel_nsa':
        q = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS * 16, HEAD_DIM, device=device, requires_grad=True, dtype=dtype)
        S = 16        
        block_size = 64
        block_counts = 16
        indices = torch.full((BATCH_SIZE, seq_len, NUM_HEADS, S), seq_len, dtype=torch.long, device=device)
        for b in range(BATCH_SIZE):
            for t in range(seq_len):
                for h in range(NUM_HEADS):
                    i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
                    indices[b, t, h, :len(i_i)] = i_i
        indices = indices.sort(-1)[0]
        return (q, k, v, None, None, None, indices, block_counts, block_size)
    
    elif operator_name == 'fused_chunk_based' or operator_name.startswith('parallel_based'):
        # fused_chunk_based 需要特殊的张量布局和维度
        if operator_name == 'fused_chunk_based':
            # fused_chunk_based 需要 K=16 且使用 head_first=True
            q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, 16, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, 16, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        else:
            # parallel_based 使用标准布局
            q = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        return (q, k, v)
    
    else:
        # 默认情况，只返回q, k, v
        return (q, k, v)

def benchmark_operator(operator_name: str, seq_len: int) -> Tuple[float, float]:
    """对单个算子进行benchmark"""
    try:
        # 直接从 fla.ops 导入所有算子
        import importlib
        operator_module = importlib.import_module('fla.ops')
        operator_func = getattr(operator_module, operator_name)
        
        # 创建张量
        tensors = create_tensors_for_operator(operator_name, seq_len)
        
        # 前向传播benchmark
        def forward():
            return operator_func(*tensors)
        
        fwd_time = triton.testing.do_bench(forward, quantiles=QUANTILES)[0]
        
        # 反向传播benchmark
        try:
            output = forward()
            if isinstance(output, tuple):
                output = output[0]
            grad = torch.randn_like(output)
        
            def backward():
                for t in tensors:
                    if isinstance(t, torch.Tensor) and t.grad is not None:
                        t.grad.zero_()
                out = forward()
                if isinstance(out, tuple):
                    out = out[0]
                out.backward(grad)
        
            bwd_time = triton.testing.do_bench(backward, quantiles=QUANTILES)[0]      
        except Exception:
            bwd_time = None
            print(f"  {operator_name} seems not support backward, skipping backward benchmark.")
        
        return fwd_time, bwd_time
        
    except Exception as e:
        print(f"    Benchmark failed: {e}")
        return float('inf'), float('inf')

def run_benchmarks(operators: Optional[List[str]] = None) -> Dict:
    """运行benchmark"""
    if not HAS_FLA:
        raise RuntimeError("FLA not available")
    
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")
    
    # 确定要测试的算子
    if operators is None:
        operators = STABLE_OPERATORS
    else:
        # 过滤出稳定的算子
        available_ops = set(STABLE_OPERATORS)
        operators = [op for op in operators if op in available_ops]
        if not operators:
            raise ValueError(f"No available operators found. Available: {list(available_ops)}")
    
    print(f"Running benchmarks for {len(operators)} operators...")
    print(f"Sequence lengths: {SEQ_LENGTHS}")
    print(f"Data type: {dtype}")
    print("=" * 80)
    
    results = {}
    
    for op_name in operators:
        print(f"\nBenchmarking {op_name}...")
        
        op_results = {'forward': {}, 'backward': {}}
        
        for seq_len in SEQ_LENGTHS:
            try:
                print(f"  Testing seq_len={seq_len}...", end=" ")
                
                # 运行benchmark
                fwd_time, bwd_time = benchmark_operator(op_name, seq_len)
                
                op_results['forward'][seq_len] = fwd_time
                op_results['backward'][seq_len] = bwd_time
                
                if fwd_time != float('inf') and bwd_time != float('inf'):
                    print(f"fwd={fwd_time:.2f}ms, bwd={bwd_time:.2f}ms")
                else:
                    print("FAILED")
                    
            except Exception as e:
                print(f"FAILED: {e}")
                op_results['forward'][seq_len] = float('inf')
                op_results['backward'][seq_len] = float('inf')
        
        results[op_name] = op_results
    
    return results

def save_results(results: Dict, output_dir: str, export_formats: List[str]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    excel_path = os.path.join(output_dir, "benchmark_results.xlsx")

    seq_lens = sorted(SEQ_LENGTHS)
    operators = list(results.keys())
    fieldnames = ["seq_len"] + [f"{op}_{phase}" for op in operators for phase in ("fwd", "bwd", "total")]

    rows = []
    for seq_len in seq_lens:
        row = {"seq_len": seq_len}
        for op in operators:
            fwd = results[op]["forward"].get(seq_len, float("inf"))
            bwd = results[op]["backward"].get(seq_len, float("inf"))
            row[f"{op}_fwd"] = None if fwd == float("inf") else fwd
            row[f"{op}_bwd"] = None if bwd == float("inf") else bwd
            row[f"{op}_total"] = None if (fwd == float("inf") or bwd == float("inf")) else fwd + bwd
        rows.append(row)

    # 写入 CSV（可选）
    if "csv" in export_formats:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV to {csv_path}")
    else:
        csv_path = None  # 不导出就返回 None

    # 写入 Excel（可选）
    if HAS_PANDAS and "xlsx" in export_formats:
        df = pd.DataFrame(rows)
        try:
            df.to_excel(excel_path, index=False)
            print(f"Saved Excel to {excel_path}")
        except ImportError:
            print("Warning: can't export Excel—need openpyxl or xlsxwriter.")

    # 控制台打印（可选）
    if "console" in export_formats:
        print("\nConsole Output (15 columns per page):")
        if HAS_PANDAS:
            df = pd.DataFrame(rows)
            cols = df.columns.tolist()
        else:
            df = rows
            cols = fieldnames

        base_col = ["seq_len"]
        op_cols = [col for col in cols if col != "seq_len"]
        for i in range(0, len(op_cols), 15):
            sub_cols = base_col + op_cols[i:i + 15]
            if HAS_PANDAS:
                print(df[sub_cols].to_string(index=False))
            else:
                print("\t".join(sub_cols))
                for row in rows:
                    print("\t".join(str(row.get(col, "")) for col in sub_cols))
            print("\n" + "-" * 100)

    # 绘图（可选）
    if HAS_PLOTTING:
        create_plots(results, output_dir)

    return csv_path or ""

def create_plots(results: Dict, output_dir: str):
    """创建性能图表"""
    try:
        seq_lens = sorted(SEQ_LENGTHS)
        operators = list(results.keys())
        
        # 设置样式
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Operator Performance Benchmark (Fixed)', fontsize=16)
        
        # 前向传播时间
        ax1 = axes[0, 0]
        for op_name in operators:
            op_results = results[op_name]
            fwd_times = [op_results['forward'].get(seq_len, float('inf')) for seq_len in seq_lens]
            valid_times = [(seq_len, time) for seq_len, time in zip(seq_lens, fwd_times) if time != float('inf')]
            if valid_times:
                seq_lens_valid, times_valid = zip(*valid_times)
                ax1.plot(seq_lens_valid, times_valid, marker='o', label=op_name)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Forward Time (ms)')
        ax1.set_title('Forward Pass Performance')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # 反向传播时间
        ax2 = axes[0, 1]
        for op_name in operators:
            op_results = results[op_name]
            bwd_times = [op_results['backward'].get(seq_len, float('inf')) for seq_len in seq_lens]
            valid_times = [(seq_len, time) for seq_len, time in zip(seq_lens, bwd_times) if time != float('inf')]
            if valid_times:
                seq_lens_valid, times_valid = zip(*valid_times)
                ax2.plot(seq_lens_valid, times_valid, marker='s', label=op_name)
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Backward Time (ms)')
        ax2.set_title('Backward Pass Performance')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # 总时间
        ax3 = axes[1, 0]
        for op_name in operators:
            op_results = results[op_name]
            total_times = []
            for seq_len in seq_lens:
                fwd_time = op_results['forward'].get(seq_len, float('inf'))
                bwd_time = op_results['backward'].get(seq_len, float('inf'))
                total_time = fwd_time + bwd_time if fwd_time != float('inf') and bwd_time != float('inf') else float('inf')
                total_times.append(total_time)
            
            valid_times = [(seq_len, time) for seq_len, time in zip(seq_lens, total_times) if time != float('inf')]
            if valid_times:
                seq_lens_valid, times_valid = zip(*valid_times)
                ax3.plot(seq_lens_valid, times_valid, marker='^', label=op_name)
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Total Time (ms)')
        ax3.set_title('Total Performance')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # 性能对比
        ax4 = axes[1, 1]
        # 选择几个主要算子进行对比
        main_ops = ['chunk_gla', 'chunk_retention', 'chunk_linear_attn', 'chunk_rwkv6', 'chunk_rwkv7']
        for op_name in main_ops:
            if op_name in results:
                op_results = results[op_name]
                total_times = []
                for seq_len in seq_lens:
                    fwd_time = op_results['forward'].get(seq_len, float('inf'))
                    bwd_time = op_results['backward'].get(seq_len, float('inf'))
                    total_time = fwd_time + bwd_time if fwd_time != float('inf') and bwd_time != float('inf') else float('inf')
                    total_times.append(total_time)
                
                valid_times = [(seq_len, time) for seq_len, time in zip(seq_lens, total_times) if time != float('inf')]
                if valid_times:
                    seq_lens_valid, times_valid = zip(*valid_times)
                    ax4.plot(seq_lens_valid, times_valid, marker='o', label=op_name)
        
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Total Time (ms)')
        ax4.set_title('Main Operators Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'benchmark_plots_fixed.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plots saved to: {plot_path}")
        
    except Exception as e:
        print(f"Failed to create plots: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed Operator Benchmark')
    parser.add_argument('--operators',"-op", nargs='+', help='Specific operators to benchmark')
    parser.add_argument('--seq-lens', "-T", nargs='+', type=int, help='Specific sequence lengths to test')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory for results')
    parser.add_argument(
        "--export-formats", "-e",
        nargs="+",
        choices=["csv","xlsx","console"],
        default=["csv","console","xlsx"],
        help="Which outputs to generate: csv, xlsx, console"
    )    
    
    args = parser.parse_args()
    
    # 更新序列长度
    if args.seq_lens:
        global SEQ_LENGTHS
        SEQ_LENGTHS = args.seq_lens
    
    try:
        # 运行benchmark
        results = run_benchmarks(args.operators)
        
        # 保存结果
        save_results(results, output_dir=args.output_dir, export_formats=args.export_formats,)
        
        print(f"\nBenchmark completed successfully!")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())