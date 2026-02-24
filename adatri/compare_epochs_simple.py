"""
比较不同epoch的adapter测试结果指标（简化版，不依赖外部模块）
"""
import json
from pathlib import Path
from typing import Dict, List
import argparse
import re
from collections import Counter


def simple_bleu(reference: str, candidate: str, n: int = 1) -> float:
    """简单的BLEU-N计算（n-gram重叠率）"""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    if n == 1:
        ref_ngrams = Counter(ref_tokens)
        cand_ngrams = Counter(cand_tokens)
        matches = sum(min(cand_ngrams[ngram], ref_ngrams[ngram]) for ngram in cand_ngrams)
        return matches / len(cand_tokens)
    else:
        # 对于n>1，计算n-gram重叠
        ref_ngrams = []
        cand_ngrams = []
        for i in range(len(ref_tokens) - n + 1):
            ref_ngrams.append(tuple(ref_tokens[i:i+n]))
        for i in range(len(cand_tokens) - n + 1):
            cand_ngrams.append(tuple(cand_tokens[i:i+n]))
        
        if len(cand_ngrams) == 0:
            return 0.0
        
        ref_ngram_count = Counter(ref_ngrams)
        cand_ngram_count = Counter(cand_ngrams)
        matches = sum(min(cand_ngram_count[ngram], ref_ngram_count[ngram]) for ngram in cand_ngram_count)
        return matches / len(cand_ngrams)


def simple_rouge_l(reference: str, candidate: str) -> float:
    """简单的ROUGE-L计算（最长公共子序列）"""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return 0.0
    
    # 计算LCS长度
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_len = dp[m][n]
    precision = lcs_len / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
    recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_simple_metrics(generated_reports: List[str], gt_reports: List[str]) -> Dict[str, float]:
    """计算简单的NLG指标"""
    if len(generated_reports) != len(gt_reports):
        return {}
    
    bleu1_scores = []
    bleu4_scores = []
    rouge_l_scores = []
    
    for gen, gt in zip(generated_reports, gt_reports):
        bleu1 = simple_bleu(gt, gen, n=1)
        bleu4 = simple_bleu(gt, gen, n=4)
        rouge_l = simple_rouge_l(gt, gen)
        
        bleu1_scores.append(bleu1)
        bleu4_scores.append(bleu4)
        rouge_l_scores.append(rouge_l)
    
    return {
        'BLEU-1': sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
        'BLEU-4': sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0.0,
        'ROUGE-L': sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
    }


def load_epoch_results_from_summary(test_output_dir: Path) -> Dict[int, Dict]:
    """从test_summary文件加载不同epoch的测试结果"""
    epoch_results = {}
    
    # 查找test_summary文件
    summary_files = list(test_output_dir.glob("test_summary_*.json"))
    
    if not summary_files:
        print("Warning: No test_summary file found")
        return epoch_results
    
    # 使用最新的summary文件
    summary_file = sorted(summary_files)[-1]
    print(f"Loading from summary file: {summary_file.name}")
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 从results字段提取数据
        if 'results' in data:
            for result in data['results']:
                epoch = result.get('epoch')
                if epoch is None:
                    continue
                
                if 'generated_report' in result and 'ground_truth_report' in result:
                    if epoch not in epoch_results:
                        epoch_results[epoch] = {
                            'generated_reports': [],
                            'gt_reports': [],
                            'file': summary_file.name
                        }
                    
                    epoch_results[epoch]['generated_reports'].append(result['generated_report'])
                    epoch_results[epoch]['gt_reports'].append(result['ground_truth_report'])
    except Exception as e:
        print(f"Warning: Failed to load {summary_file}: {e}")
    
    return epoch_results


def compare_epochs(test_output_dir: str, epochs: List[int] = [2, 4, 6, 8, 10]):
    """比较不同epoch的指标"""
    test_output_dir = Path(test_output_dir)
    
    if not test_output_dir.exists():
        raise ValueError(f"Test output directory does not exist: {test_output_dir}")
    
    print(f"Loading results from {test_output_dir}...")
    epoch_results = load_epoch_results_from_summary(test_output_dir)
    
    # 过滤出指定的epochs
    epoch_results = {e: epoch_results[e] for e in epochs if e in epoch_results}
    
    if len(epoch_results) == 0:
        print("No valid epoch results found!")
        return
    
    print(f"Found results for epochs: {sorted(epoch_results.keys())}")
    
    # 计算指标
    print("\nComputing metrics for each epoch...")
    epoch_metrics = {}
    for epoch, data in epoch_results.items():
        generated_reports = data['generated_reports']
        gt_reports = data['gt_reports']
        
        if len(generated_reports) != len(gt_reports):
            print(f"Warning: Epoch {epoch} has mismatched report counts")
            continue
        
        # 计算简单指标
        metrics = compute_simple_metrics(generated_reports, gt_reports)
        epoch_metrics[epoch] = metrics
        print(f"Epoch {epoch}: {len(generated_reports)} samples")
    
    if len(epoch_metrics) == 0:
        print("Failed to compute metrics!")
        return
    
    # 输出比较结果
    print(f"\n{'='*80}")
    print(f"Epoch Comparison Results")
    print(f"{'='*80}")
    
    # 获取所有指标名称
    all_metrics = set()
    for metrics in epoch_metrics.values():
        all_metrics.update(metrics.keys())
    
    # 按指标分组显示
    for metric_name in sorted(all_metrics):
        print(f"\n{metric_name}:")
        print(f"  {'Epoch':<8} {'Value':<12} {'Samples'}")
        print(f"  {'-'*8} {'-'*12} {'-'*10}")
        
        # 按值排序
        sorted_epochs = sorted(
            epoch_metrics.items(),
            key=lambda x: x[1].get(metric_name, 0.0),
            reverse=True
        )
        
        for epoch, metrics in sorted_epochs:
            value = metrics.get(metric_name, 0.0)
            num_samples = len(epoch_results[epoch]['generated_reports'])
            print(f"  {epoch:<8} {value:<12.4f} {num_samples}")
        
        # 找出最高值
        best_epoch, best_metrics = sorted_epochs[0]
        best_value = best_metrics.get(metric_name, 0.0)
        print(f"  → Best: Epoch {best_epoch} with {metric_name} = {best_value:.4f}")
    
    print(f"\n{'='*80}")
    print("Summary: Best Epoch for Each Metric")
    print(f"{'='*80}")
    for metric_name in sorted(all_metrics):
        best_epoch = max(
            epoch_metrics.items(),
            key=lambda x: x[1].get(metric_name, 0.0)
        )[0]
        best_value = epoch_metrics[best_epoch][metric_name]
        print(f"  {metric_name:12s}: Epoch {best_epoch} ({best_value:.4f})")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Compare metrics across different epochs')
    parser.add_argument('--test_output_dir', type=str,
                        default='/mnt/sda/VLM/code/hypercode/adatri/test_output',
                        help='测试输出目录')
    parser.add_argument('--epochs', type=int, nargs='+',
                        default=[2, 4, 6, 8, 10],
                        help='要比较的epoch列表（默认：2 4 6 8 10）')
    
    args = parser.parse_args()
    
    compare_epochs(args.test_output_dir, args.epochs)


if __name__ == "__main__":
    main()
