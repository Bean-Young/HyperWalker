"""
比较不同epoch的adapter测试结果指标
"""
import json
from pathlib import Path
from typing import Dict, List
import argparse

# 导入metrics
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics.nlg_metrics import NLGMetricEvaluator


def load_epoch_results_from_summary(test_output_dir: Path) -> Dict[int, Dict]:
    """
    从test_summary文件加载不同epoch的测试结果
    Args:
        test_output_dir: 测试输出目录
    Returns:
        epoch_results: {epoch: {generated_reports: [], gt_reports: []}}
    """
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


def compute_metrics_for_epochs(epoch_results: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    为每个epoch计算指标
    Args:
        epoch_results: epoch结果字典
    Returns:
        epoch_metrics: {epoch: metrics_dict}
    """
    nlg_evaluator = NLGMetricEvaluator()
    epoch_metrics = {}
    
    for epoch, data in epoch_results.items():
        generated_reports = data['generated_reports']
        gt_reports = data['gt_reports']
        
        if len(generated_reports) != len(gt_reports):
            print(f"Warning: Epoch {epoch} has mismatched report counts")
            continue
        
        # 计算NLG指标
        metrics = nlg_evaluator.get_metrics(generated_reports, gt_reports)
        epoch_metrics[epoch] = metrics
    
    return epoch_metrics


def compare_epochs(test_output_dir: str, epochs: List[int] = [2, 4, 6, 8, 10]):
    """
    比较不同epoch的指标
    Args:
        test_output_dir: 测试输出目录
        epochs: epoch列表
    """
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
    epoch_metrics = compute_metrics_for_epochs(epoch_results)
    
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
        print(f"  {'Epoch':<8} {'Value':<12} {'File'}")
        print(f"  {'-'*8} {'-'*12} {'-'*50}")
        
        # 按值排序
        sorted_epochs = sorted(
            epoch_metrics.items(),
            key=lambda x: x[1].get(metric_name, 0.0),
            reverse=True
        )
        
        for epoch, metrics in sorted_epochs:
            value = metrics.get(metric_name, 0.0)
            file_name = epoch_results[epoch]['file']
            print(f"  {epoch:<8} {value:<12.4f} {file_name}")
        
        # 找出最高值
        best_epoch, best_value = sorted_epochs[0]
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
