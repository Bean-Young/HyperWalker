from typing import Dict, List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


class NLGMetricEvaluator:
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self.smooth_fn = SmoothingFunction().method4
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=True
        )

    @staticmethod
    def _tokenize(text: str):
        return text.strip().split()

    def _compute_single_metrics(self, pred: str, gt: str) -> Dict[str, float]:
        """计算单对报告的指标"""
        if self.lowercase:
            pred = pred.lower()
            gt = gt.lower()

        pred_tok = self._tokenize(pred)
        gt_tok = self._tokenize(gt)

        bleu_1 = sentence_bleu(
            [gt_tok],
            pred_tok,
            weights=(1.0, 0, 0, 0),
            smoothing_function=self.smooth_fn
        )

        bleu_4 = sentence_bleu(
            [gt_tok],
            pred_tok,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smooth_fn
        )

        meteor = meteor_score(
            references=[gt_tok],
            hypothesis=pred_tok
        )

        rouge_l = self.rouge_scorer.score(gt, pred)["rougeL"].fmeasure

        return {
            "BLEU-1": float(bleu_1),
            "BLEU-4": float(bleu_4),
            "METEOR": float(meteor),
            "ROUGE-L": float(rouge_l),
        }

    def get_metrics(self, generated_reports: List[str], gt_reports: List[str]) -> Dict[str, float]:
        """计算多个报告对的平均指标
        
        Args:
            generated_reports: 生成的报告列表
            gt_reports: 真实报告列表
            
        Returns:
            包含平均指标的字典
        """
        if len(generated_reports) != len(gt_reports):
            raise ValueError(f"generated_reports 和 gt_reports 的长度必须相同: "
                           f"{len(generated_reports)} vs {len(gt_reports)}")
        
        all_metrics = []
        for pred, gt in zip(generated_reports, gt_reports):
            metrics = self._compute_single_metrics(pred, gt)
            all_metrics.append(metrics)
        
        # 计算平均值
        avg_metrics = {
            "BLEU-1": sum(m["BLEU-1"] for m in all_metrics) / len(all_metrics),
            "BLEU-4": sum(m["BLEU-4"] for m in all_metrics) / len(all_metrics),
            "METEOR": sum(m["METEOR"] for m in all_metrics) / len(all_metrics),
            "ROUGE-L": sum(m["ROUGE-L"] for m in all_metrics) / len(all_metrics),
        }
        
        return avg_metrics


if __name__ == "__main__":
    evaluator = NLGMetricEvaluator()

    generated_reports = [
        "The cat sits on the mat",
        "A dog runs in the park"
    ]
    gt_reports = [
        "The cat is sitting on the mat",
        "A dog is running in the park"
    ]

    metrics = evaluator.get_metrics(generated_reports, gt_reports)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
