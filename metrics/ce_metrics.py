import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

class CEMetricEvaluator:
    def __init__(self):
        # 使用当前文件所在目录作为基准路径
        current_dir = Path(__file__).parent
        self.csv_path = str(current_dir / "tmp" / "tmp.csv")
        self.checkpoint_path = str(current_dir / "CheXbert" / "chexbert.pth")
        self.output_dir = str(current_dir / "tmp")
        self.chexbert_src_dir = str(current_dir / "CheXbert" / "src")
        
        # 确保目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 设置 Hugging Face 镜像站点
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # 禁用 tokenizers 并行化警告（避免 fork 后的死锁警告）
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def get_metrics(self, generated_reports: list, label_vecs: list) -> dict:
        # 确保所有报告都是字符串类型，处理None和非字符串值
        cleaned_reports = []
        for idx, report in enumerate(generated_reports):
            if report is None:
                cleaned_reports.append("")  # 将None替换为空字符串
                print(f"Warning: generated_reports[{idx}] is None, replaced with empty string")
            elif not isinstance(report, str):
                cleaned_reports.append(str(report))  # 转换为字符串
                print(f"Warning: generated_reports[{idx}] is not string (type: {type(report)}), converted to string")
            else:
                cleaned_reports.append(report)
        
        # 确保label_vecs和cleaned_reports长度一致
        if len(cleaned_reports) != len(label_vecs):
            raise ValueError(f"Length mismatch: generated_reports ({len(cleaned_reports)}) != label_vecs ({len(label_vecs)})")
        
        try:
            csv_data = pd.DataFrame({"Report Impression": cleaned_reports})
            csv_data.to_csv(self.csv_path, index=False)
        except Exception as e:
            raise ValueError(f"Failed to create CSV: {e}. First few reports: {cleaned_reports[:3]}")
        
        # 保存当前工作目录和 sys.path
        original_cwd = os.getcwd()
        # 将 CheXbert/src 目录添加到 sys.path，以便 Python 能找到模块
        if self.chexbert_src_dir not in sys.path:
            sys.path.insert(0, self.chexbert_src_dir)
        
        try:
            # 更改工作目录到 CheXbert/src，以便相对导入能正常工作
            os.chdir(self.chexbert_src_dir)
            # 动态导入，确保在正确的工作目录下导入
            from label import label, save_preds
            y_pred = label(self.checkpoint_path, self.csv_path)
            save_preds(y_pred, self.csv_path, self.output_dir)
        finally:
            # 恢复原始工作目录
            os.chdir(original_cwd)
        pred_labels = []
        true_labels = []
        labeled_reports = pd.read_csv(os.path.join(self.output_dir, "labeled_reports.csv")).fillna(0)
        labels_cls = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
        
        for idx, lvec in enumerate(label_vecs):
            # 确保lvec是列表类型
            if not isinstance(lvec, list):
                # 如果不是列表，尝试转换或使用默认值
                if lvec is None:
                    lvec = [0] * len(labels_cls)
                else:
                    try:
                        lvec = list(lvec)
                    except:
                        lvec = [0] * len(labels_cls)
            
            # 确保lvec长度正确
            if len(lvec) != len(labels_cls):
                # 如果长度不匹配，使用默认值
                lvec = [0] * len(labels_cls)
            
            pred_label_vec = [labeled_reports[cl].iloc[idx] for cl in labels_cls]
            pred_label_vec = [0 if x == -1 else x for x in pred_label_vec]
            true_label_vec = [0 if x == -1 else (int(x) if isinstance(x, (int, float)) else 0) for x in lvec]
            pred_labels.append(pred_label_vec)
            true_labels.append(true_label_vec)
        report = classification_report(
            pred_labels, 
            true_labels, 
            target_names=labels_cls,
            output_dict=True,
            digits=4,
            zero_division=0
        )
        return report["macro avg"]

if __name__ == "__main__":
    evaluator = CEMetricEvaluator()
    generated_reports = ["The patient has a left-sided pneumothorax.", "The patient has a right-sided pneumothorax."]
    label_vecs = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    report = evaluator.get_metrics(generated_reports, label_vecs)
    print(report)