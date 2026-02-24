"""
Data Loader
数据加载工具
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Optional


class DataLoader:
    """
    数据加载器
    从matched_data/ehrdata目录加载数据，支持训练/测试集划分
    """
    def __init__(
        self, 
        ehrdata_dir: str = "/mnt/sda/VLM/matched_data/ehrdata",
        split_statistics_file: str = "/mnt/sda/VLM/matched_data/ehrdata/split_statistics.json",
        train_sample_ratio: float = 0.01  # 训练数据采样比例（1%）
    ):
        """
        Args:
            ehrdata_dir: EHR数据目录（按subject_id组织）
            split_statistics_file: 数据集划分统计文件
            train_sample_ratio: 训练数据采样比例（默认1%）
        """
        self.ehrdata_dir = Path(ehrdata_dir)
        self.split_statistics_file = Path(split_statistics_file)
        self.train_sample_ratio = train_sample_ratio
        
        # 加载数据集划分信息
        self._load_split_statistics()
    
    def _load_split_statistics(self):
        """加载数据集划分统计"""
        with open(self.split_statistics_file, 'r') as f:
            split_data = json.load(f)
        
        self.train_subjects = set(split_data.get('train', []))
        self.val_subjects = set(split_data.get('val', []))
        self.test_subjects = set(split_data.get('test', []))
        
        # 训练数据随机采样
        train_list = list(self.train_subjects)
        sample_size = max(1, int(len(train_list) * self.train_sample_ratio))
        self.train_subjects_sampled = set(random.sample(train_list, sample_size))
    
    def load_study(self, subject_id: int, study_id: int) -> Optional[Dict]:
        """加载单个study的数据"""
        study_file = self.ehrdata_dir / str(subject_id) / f"{study_id}.json"
        with open(study_file, 'r') as f:
            return json.load(f)
    
    def load_all_studies(self, split: str = "train") -> List[Dict]:
        """
        加载指定split的所有studies
        Args:
            split: "train", "val", or "test"
        Returns:
            studies: 所有study数据的列表
        """
        studies = []
        
        if split == "train":
            subject_ids = self.train_subjects_sampled
        elif split == "test":
            subject_ids = self.test_subjects
        else:
            subject_ids = []
        
        for subject_id in subject_ids:
            subject_dir = self.ehrdata_dir / str(subject_id)
          
            # 加载该subject下的所有study文件
            for study_file in subject_dir.glob("*.json"):
                with open(study_file, 'r') as f:
                    study_data = json.load(f)
                    studies.append(study_data)
            
        return studies
    
    def get_train_studies(self) -> List[Dict]:
        """获取训练集studies（已采样）"""
        return self.load_all_studies("train")
    
    def get_test_studies(self) -> List[Dict]:
        """获取测试集studies（全部）"""
        return self.load_all_studies("test")
    
    def get_ehr_data(self, study: Dict) -> Dict:
        """从study中提取EHR数据"""
        return study.get('ehr_data', {})
    
    def get_image_paths(self, study: Dict) -> List[str]:
        """从study中提取图像路径"""
        image_paths = []
        
        if not isinstance(study, dict):
            return image_paths
        
        if 'images' in study:
            images = study['images']
            # 确保 images 是列表
            if isinstance(images, list):
                for img_info in images:
                    if isinstance(img_info, dict) and img_info.get('exists') and 'absolute_path' in img_info:
                        image_paths.append(img_info['absolute_path'])
        
        return image_paths
    
    def get_report_text(self, study: Dict) -> Optional[str]:
        """从study中提取报告文本"""
        return study.get('report', '')
    
    def format_ehr_field(self, field_name: str, field_data: any) -> str:
        """
        格式化单个EHR字段为文本
        Args:
            field_name: 字段名（如 "patient_info", "admissions"）
            field_data: 字段数据
        Returns:
            formatted_text: 格式化后的文本
        """
        if field_name == "patient_info":
            # 格式化患者基本信息
            if not isinstance(field_data, dict):
                return str(field_data)
            info = field_data
            parts = []
            if isinstance(info, dict):
                if 'gender' in info:
                    parts.append(f"Gender: {info['gender']}")
                if 'anchor_age' in info:
                    parts.append(f"Age: {info['anchor_age']}")
                if 'anchor_year_group' in info:
                    parts.append(f"Year Group: {info['anchor_year_group']}")
            return " | ".join(parts) if parts else ""
        else:
            if isinstance(field_data, list):
                formatted_items = []
                for item in field_data:
                    if isinstance(item, dict):
                        item_str = ", ".join([f"{k}: {v}" for k, v in item.items() if v is not None])
                        formatted_items.append(item_str)
                    else:
                        formatted_items.append(str(item))
                return " | ".join(formatted_items)
            elif isinstance(field_data, dict):
                # 如果是字典但不是列表，直接格式化
                return ", ".join([f"{k}: {v}" for k, v in field_data.items() if v is not None])
            else:
                return str(field_data)

