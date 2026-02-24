"""
Train Adapter for VLM
在VLM的最后一个decoder的最后一个FFN层后面添加adapter
当熵值低于阈值时停止训练（默认0.1）
使用peft库的inject_adapter_in_model方法直接插入LoRA adapter
"""
import os
# 设置环境变量减少显存碎片（使用新的环境变量名称）
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoTokenizer
from peft import LoraConfig, inject_adapter_in_model
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
import random

from data_loader import DataLoader


def last_layer_ffn_targets(model) -> list[str]:
    """获取最后一个decoder层的FFN目标模块名称"""
    last = len(model.model.language_model.layers) - 1
    base = f"model.language_model.layers.{last}.mlp"
    return [f"{base}.gate_proj", f"{base}.up_proj", f"{base}.down_proj"]


def freeze_all_params(model):
    """冻结模型所有参数"""
    for p in model.parameters():
        p.requires_grad = False


def inject_last_ffn_lora(model, adapter_name="last_ffn", r=16, lora_alpha=32, lora_dropout=0.05):
    """
    在最后一个FFN层注入LoRA adapter
    Args:
        model: 模型
        adapter_name: adapter名称
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    Returns:
        model: 注入adapter后的模型
        cfg: LoRA配置
        targets: 目标模块列表
    """
    targets = last_layer_ffn_targets(model)
    
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=targets,
    )
    model = inject_adapter_in_model(cfg, model, adapter_name=adapter_name)
    return model, cfg, targets


def print_trainables(model, max_lines=40):
    """打印可训练参数信息"""
    rows = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in model.named_parameters())
    trainable = sum(n for _, n in rows)
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total:.6%})")
    for n, _ in rows[:max_lines]:
        print("  ", n)


class AdapterTrainer:
    """
    Adapter训练器
    在VLM的最后一个decoder的最后一个FFN层后面添加adapter并训练
    当熵值低于阈值时停止训练
    使用peft库的inject_adapter_in_model方法直接插入LoRA adapter
    """
    def __init__(
        self,
        model_path: str = None,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        adapter_name: str = "last_ffn",
        train_sample_ratio: float = 0.05,  # 使用5%的训练数据
        output_dir: str = "/mnt/sda/VLM/code/hypercode/adamodel"
    ):
        """
        Args:
            model_path: VLM模型路径
            r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            adapter_name: adapter名称
            train_sample_ratio: 训练数据采样比例（默认5%）
            output_dir: 模型保存目录
        """
        if model_path is None:
            model_path = "/mnt/sda/VLM/code/model_cache/models--google--medgemma-4b-it/snapshots/290cda5eeccbee130f987c4ad74a59ae6f196408"
        
        self.model_path = model_path
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.adapter_name = adapter_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型 - 使用 Flash Attention 2 以降低显存占用
        print("Loading model with Flash Attention 2...")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"  # 使用 Flash Attention 2 降低显存占用
        )
        self.model.config.use_cache = False
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 冻结基座模型
        freeze_all_params(self.model)
        
        # 注入LoRA adapter到最后一个FFN层
        print("Injecting LoRA adapter...")
        self.model, self.lora_cfg, self.target_modules = inject_last_ffn_lora(
            self.model,
            adapter_name=self.adapter_name,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout
        )
        
        # 修复 backward() 报错的关键设置
        # 在开启梯度检查点时，必须让输入层强制开启梯度计算
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        # 开启梯度检查点，使用 use_reentrant=False 可以避免警告
        # 检查是否支持 gradient_checkpointing_kwargs
        import inspect
        sig = inspect.signature(self.model.gradient_checkpointing_enable)
        if "gradient_checkpointing_kwargs" in sig.parameters:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            self.model.gradient_checkpointing_enable()
        
        # 确保 LoRA 参数是可训练的（inject_adapter_in_model 之后再检查一遍）
        for name, param in self.model.named_parameters():
            if self.adapter_name in name:
                param.requires_grad = True
        
        print("Target modules:", self.target_modules)
        print_trainables(self.model)
        
        # 加载数据
        print("Loading data...")
        self.data_loader = DataLoader(train_sample_ratio=train_sample_ratio)
        self.train_studies = self.data_loader.get_train_studies()
        print(f"Loaded {len(self.train_studies)} training studies (using {train_sample_ratio*100:.1f}% of training data)")
    
    def prepare_training_data(self, study: Dict) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        准备训练数据：从study中提取EHR、图像和报告，生成输入和标签
        Args:
            study: study数据字典
        Returns:
            (input_ids, labels) 或 None
        """
        # 获取EHR数据并格式化为文本
        ehr_data = self.data_loader.get_ehr_data(study)
        if not isinstance(ehr_data, dict):
            return None
        
        ehr_text_parts = []
        
        # 格式化EHR字段
        ehr_field_names = [
            'patient_info',
            'admissions',
            'all_diagnoses',
            'all_procedures',
            'transfers',
            'icustays',
            'chartevents',
            'prescriptions',
            'all_labevents',
            'microbiologyevents',
            'inputevents',
            'outputevents'
        ]
        
        for field_name in ehr_field_names:
            if field_name in ehr_data and ehr_data[field_name] is not None:
                field_text = self.data_loader.format_ehr_field(field_name, ehr_data[field_name])
                if field_text and isinstance(field_text, str):
                    ehr_text_parts.append(f"{field_name}: {field_text}")
        
        ehr_text = "\n".join(ehr_text_parts) if ehr_text_parts else "No patient information available."
        
        # 获取图像路径
        image_paths = self.data_loader.get_image_paths(study)
        if not image_paths or len(image_paths) == 0:
            return None
        
        # 确保 image_paths 是列表
        if not isinstance(image_paths, list):
            return None
        
        
        # 获取报告文本
        report_text = self.data_loader.get_report_text(study)
        if not report_text or not isinstance(report_text, str):
            return None
        
        # 构建消息：prompt + EHR文本 + 所有图像 + 报告生成任务
        # 注意：将 prompt 放在最前面，确保即使截断也不会丢失 prompt
        user_content = []
        
        # 先添加prompt文本（放在最前面，确保不被截断）
        prompt_text = "Please generate a paragraph of radiology report for this chest X-ray image."
        user_content.append({"type": "text", "text": prompt_text})
        
        # 添加EHR文本
        if ehr_text and isinstance(ehr_text, str):
            user_content.append({"type": "text", "text": ehr_text})
        
        # 添加所有图像（确保每个路径都是字符串）
        for image_path in image_paths:
            if not isinstance(image_path, str):
                # 如果不是字符串，转换为字符串
                image_path = str(image_path)
            # 验证图像路径是否存在
            from pathlib import Path
            if not Path(image_path).exists():
                # 如果路径不存在，跳过这个图像
                continue
            # 使用 "path" 键存储本地文件路径（processor支持 "image", "url", "path", "base64"）
            user_content.append({"type": "image", "path": image_path})
        
        # 如果所有图像都不存在，返回None
        image_items = [item for item in user_content if item.get("type") == "image"]
        if len(image_items) == 0:
            return None
        
        # 验证 user_content 中的所有元素都是字典
        for idx, item in enumerate(user_content):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid content item at index {idx}: {item}, type: {type(item)}")
            if "type" not in item:
                raise ValueError(f"Content item at index {idx} missing 'type' key: {item}")
        
        # 将 assistant 的 content 也改为列表格式，保持一致性
        messages = [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": report_text}]
            }
        ]
        
        # 验证 messages 格式
        if not isinstance(messages, list):
            raise ValueError(f"messages must be a list, got {type(messages)}")
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(f"Invalid message format: {msg}")
            if not isinstance(msg["content"], list) and not isinstance(msg["content"], str):
                raise ValueError(f"Message content must be list or str, got {type(msg['content'])}: {msg['content']}")
        
        # 使用processor处理（不截断，异常长的样本会在后面跳过）
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 检查 inputs 的类型（BatchFeature 也支持字典操作）
        if not (isinstance(inputs, dict) or hasattr(inputs, 'get')):
            return None
        
        # 移动到模型设备
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return None
        
        # 检查序列长度，跳过异常过长的样本
        max_seq_length = 16384  # 最大序列长度阈值
        seq_length = input_ids.shape[1] if len(input_ids.shape) > 1 else len(input_ids)
        if seq_length > max_seq_length:
            # 返回特殊标记，表示样本过长被跳过
            return {"skip_reason": "sequence_too_long", "seq_length": seq_length, "max_length": max_seq_length}
        
        input_ids = input_ids.to(self.model.device)
        
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.model.device, dtype=torch.bfloat16)
        
        # 创建labels（用于计算loss）
        labels = input_ids.clone()
        # 将padding token的label设为-100（忽略）
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "labels": labels
        }
    
    def compute_entropy(self, logits: torch.Tensor) -> float:
        """
        计算模型输出的熵值（节省显存版本）
        将计算移到CPU上，避免GPU显存溢出
        Args:
            logits: (batch_size, seq_len, vocab_size) 模型输出的logits
        Returns:
            entropy: 平均熵值
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # 只计算前20个位置的熵值，进一步减少计算量
        max_positions = min(20, seq_len)
        logits_subset = logits[:, :max_positions, :]  # (batch_size, max_positions, vocab_size)
        
        # 将 logits 移到 CPU 上计算熵值，释放 GPU 显存
        logits_cpu = logits_subset.cpu()
        del logits_subset  # 立即删除 GPU 上的数据
        
        # 使用 torch.distributions.Categorical 计算熵值
        dist = torch.distributions.Categorical(logits=logits_cpu)
        entropy = dist.entropy()  # (batch_size, max_positions)
        
        # 返回平均熵值
        avg_entropy = entropy.mean().item()
        
        # 立即删除大对象释放内存
        del logits_cpu, dist, entropy
        
        return avg_entropy
    
    def train(
        self,
        entropy_threshold: float = 0.1,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        max_epochs: int = 100  # Maximum epochs (safety limit)
    ):
        """
        训练adapter，当熵值低于阈值时停止
        Args:
            entropy_threshold: 熵值阈值，低于此值则停止训练（默认0.1）
            batch_size: 批次大小
            learning_rate: 学习率
            save_steps: 每多少步保存一次
            max_epochs: 最大训练轮数（防止无限训练）
        """
        # 设置优化器（只优化adapter参数）
        adapter_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        if len(adapter_params) == 0:
            raise RuntimeError("No trainable parameters found! Adapter may not be properly injected.")
        print(f"Optimizer will train {len(adapter_params)} parameter groups with {sum(p.numel() for p in adapter_params):,} parameters")
        optimizer = torch.optim.AdamW(adapter_params, lr=learning_rate)
        
        # 设置模型为训练模式
        self.model.train()
        
        # 训练循环
        global_step = 0
        total_loss = 0.0
        total_entropy = 0.0
        entropy_count = 0
        last_epoch = 0  # 记录最后完成的epoch
        
        for epoch in range(max_epochs):
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
            print(f"Entropy threshold: {entropy_threshold}")
            
            # 每一轮只使用0.5%的数据：从5%中随机选择10%
            epoch_sample_ratio = 0.10  # 10% of 5% = 0.5%
            num_samples = max(1, int(len(self.train_studies) * epoch_sample_ratio))
            epoch_studies = random.sample(self.train_studies, num_samples)
            print(f"Using {len(epoch_studies)} samples ({epoch_sample_ratio*100:.0f}% of {len(self.train_studies)} total samples, i.e., 0.5% of full dataset)")
            
            epoch_loss = 0.0
            epoch_entropy = 0.0
            epoch_entropy_count = 0  # 实际计算了熵值的样本数
            num_batches = 0
            num_samples_processed = 0  # 实际处理的样本数
            num_skipped = 0  # 跳过的样本数
            skip_reasons = {}  # 跳过原因统计
            
            for i in tqdm(range(0, len(epoch_studies), batch_size), desc=f"Epoch {epoch + 1}"):
                batch_studies = epoch_studies[i:i + batch_size]
                
                # 准备批次数据
                batch_data = []
                for study in batch_studies:
                    # 先检查基本数据
                    ehr_data = self.data_loader.get_ehr_data(study)
                    image_paths = self.data_loader.get_image_paths(study)
                    report_text = self.data_loader.get_report_text(study)
                    
                    if not isinstance(ehr_data, dict):
                        num_skipped += 1
                        skip_reasons['invalid_ehr'] = skip_reasons.get('invalid_ehr', 0) + 1
                        continue
                    if not image_paths or len(image_paths) == 0:
                        num_skipped += 1
                        skip_reasons['no_images'] = skip_reasons.get('no_images', 0) + 1
                        continue
                    if not report_text or not isinstance(report_text, str):
                        num_skipped += 1
                        skip_reasons['no_report'] = skip_reasons.get('no_report', 0) + 1
                        continue
                    
                    # 尝试准备数据
                    data = self.prepare_training_data(study)
                    if data is not None:
                        # 检查是否是跳过标记
                        if isinstance(data, dict) and "skip_reason" in data:
                            num_skipped += 1
                            skip_reason = data["skip_reason"]
                            skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                        else:
                            batch_data.append(data)
                    else:
                        num_skipped += 1
                        skip_reasons['processor_failed'] = skip_reasons.get('processor_failed', 0) + 1
                
                if not batch_data:
                    continue
                
                # 由于batch_size=1，直接处理单个样本
                if len(batch_data) > 0:
                    data = batch_data[0]
                    
                    optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = self.model(
                        input_ids=data["input_ids"],
                        pixel_values=data.get("pixel_values"),
                        labels=data["labels"]
                    )
                    
                    loss = outputs.loss
                    
                    # 在反向传播前，如果不是计算熵值的步，立即删除 logits 释放显存
                    compute_entropy_step = (global_step + 1) % 10 == 0
                    if not compute_entropy_step and hasattr(outputs, "logits"):
                        del outputs.logits
                    
                    # 反向传播
                    loss.backward()
                    
                    # 立即删除 outputs 释放显存（在 step 之前）
                    del outputs
                    
                    # 在计算熵之前先做 step 并释放不必要的计算图
                    optimizer.step()
                    
                    # 记录 loss（在删除之前）
                    loss_value = loss.item()
                    epoch_loss += loss_value
                    total_loss += loss_value
                    del loss
                    
                    # 计算熵值（如果需要，重新前向传播一次，但只在 eval 模式下）
                    entropy = float('inf')
                    if compute_entropy_step:
                        # 切换到 eval 模式，重新前向传播计算熵值（不计算 loss，节省显存）
                        self.model.eval()
                        with torch.no_grad():
                            eval_outputs = self.model(
                                input_ids=data["input_ids"],
                                pixel_values=data.get("pixel_values")
                            )
                            if hasattr(eval_outputs, "logits"):
                                entropy = self.compute_entropy(eval_outputs.logits)
                                
                                epoch_entropy += entropy
                                epoch_entropy_count += 1
                                total_entropy += entropy
                                entropy_count += 1
                                
                                # 检查停止条件
                                if entropy < entropy_threshold:
                                    print(f"Threshold reached! Entropy ({entropy:.4f}) < threshold ({entropy_threshold})")
                                    # 保存最终模型
                                    self.save_checkpoint(epoch + 1, loss_value, entropy=entropy, final=True)
                                    return
                            del eval_outputs
                        # 切换回训练模式
                        self.model.train()
                    
                    # 彻底清理
                    del data
                    
                    num_batches += 1
                    num_samples_processed += 1
                    global_step += 1
                    
            
            # 每个epoch结束后的平均loss和熵值
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            avg_entropy = epoch_entropy / epoch_entropy_count if epoch_entropy_count > 0 else float('inf')
            print(f"Epoch {epoch + 1} - Processed {num_samples_processed} samples, Skipped {num_skipped} samples")
            if skip_reasons:
                print(f"Epoch {epoch + 1} - Skip reasons: {skip_reasons}")
            print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}, Average entropy per sample: {avg_entropy:.4f}")
            
            # 每2轮保存一次adapter
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint(epoch + 1, avg_loss, entropy=avg_entropy)
            
            last_epoch = epoch + 1  # 更新最后完成的epoch
            
            # 检查停止条件：如果平均熵值低于阈值，停止训练
            if avg_entropy < entropy_threshold:
                print(f"\nTraining stopped: Average entropy ({avg_entropy:.4f}) < threshold ({entropy_threshold})")
                break
        
        # 最终保存
        final_avg_entropy = total_entropy / entropy_count if entropy_count > 0 else float('inf')
        print(f"\nTraining completed! Final average entropy: {final_avg_entropy:.4f}")
        self.save_checkpoint(last_epoch, total_loss / global_step if global_step > 0 else 0.0, 
                           entropy=final_avg_entropy, final=True)
    
    def save_checkpoint(self, epoch: int, loss: float, entropy: float = None, final: bool = False):
        """保存checkpoint，使用peft的save_pretrained方法，只保存adapter"""
        if final:
            checkpoint_dir = self.output_dir / "adapter_final"
        else:
            checkpoint_dir = self.output_dir / f"adapter_epoch_{epoch}"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用peft的save_pretrained方法保存adapter
        # 这会保存adapter权重和配置
        self.model.save_pretrained(str(checkpoint_dir), adapter_name=self.adapter_name)
        
        # 保存额外的训练信息
        config_data = {
            'epoch': epoch,
            'loss': loss,
            'r': self.r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'adapter_name': self.adapter_name,
            'target_modules': self.target_modules,
        }
        if entropy is not None:
            config_data['entropy'] = entropy
        
        config_path = checkpoint_dir / "training_info.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        entropy_str = f", entropy: {entropy:.4f}" if entropy is not None else ""
        print(f"Saved checkpoint to {checkpoint_dir} (loss: {loss:.4f}{entropy_str})")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Adapter for VLM')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to VLM model')
    parser.add_argument('--r', type=int, default=16,
                        help='LoRA rank (default: 16)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha (default: 32)')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout (default: 0.05)')
    parser.add_argument('--adapter_name', type=str, default="last_ffn",
                        help='Adapter name (default: last_ffn)')
    parser.add_argument('--entropy_threshold', type=float, default=0.1,
                        help='Entropy threshold to stop training (default: 0.1)')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of training epochs (safety limit)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--output_dir', type=str, default="/mnt/sda/VLM/code/hypercode/adamodel",
                        help='Output directory for checkpoints (default: /mnt/sda/VLM/code/hypercode/adamodel)')
    parser.add_argument('--train_sample_ratio', type=float, default=0.05,
                        help='Training data sample ratio (default: 0.05, i.e., 5%%)')
    
    args = parser.parse_args()
    
    # 创建trainer
    trainer = AdapterTrainer(
        model_path=args.model_path,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        adapter_name=args.adapter_name,
        train_sample_ratio=args.train_sample_ratio,
        output_dir=args.output_dir
    )
    
    # 开始训练
    trainer.train(
        entropy_threshold=args.entropy_threshold,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs
    )


if __name__ == "__main__":
    main()

