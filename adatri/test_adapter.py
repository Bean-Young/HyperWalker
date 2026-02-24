"""
测试Adapter模型
使用adapter_epoch_10的权重进行单样本测试
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import json
from datetime import datetime

from data_loader import DataLoader


def load_adapter_model(model_path: str = None, adapter_path: str = None):
    """
    加载基础模型和adapter
    Args:
        model_path: 基础VLM模型路径
        adapter_path: Adapter权重路径
    Returns:
        model: 加载adapter后的模型
        processor: 处理器
        tokenizer: 分词器
    """
    # 确定基础模型路径
    if model_path is None:
        model_path = "/mnt/sda/VLM/code/model_cache/models--google--medgemma-4b-it/snapshots/290cda5eeccbee130f987c4ad74a59ae6f196408"
    
    if adapter_path is None:
        adapter_path = "/mnt/sda/VLM/code/hypercode/adatri/adamodel/adapter_epoch_10"
    
    print(f"Loading base model from: {model_path}")
    # 加载基础模型
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    )
    
    print(f"Loading adapter from: {adapter_path}")
    # 加载adapter
    adapter_path_obj = Path(adapter_path)
    
    # 检查是否有adapter_config.json
    adapter_config_file = adapter_path_obj / "adapter_config.json"
    if adapter_config_file.exists():
        # 标准PEFT adapter格式
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path_obj),
            adapter_name="last_ffn",
            local_files_only=True
        )
    else:
        # 没有adapter_config.json，从training_info.json创建配置
        training_info_file = adapter_path_obj / "training_info.json"
        if training_info_file.exists():
            import json
            with open(training_info_file, 'r') as f:
                training_info = json.load(f)
            
            # 从training_info.json创建LoRA配置
            from peft import LoraConfig, get_peft_model
            
            peft_config = LoraConfig(
                r=training_info.get("r", 16),
                lora_alpha=training_info.get("lora_alpha", 32),
                target_modules=training_info.get("target_modules", []),
                lora_dropout=training_info.get("lora_dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # 创建PEFT模型
            model = get_peft_model(model, peft_config, adapter_name="last_ffn")
            
            # 直接加载本地safetensors文件
            import safetensors.torch
            adapter_weights = {}
            
            # 检查是否有分片的safetensors文件
            index_file = adapter_path_obj / "model.safetensors.index.json"
            if index_file.exists():
                # 分片权重文件
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                
                weight_map = index_data.get("weight_map", {})
                for weight_name, shard_file in weight_map.items():
                    shard_path = adapter_path_obj / shard_file
                    if shard_path.exists():
                        shard_weights = safetensors.torch.load_file(str(shard_path))
                        # 只加载LoRA相关的权重
                        for key, value in shard_weights.items():
                            if "lora" in key.lower() or "last_ffn" in key:
                                adapter_weights[key] = value
            else:
                # 单个safetensors文件或查找所有safetensors文件
                safetensors_files = list(adapter_path_obj.glob("*.safetensors"))
                for safetensors_file in safetensors_files:
                    weights = safetensors.torch.load_file(str(safetensors_file))
                    # 只加载LoRA相关的权重
                    for key, value in weights.items():
                        if "lora" in key.lower() or "last_ffn" in key:
                            adapter_weights[key] = value
            
            if not adapter_weights:
                raise ValueError(f"No LoRA weights found in {adapter_path}")
            
            # 手动加载权重到模型
            from peft.utils import set_peft_model_state_dict
            set_peft_model_state_dict(model, adapter_weights, adapter_name="last_ffn")
        else:
            raise ValueError(f"Cannot load adapter from {adapter_path}. Missing adapter_config.json and training_info.json.")
    
    # 设置adapter为活跃状态
    model.set_adapter("last_ffn")
    model.eval()
    
    print("Loading processor and tokenizer...")
    # 加载processor和tokenizer
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    print("Model loaded successfully!")
    return model, processor, tokenizer


def prepare_test_data(study: dict, data_loader: DataLoader, processor, tokenizer, model):
    """
    准备测试数据
    Args:
        study: 样本数据
        data_loader: 数据加载器
        processor: 处理器
        tokenizer: 分词器
        model: 模型（用于获取设备）
    Returns:
        inputs: 模型输入
        ehr_text: EHR文本
        report_text: 真实报告文本
        image_path: 图像路径
    """
    # 获取EHR数据
    ehr_data = data_loader.get_ehr_data(study)
    if not ehr_data:
        return None, None, None, None
    
    # 格式化EHR文本（使用与train_adapter.py相同的字段）
    ehr_text_parts = []
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
            field_text = data_loader.format_ehr_field(field_name, ehr_data[field_name])
            if field_text and isinstance(field_text, str):
                ehr_text_parts.append(f"{field_name}: {field_text}")
    
    ehr_text = "\n".join(ehr_text_parts) if ehr_text_parts else "No patient information available."
    
    # 获取图像路径
    image_paths = data_loader.get_image_paths(study)
    if not image_paths or len(image_paths) == 0:
        return None, None, None, None
    
    image_path = image_paths[0] if isinstance(image_paths, list) else str(image_paths)
    
    # 获取报告文本
    report_text = data_loader.get_report_text(study)
    if not report_text or not isinstance(report_text, str):
        return None, None, None, None
    
    # 构建消息（与train_adapter.py保持一致）
    prompt_text = "Please generate a paragraph of radiology report for this chest X-ray image."
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "text", "text": ehr_text},
            {"type": "image", "image": image_path}
        ]
    }]
    
    # 使用processor处理
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    # 移动到模型设备（确保所有tensor都在同一设备上）
    device = next(model.parameters()).device
    
    # 将inputs字典中的所有tensor移动到设备
    if isinstance(inputs, dict):
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)
            elif isinstance(value, (list, tuple)):
                # 处理列表或元组中的tensor
                inputs[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
    else:
        # 如果不是字典，尝试使用get方法
        if hasattr(inputs, 'get'):
            for key in ['input_ids', 'pixel_values', 'attention_mask', 'position_ids']:
                if hasattr(inputs, key):
                    value = getattr(inputs, key)
                    if isinstance(value, torch.Tensor):
                        setattr(inputs, key, value.to(device))
    
    return inputs, ehr_text, report_text, image_path


def generate_report(model, inputs, tokenizer, max_new_tokens=512):
    """
    生成报告
    Args:
        model: 模型
        inputs: 输入
        tokenizer: 分词器
        max_new_tokens: 最大生成token数
    Returns:
        generated_text: 生成的报告文本（只包含新生成的部分，不包含输入）
    """
    # 确保所有输入都在正确的设备上
    device = next(model.parameters()).device
    
    # 获取输入长度（用于后续截取新生成的部分）
    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        input_length = input_ids.shape[1]
    else:
        input_length = 0
    
    # 准备generate的参数，确保所有tensor在正确设备上
    generate_kwargs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            generate_kwargs[key] = value.to(device)
        else:
            generate_kwargs[key] = value
    
    with torch.no_grad():
        outputs = model.generate(
            **generate_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # 只解码新生成的部分（从input_length开始）
        # outputs[0] 包含 [输入部分 + 新生成部分]
        if input_length > 0 and len(outputs[0]) > input_length:
            generated_ids = outputs[0][input_length:]  # 只取新生成的部分
        else:
            generated_ids = outputs[0]  # 如果没有输入或输出长度异常，使用全部
        
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text


def test_single_adapter(epoch: int, model_path: str, data_loader: DataLoader, test_study: dict, output_dir: Path):
    """
    测试单个epoch的adapter
    Args:
        epoch: epoch编号（2, 4, 6, 8, 10）
        model_path: 基础模型路径
        data_loader: 数据加载器
        test_study: 测试样本
        output_dir: 输出目录
    Returns:
        result_data: 结果数据字典
    """
    adapter_path = f"/mnt/sda/VLM/code/hypercode/adatri/adamodel/adapter_epoch_{epoch}"
    
    print(f"\n{'='*80}")
    print(f"Testing Adapter Epoch {epoch}")
    print(f"{'='*80}")
    
    # 加载模型和adapter
    model, processor, tokenizer = load_adapter_model(model_path=model_path, adapter_path=adapter_path)
    
    # 准备测试数据
    inputs, ehr_text, report_text, image_path = prepare_test_data(
        test_study, data_loader, processor, tokenizer, model
    )
    
    if inputs is None:
        print(f"Failed to prepare test data for epoch {epoch}!")
        return None
    
    print(f"Generating report with epoch {epoch} adapter...")
    # 生成报告
    generated_report = generate_report(model, inputs, tokenizer)
    
    # 计算CE loss（在真实报告上）
    print(f"Calculating CE loss for epoch {epoch}...")
    device = next(model.parameters()).device
    
    # 构建包含真实报告的输入用于计算loss
    prompt_text = "Please generate a paragraph of radiology report for this chest X-ray image."
    
    messages_with_gt = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "text", "text": ehr_text},
            {"type": "image", "image": image_path}
        ]
    }, {
        "role": "assistant",
        "content": [{"type": "text", "text": report_text}]
    }]
    
    inputs_with_gt = processor.apply_chat_template(
        messages_with_gt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    # 移动到设备
    for key, value in inputs_with_gt.items():
        if isinstance(value, torch.Tensor):
            inputs_with_gt[key] = value.to(device)
    
    # 创建labels（只对assistant回复部分计算loss）
    input_ids_gt = inputs_with_gt["input_ids"]
    labels = input_ids_gt.clone()
    labels.fill_(-100)  # 默认全部忽略
    
    # 找到assistant回复在input_ids中的位置
    report_ids = tokenizer(
        report_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=512
    )["input_ids"].to(device)
    
    report_len = report_ids.shape[1]
    input_len = input_ids_gt.shape[1]
    
    if report_len > 0 and report_len < input_len:
        # 尝试在input_ids中找到report的位置
        found_match = False
        for start_idx in range(max(0, input_len - report_len - 10), input_len - report_len + 1):
            if start_idx + report_len <= input_len:
                candidate = input_ids_gt[0, start_idx:start_idx + report_len]
                if torch.equal(candidate, report_ids[0, :min(report_len, candidate.shape[0])]):
                    labels[0, start_idx:start_idx + report_len] = report_ids[0, :min(report_len, candidate.shape[0])]
                    found_match = True
                    break
        
        # 如果没找到，使用简化方法（假设在最后）
        if not found_match:
            labels[0, -report_len:] = report_ids[0]
    
    # 计算CE loss
    model.eval()
    with torch.no_grad():
        loss_outputs = model(
            input_ids=input_ids_gt,
            pixel_values=inputs_with_gt.get("pixel_values"),
            labels=labels
        )
        ce_loss = loss_outputs.loss.item()
    
    num_valid_labels = (labels[0] != -100).sum().item()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"test_result_epoch_{epoch}_{timestamp}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Adapter Test Result - Epoch {epoch}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Adapter Path: {adapter_path}\n")
        f.write(f"Sample Subject ID: {test_study.get('subject_id', 'unknown')}\n")
        f.write(f"Image Path: {image_path}\n")
        f.write(f"CE Loss: {ce_loss:.4f}\n")
        f.write(f"Valid Labels: {num_valid_labels}/{input_len}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("EHR Text:\n")
        f.write("-"*80 + "\n")
        f.write(ehr_text + "\n\n")
        
        f.write("-"*80 + "\n")
        f.write("Generated Report:\n")
        f.write("-"*80 + "\n")
        f.write(generated_report + "\n\n")
        
        f.write("-"*80 + "\n")
        f.write("Ground Truth Report:\n")
        f.write("-"*80 + "\n")
        f.write(report_text + "\n\n")
    
    # 同时保存JSON格式
    json_file = output_dir / f"test_result_epoch_{epoch}_{timestamp}.json"
    result_data = {
        "epoch": epoch,
        "timestamp": timestamp,
        "adapter_path": adapter_path,
        "sample": {
            "subject_id": test_study.get('subject_id', 'unknown'),
            "image_path": image_path
        },
        "metrics": {
            "ce_loss": ce_loss,
            "valid_labels": num_valid_labels,
            "total_input_length": input_len
        },
        "ehr_text": ehr_text,
        "generated_report": generated_report,
        "ground_truth_report": report_text
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved:")
    print(f"  Text: {output_file}")
    print(f"  JSON: {json_file}")
    print(f"  CE Loss: {ce_loss:.4f} (valid labels: {num_valid_labels}/{input_len})")
    
    # 清理模型以释放显存
    del model
    torch.cuda.empty_cache()
    
    return result_data


def main():
    """主函数：测试所有epoch的adapter"""
    # 要测试的epoch列表
    epochs = [2, 4, 6, 8, 10]
    
    # 基础模型路径
    model_path = "/mnt/sda/VLM/code/model_cache/models--google--medgemma-4b-it/snapshots/290cda5eeccbee130f987c4ad74a59ae6f196408"
    
    # 加载数据
    print("Loading test data...")
    data_loader = DataLoader()
    train_studies = data_loader.get_train_studies()
    
    if len(train_studies) == 0:
        print("No training studies found!")
        return
    
    # 选择第一个样本进行测试
    test_study = train_studies[0]
    print(f"Testing with sample: {test_study.get('subject_id', 'unknown')}")
    
    # 创建输出目录
    output_dir = Path("/mnt/sda/VLM/code/hypercode/adatri/test_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 存储所有结果
    all_results = []
    
    # 测试每个epoch的adapter
    for epoch in epochs:
        try:
            result = test_single_adapter(epoch, model_path, data_loader, test_study, output_dir)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error testing epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存汇总结果
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = output_dir / f"test_summary_{timestamp}.txt"
        summary_json = output_dir / f"test_summary_{timestamp}.json"
        
        # 文本汇总
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Adapter Test Summary - All Epochs\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sample Subject ID: {test_study.get('subject_id', 'unknown')}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("CE Loss Comparison:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Epoch':<10} {'CE Loss':<15} {'Valid Labels':<20}\n")
            f.write("-"*80 + "\n")
            
            for result in all_results:
                epoch = result['epoch']
                ce_loss = result['metrics']['ce_loss']
                valid_labels = result['metrics']['valid_labels']
                total_len = result['metrics']['total_input_length']
                f.write(f"{epoch:<10} {ce_loss:<15.4f} {valid_labels}/{total_len}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Detailed Results:\n")
            f.write("="*80 + "\n\n")
            
            for result in all_results:
                epoch = result['epoch']
                f.write(f"\nEpoch {epoch}:\n")
                f.write("-"*80 + "\n")
                f.write(f"CE Loss: {result['metrics']['ce_loss']:.4f}\n")
                f.write(f"Generated Report: {result['generated_report'][:300]}...\n")
                f.write(f"Ground Truth: {result['ground_truth_report'][:300]}...\n\n")
        
        # JSON汇总
        summary_data = {
            "timestamp": timestamp,
            "sample": {
                "subject_id": test_study.get('subject_id', 'unknown')
            },
            "results": all_results,
            "comparison": {
                "epochs": [r['epoch'] for r in all_results],
                "ce_losses": [r['metrics']['ce_loss'] for r in all_results],
                "best_epoch": min(all_results, key=lambda x: x['metrics']['ce_loss'])['epoch'] if all_results else None,
                "best_ce_loss": min([r['metrics']['ce_loss'] for r in all_results]) if all_results else None
            }
        }
        
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        # 打印汇总
        print("\n" + "="*80)
        print("Test Summary - All Epochs")
        print("="*80)
        print(f"{'Epoch':<10} {'CE Loss':<15} {'Valid Labels':<20}")
        print("-"*80)
        
        for result in all_results:
            epoch = result['epoch']
            ce_loss = result['metrics']['ce_loss']
            valid_labels = result['metrics']['valid_labels']
            total_len = result['metrics']['total_input_length']
            print(f"{epoch:<10} {ce_loss:<15.4f} {valid_labels}/{total_len}")
        
        if all_results:
            best_result = min(all_results, key=lambda x: x['metrics']['ce_loss'])
            print(f"\nBest Epoch: {best_result['epoch']} (CE Loss: {best_result['metrics']['ce_loss']:.4f})")
        
        print(f"\nSummary saved to:")
        print(f"  Text: {summary_file}")
        print(f"  JSON: {summary_json}")


if __name__ == "__main__":
    main()

