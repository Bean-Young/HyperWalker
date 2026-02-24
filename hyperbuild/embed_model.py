"""
Embedding Model for Hypergraph Building
直接使用MedGemma生成embedding
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


class EmbedModel:
    """
    Embedding模型
    直接使用MedGemma生成image和ehr的embedding
    """
    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: MedGemma模型路径
        """
        # 默认使用medgemma路径
        if model_path is None:
            model_path = "/mnt/sda/VLM/code/model_cache/models--google--medgemma-4b-it/snapshots/290cda5eeccbee130f987c4ad74a59ae6f196408"
        
        self.model_path = model_path
        
        # 加载MedGemma模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def image_embed(self, image_path: str):
        """
        生成图像的embedding（未pooling）
        Args:
            image_path: 图像路径
        Returns:
            image_embeds: (1, seq_len, embed_dim)
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}
                ]
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            image_embeds = self.model.vision_tower(inputs["pixel_values"]).last_hidden_state
            image_embeds = self.model.multi_modal_projector(image_embeds)
        return image_embeds

    def ehr_embed(self, ehr: str):
        """
        生成EHR文本的embedding（未pooling）
        Args:
            ehr: EHR文本
        Returns:
            text_embeds: (1, seq_len, embed_dim)
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ehr}
                ]
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            text_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
        return text_embeds

    def embed_pooling(self, embeds, dim=1):
        """
        对embedding进行池化
        Args:
            embeds: embedding tensor
            dim: 池化维度
        Returns:
            pooled_embeds: 池化后的embedding
        """
        return embeds.mean(dim=dim)

    def batch_ehr_embed(self, ehr_texts: list):
        """
        批量生成EHR文本的embedding（未pooling）
        Args:
            ehr_texts: EHR文本列表
        Returns:
            text_embeds_list: embedding列表，每个为 (1, seq_len, embed_dim)
        """
        if not ehr_texts:
            return []
        
        # 为每个文本创建messages
        all_inputs = []
        for ehr in ehr_texts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ehr}
                    ]
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            )
            all_inputs.append(inputs)
        
        # 批量处理（需要padding）
        max_length = max(inp["input_ids"].shape[1] for inp in all_inputs)
        
        batch_input_ids = []
        batch_attention_mask = []
        
        for inputs in all_inputs:
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            padding_length = max_length - seq_len
            
            # Padding
            padded_ids = torch.cat([
                input_ids,
                torch.zeros((1, padding_length), dtype=input_ids.dtype)
            ], dim=1)
            attention_mask = torch.cat([
                torch.ones((1, seq_len), dtype=torch.bool),
                torch.zeros((1, padding_length), dtype=torch.bool)
            ], dim=1)
            
            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(attention_mask)
        
        # 合并为batch
        # input_ids必须是整数类型（Long），不能是bfloat16
        batch_input_ids = torch.cat(batch_input_ids, dim=0).to(self.model.device).long()
        batch_attention_mask = torch.cat(batch_attention_mask, dim=0).to(self.model.device)
        
        # 批量生成embedding
        with torch.no_grad():
            batch_embeds = self.model.get_input_embeddings()(batch_input_ids)  # (batch_size, seq_len, embed_dim)
            # embedding输出可以转换为bfloat16以节省显存
            batch_embeds = batch_embeds.to(torch.bfloat16)
        
        # 分割回单个embedding，并应用attention mask
        text_embeds_list = []
        for i, inputs in enumerate(all_inputs):
            seq_len = inputs["input_ids"].shape[1]
            # 只取有效部分
            embeds = batch_embeds[i:i+1, :seq_len, :]  # (1, seq_len, embed_dim)
            text_embeds_list.append(embeds)
        
        return text_embeds_list
    
    def batch_image_embed(self, image_paths: list):
        """
        批量生成图像的embedding（未pooling）
        Args:
            image_paths: 图像路径列表
        Returns:
            image_embeds_list: embedding列表，每个为 (1, seq_len, embed_dim)
        """
        if not image_paths:
            return []
        
        # 为每个图像创建messages
        all_inputs = []
        for image_path in image_paths:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path}
                    ]
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            )
            all_inputs.append(inputs)
        
        # 批量处理pixel_values（图像需要特殊处理，因为每个图像的尺寸可能不同）
        # 对于图像，processor通常会自动处理batch，但我们需要手动处理
        batch_pixel_values = []
        batch_input_ids = []
        
        for inputs in all_inputs:
            pv = inputs["pixel_values"]
            # 确保pixel_values是4维的 (channels, height, width) 或 (1, channels, height, width)
            # 如果是5维，说明processor已经做了batch处理，需要squeeze
            if pv.dim() == 5:
                # (1, num_images, channels, height, width) -> (num_images, channels, height, width)
                pv = pv.squeeze(0)
            elif pv.dim() == 4 and pv.shape[0] == 1:
                # (1, channels, height, width) -> (channels, height, width)
                pv = pv.squeeze(0)
            batch_pixel_values.append(pv)
            batch_input_ids.append(inputs["input_ids"])
        
        # 合并pixel_values（processor应该已经处理了batch）
        # 如果尺寸不同，需要padding
        if len(batch_pixel_values) == 1:
            pixel_values = batch_pixel_values[0].unsqueeze(0).to(self.model.device, dtype=torch.bfloat16)
        else:
            # 检查所有pixel_values的尺寸是否相同
            first_shape = batch_pixel_values[0].shape
            all_same_size = all(pv.shape == first_shape for pv in batch_pixel_values)
            
            if all_same_size:
                # 如果尺寸相同，直接stack
                pixel_values = torch.stack(batch_pixel_values, dim=0).to(self.model.device, dtype=torch.bfloat16)
            else:
                # 如果尺寸不同，需要padding到最大尺寸
                max_h, max_w = 0, 0
                for pv in batch_pixel_values:
                    if pv.dim() == 3:
                        _, h, w = pv.shape
                    else:
                        h, w = pv.shape[-2:]
                    max_h = max(max_h, h)
                    max_w = max(max_w, w)
                
                padded_pixel_values = []
                for pv in batch_pixel_values:
                    if pv.dim() == 3:
                        _, h, w = pv.shape
                        if h < max_h or w < max_w:
                            pad_h = max_h - h
                            pad_w = max_w - w
                            pv = torch.nn.functional.pad(pv, (0, pad_w, 0, pad_h))
                    padded_pixel_values.append(pv)
                pixel_values = torch.stack(padded_pixel_values, dim=0).to(self.model.device, dtype=torch.bfloat16)
        
        # 批量生成embedding
        with torch.no_grad():
            batch_image_embeds = self.model.vision_tower(pixel_values).last_hidden_state  # (batch_size, seq_len, embed_dim)
            batch_image_embeds = self.model.multi_modal_projector(batch_image_embeds)  # (batch_size, seq_len, embed_dim)
        
        # 分割回单个embedding
        image_embeds_list = []
        for i in range(len(image_paths)):
            # 获取原始序列长度（从vision_tower的输出）
            embeds = batch_image_embeds[i:i+1, :, :]  # (1, seq_len, embed_dim)
            image_embeds_list.append(embeds)
        
        return image_embeds_list

