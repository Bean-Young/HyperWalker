"""
Image-EHR Embedding Model
图像和EHR的embedding生成模型
使用MedGemma生成embedding
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor


class ImageEHREmbed:
    """
    Image-EHR Embedding模型
    用于生成图像和EHR的embedding
    """
    def __init__(self, model_path: str = None) -> None:
        """
        Args:
            model_path: VLM模型路径（MedGemma）
        """
        if model_path is None:
            model_path = "/mnt/sda/VLM/code/model_cache/models--google--medgemma-4b-it/snapshots/290cda5eeccbee130f987c4ad74a59ae6f196408"
        
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

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

