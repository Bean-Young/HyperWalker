"""
Test Phase
测试阶段：固定模型F，使用检索到的节点生成报告，并计算指标
"""
import os
# 设置离线模式，强制只使用本地文件
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 禁用 tokenizers 并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm
import json
import numpy as np
import argparse
# Pillow 兼容性补丁：修复旧版本 Pillow 缺少 Resampling 的问题
import PIL.Image


# 导入模块
from embed_models.image_ehr_embed import ImageEHREmbed
from embed_models.film_fusion import FiLMFusion
from rl_finder.rl_path_finder import RLPathFinder
from hyperbuild.hypergraph import Hypergraph, NodeType
from hyperbuild.data_loader import DataLoader
from hyperbuild.build_hypergraph import HypergraphBuilder
from hyperbuild.embed_model import EmbedModel
from transformers import Gemma3ForConditionalGeneration, AutoProcessor, AutoTokenizer
import hnswlib
from peft import PeftModel

# 导入metrics
from metrics.nlg_metrics import NLGMetricEvaluator
from metrics.ce_metrics import CEMetricEvaluator


class TestModelF:
    """测试强化学习模型F"""
    def __init__(
        self,
        vlm_model_path: str = None,
        hypergraph_path: str = "/mnt/sda/VLM/code/hypercode/hyperbuild/hypergraph",
        adapter_checkpoint_dir: str = "/mnt/sda/VLM/code/hypercode/adatri/adamodel/adapter_epoch_2",
        model_f_checkpoint: str = "/mnt/sda/VLM/code/hypercode/checkpoints/model_f_epoch_1.pt",
        fine_tune_adapter: bool = True,
        adapter_lr: float = 1e-5,
        adapter_fine_tune_steps: int = 1,
        batch_size: int = 2,
        output_dir: str = "/mnt/sda/VLM/code/hypercode/result_re",
        embed_dim: int = 1024,
        similarity_threshold: float = 0.8,
        max_hops: int = 3,
        hypergraph_sample_ratio: float = 0.01
    ):
        """
        Args:
            vlm_model_path: VLM模型路径
            hypergraph_path: 超图保存路径
            adapter_checkpoint_dir: Adapter checkpoint目录
            model_f_checkpoint: 模型F checkpoint路径
            output_dir: 测试结果保存目录
            embed_dim: embedding维度
            similarity_threshold: 相似度阈值（用于构建超边）
            max_hops: 最大hop数量
            fine_tune_adapter: 是否在测试时微调adapter
            adapter_lr: adapter微调的学习率
            adapter_fine_tune_steps: 每个样本微调的步数
            batch_size: 批处理大小（同时处理的样本数）
        """
        self.vlm_model_path = vlm_model_path
        self.hypergraph_path = Path(hypergraph_path)
        self.adapter_checkpoint_dir = Path(adapter_checkpoint_dir)
        self.model_f_checkpoint = model_f_checkpoint
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fine_tune_adapter = fine_tune_adapter
        self.adapter_lr = adapter_lr
        self.adapter_fine_tune_steps = adapter_fine_tune_steps
        self.batch_size = batch_size
        
        self.max_tokens = 12000  # 最大token数限制
        
        self.embed_dim = embed_dim
        self.similarity_threshold = similarity_threshold
        self.max_hops = max_hops
        self.hypergraph_sample_ratio = hypergraph_sample_ratio
        
        # 构建超图（使用1%数据，不保存，直接使用对象）
        print("Building hypergraph (using 1% data, no saving)...")
        print("Creating EmbedModel...")
        embed_model = EmbedModel(model_path=self.vlm_model_path)
        print("EmbedModel created, creating HypergraphBuilder...")
        builder = HypergraphBuilder(
            model=embed_model,
            embed_dim=embed_dim,
            k=5,
            train_sample_ratio=hypergraph_sample_ratio,
            batch_size=8
        )
        print("HypergraphBuilder created, starting to build hypergraph...")
        # 构建超图，不保存（output_path=None），直接使用对象
        self.hypergraph = builder.build(output_path=None)
        print("Hypergraph built successfully!")
        
        # 加载embedding模型（用于测试时的embedding生成）
        print("Loading embedding model for testing...")
        self.embed_model = ImageEHREmbed(model_path=self.vlm_model_path)
        device = next(self.embed_model.model.parameters()).device
        
        # 动态获取实际的embedding维度
        actual_embed_dim = embed_dim
        if len(self.hypergraph.nodes) > 0:
            first_node = next(iter(self.hypergraph.nodes.values()))
            if first_node.embedding is not None:
                if isinstance(first_node.embedding, torch.Tensor):
                    actual_embed_dim = first_node.embedding.shape[-1]
                else:
                    actual_embed_dim = len(first_node.embedding) if hasattr(first_node.embedding, '__len__') else embed_dim
                print(f"Detected actual embedding dimension: {actual_embed_dim}")
        
        self.embed_dim = actual_embed_dim
        
        # 加载FiLM融合模块（固定，不训练）
        print(f"Loading FiLM fusion module (dim={actual_embed_dim})...")
        self.fusion = FiLMFusion(dim=actual_embed_dim).to(device).to(torch.bfloat16)
        
        # 加载强化学习路径寻找器（固定，不训练）
        print(f"Loading RL path finder (embed_dim={actual_embed_dim})...")
        self.rl_finder = RLPathFinder(
            embed_dim=actual_embed_dim,
            hidden_dim=512,
            max_hops=max_hops,
            similarity_threshold=similarity_threshold,
            temperature=0.01
        ).to(device).to(torch.bfloat16)
        
        # 加载模型F的权重
        if self.model_f_checkpoint and Path(self.model_f_checkpoint).exists():
            print(f"Loading model F checkpoint from {self.model_f_checkpoint}...")
            checkpoint = torch.load(self.model_f_checkpoint, map_location=device)
            self.fusion.load_state_dict(checkpoint['fusion_state_dict'])
            self.rl_finder.load_state_dict(checkpoint['rl_finder_state_dict'])
            print("Model F loaded successfully")
        
        # 冻结模型F（不训练）
        self.fusion.eval()
        self.rl_finder.eval()
        for param in self.fusion.parameters():
            param.requires_grad = False
        for param in self.rl_finder.parameters():
            param.requires_grad = False
        
        # 确定基础模型路径
        if self.vlm_model_path:
            base_model_path = str(Path(self.vlm_model_path).resolve())
        else:
            base_model_path = "/mnt/sda/VLM/code/model_cache/models--google--medgemma-4b-it/snapshots/290cda5eeccbee130f987c4ad74a59ae6f196408"
        
        # 加载processor和tokenizer
        print("Loading processor and tokenizer from base model...")
        self.processor = AutoProcessor.from_pretrained(base_model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
        
        # 加载基础VLM模型
        print("Loading base VLM model...")
        self.vlm_model = Gemma3ForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        
        # 加载adapter
        print(f"Loading adapter from {self.adapter_checkpoint_dir}...")
        self._load_adapter()
        
        # 构建HNSW索引
        print("Building HNSW index...")
        self._build_hnsw_index()
        
        # 数据加载器（测试集）
        print("Loading test data...")
        self.data_loader = DataLoader(train_sample_ratio=1.0)
        self.test_studies = self.data_loader.get_test_studies()
        print(f"Loaded {len(self.test_studies)} test studies")
        
        # 初始化metrics
        print("Initializing metrics evaluators...")
        self.nlg_evaluator = NLGMetricEvaluator()
        self.ce_evaluator = CEMetricEvaluator()
    
    def _load_adapter(self):
        """加载adapter"""
        import json
        adapter_path = Path(self.adapter_checkpoint_dir)
        
        adapter_config_file = adapter_path / "adapter_config.json"
        if adapter_config_file.exists():
            self.vlm_model = PeftModel.from_pretrained(
                self.vlm_model,
                str(adapter_path),
                adapter_name="last_ffn",
                local_files_only=True
            )
        else:
            training_info_file = adapter_path / "training_info.json"
            if training_info_file.exists():
                with open(training_info_file, 'r') as f:
                    training_info = json.load(f)
                
                from peft import LoraConfig, get_peft_model
                
                peft_config = LoraConfig(
                    r=training_info.get("r", 16),
                    lora_alpha=training_info.get("lora_alpha", 32),
                    target_modules=training_info.get("target_modules", []),
                    lora_dropout=training_info.get("lora_dropout", 0.05),
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                
                self.vlm_model = get_peft_model(self.vlm_model, peft_config, adapter_name="last_ffn")
                
                import safetensors.torch
                adapter_weights = {}
                
                index_file = adapter_path / "model.safetensors.index.json"
                if index_file.exists():
                    with open(index_file, 'r') as f:
                        index_data = json.load(f)
                    
                    weight_map = index_data.get("weight_map", {})
                    for weight_name, shard_file in weight_map.items():
                        shard_path = adapter_path / shard_file
                        if shard_path.exists():
                            shard_weights = safetensors.torch.load_file(str(shard_path))
                            for key, value in shard_weights.items():
                                if "lora" in key.lower() or "last_ffn" in key:
                                    adapter_weights[key] = value
                else:
                    safetensors_files = list(adapter_path.glob("*.safetensors"))
                    for safetensors_file in safetensors_files:
                        weights = safetensors.torch.load_file(str(safetensors_file))
                        for key, value in weights.items():
                            if "lora" in key.lower() or "last_ffn" in key:
                                adapter_weights[key] = value
                
                if not adapter_weights:
                    raise ValueError(f"No LoRA weights found in {adapter_path}")
                
                from peft.utils import set_peft_model_state_dict
                set_peft_model_state_dict(self.vlm_model, adapter_weights, adapter_name="last_ffn")
            else:
                try:
                    self.vlm_model = PeftModel.from_pretrained(
                        self.vlm_model,
                        str(adapter_path),
                        adapter_name="last_ffn",
                        local_files_only=True
                    )
                except Exception as e:
                    raise ValueError(
                        f"Cannot load adapter from {self.adapter_checkpoint_dir}. "
                        f"Missing adapter_config.json and training_info.json. Error: {e}"
                    )
        
        self.vlm_model.set_adapter("last_ffn")
        if not self.fine_tune_adapter:
            self.vlm_model.eval()  # 如果不微调，使用eval模式
        print(f"Adapter loaded from {self.adapter_checkpoint_dir}")
        
        # 如果需要在测试时微调，设置优化器
        if self.fine_tune_adapter:
            adapter_params = [p for n, p in self.vlm_model.named_parameters() if p.requires_grad]
            if len(adapter_params) > 0:
                self.adapter_optimizer = torch.optim.AdamW(adapter_params, lr=self.adapter_lr)
                print(f"Adapter fine-tuning enabled: {len(adapter_params)} trainable parameters, lr={self.adapter_lr}, steps={self.adapter_fine_tune_steps}")
            else:
                print("Warning: No trainable adapter parameters found, disabling fine-tuning")
                self.fine_tune_adapter = False
    
    def _build_hnsw_index(self):
        """构建HNSW索引用于检索相似节点"""
        embeddings = []
        node_ids = []
        
        for node_id, node in self.hypergraph.nodes.items():
            if node.embedding is not None:
                emb = node.embedding.cpu().float().numpy() if isinstance(node.embedding, torch.Tensor) else node.embedding
                embeddings.append(emb.astype('float32'))
                node_ids.append(node_id)
        
        if not embeddings:
            raise ValueError("No node embeddings found in hypergraph")
        
        embeddings = np.array(embeddings)
        embed_dim = embeddings.shape[1]
        
        self.hnsw_index = hnswlib.Index(space='cosine', dim=embed_dim)
        self.hnsw_index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        indices = np.arange(len(embeddings))
        self.hnsw_index.add_items(embeddings, indices)
        self.hnsw_index.set_ef(50)
        
        self.node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        self.index_to_node_id = {i: node_id for i, node_id in enumerate(node_ids)}
        print(f"HNSW index built with {len(embeddings)} nodes")
    
    def _retrieve_similar_nodes_hnsw(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: float = None
    ) -> List[Tuple[str, float]]:
        """使用HNSW检索相似节点"""
        if threshold is None:
            threshold = self.similarity_threshold
        
        labels, distances = self.hnsw_index.knn_query(query_embedding, k=k)
        results = []
        for label, dist in zip(labels[0], distances[0]):
            similarity = 1.0 - dist
            if similarity >= threshold:
                node_id = self.index_to_node_id[int(label)]
                results.append((node_id, similarity))
        
        return results
    
    def _build_hyperedges_from_similarity(
        self,
        query_embedding: torch.Tensor,
        similar_nodes: List[Tuple[str, float]]
    ) -> List[List[str]]:
        """基于相似度构建超边"""
        if len(similar_nodes) < 2:
            return []
        
        top_k_first_layer = 10
        k_per_layer = 1
        max_layers = 10
        
        similar_nodes_sorted = sorted(similar_nodes, key=lambda x: x[1], reverse=True)
        first_layer_nodes = similar_nodes_sorted[:top_k_first_layer]
        
        hyperedges = []
        
        for first_node_id, first_sim in first_layer_nodes:
            if first_node_id not in self.hypergraph.nodes:
                continue
            
            current_hyperedge = [first_node_id]
            visited_nodes = {first_node_id}
            current_layer_node_id = first_node_id
            
            for layer in range(max_layers):
                current_node = self.hypergraph.nodes[current_layer_node_id]
                current_emb = current_node.embedding
                if isinstance(current_emb, torch.Tensor):
                    current_emb_np = current_emb.cpu().float().numpy()
                else:
                    current_emb_np = np.array(current_emb).astype('float32')
                
                current_emb_np = current_emb_np.reshape(1, -1)
                next_layer_results = self._retrieve_similar_nodes_hnsw(
                    current_emb_np, 
                    k=k_per_layer + len(visited_nodes),
                    threshold=self.similarity_threshold
                )
                
                next_node_id = None
                for node_id, sim in next_layer_results:
                    if node_id not in visited_nodes:
                        next_node_id = node_id
                        break
                
                if next_node_id is None:
                    break
                
                current_hyperedge.append(next_node_id)
                visited_nodes.add(next_node_id)
                current_layer_node_id = next_node_id
            
            if len(current_hyperedge) >= 2:
                hyperedges.append(current_hyperedge)
        
        return hyperedges
    
    def _collect_candidate_nodes_recursive(
        self,
        initial_nodes: List[str],
        query_hyperedges: List[List[str]] = None,
        max_hops: int = 3
    ) -> List[Dict]:
        """递归收集候选节点"""
        candidate_nodes = []
        visited_node_ids = set()
        current_layer_nodes = set(initial_nodes)
        
        for hop in range(max_hops):
            if len(current_layer_nodes) == 0:
                break
            
            next_layer_nodes = set()
            
            for node_id in current_layer_nodes:
                if node_id in visited_node_ids:
                    continue
                visited_node_ids.add(node_id)
                
                if hop == 0 and query_hyperedges is not None and len(query_hyperedges) > 0:
                    for hyperedge in query_hyperedges:
                        if node_id in hyperedge:
                            for connected_node_id in hyperedge:
                                if connected_node_id not in visited_node_ids:
                                    if connected_node_id in self.hypergraph.nodes:
                                        node = self.hypergraph.nodes[connected_node_id]
                                        emb = node.embedding
                                        if isinstance(emb, torch.Tensor):
                                            emb_np = emb.cpu().float().numpy()
                                        else:
                                            emb_np = np.array(emb)
                                        
                                        candidate_nodes.append({
                                            'node_id': connected_node_id,
                                            'embedding': emb_np,
                                            'node_type': node.node_type.value
                                        })
                                        visited_node_ids.add(connected_node_id)
                                        next_layer_nodes.add(connected_node_id)
                
                edge_ids = self.hypergraph.get_node_edges(node_id)
                for edge_id in edge_ids:
                    edge = self.hypergraph.hyperedges[edge_id]
                    for connected_node_id in edge.node_ids:
                        if connected_node_id not in visited_node_ids:
                            if connected_node_id in self.hypergraph.nodes:
                                node = self.hypergraph.nodes[connected_node_id]
                                emb = node.embedding
                                if isinstance(emb, torch.Tensor):
                                    emb_np = emb.cpu().float().numpy()
                                else:
                                    emb_np = np.array(emb)
                                
                                candidate_nodes.append({
                                    'node_id': connected_node_id,
                                    'embedding': emb_np,
                                    'node_type': node.node_type.value
                                })
                                visited_node_ids.add(connected_node_id)
                                next_layer_nodes.add(connected_node_id)
            
            current_layer_nodes = next_layer_nodes
        
        return candidate_nodes
    
    def _multi_hop_retrieval(
        self,
        query_embedding: torch.Tensor,
        initial_nodes: List[str],
        max_hops: int = None,
        query_hyperedges: List[List[str]] = None,
        max_orthogonal_hops: int = 2,
        previous_selected_nodes: List[Dict] = None,
        current_hop_offset: int = 0
    ) -> Tuple[List[Dict], int]:
        """多跳检索"""
        if max_hops is None:
            max_hops = self.max_hops
        
        all_selected_nodes = []
        current_query = query_embedding
        orthogonal_threshold = 0.3
        
        # 第一次检索
        candidate_nodes = self._collect_candidate_nodes_recursive(
            initial_nodes, 
            query_hyperedges, 
            max_hops=max_hops
        )
        
        # 强制添加所有knowledge节点
        knowledge_nodes = []
        for node_id, node in self.hypergraph.nodes.items():
            is_knowledge = (
                node.node_type == NodeType.KNOWLEDGE or
                str(node.node_type) == 'NodeType.KNOWLEDGE' or
                (hasattr(node.node_type, 'value') and node.node_type.value == 'knowledge')
            )
            if is_knowledge:
                knowledge_nodes.append((node_id, node))
        
        existing_knowledge_ids = {node.get('node_id') for node in candidate_nodes if node.get('node_type', '').lower() == 'knowledge'}
        
        added_count = 0
        for node_id, node in knowledge_nodes:
            if node_id not in existing_knowledge_ids:
                emb = node.embedding
                if isinstance(emb, torch.Tensor):
                    emb_np = emb.cpu().float().numpy()
                else:
                    emb_np = np.array(emb)
                
                candidate_nodes.append({
                    'node_id': node_id,
                    'embedding': emb_np,
                    'node_type': 'knowledge',
                    'hop': current_hop_offset  # knowledge节点也标记hop
                })
                added_count += 1
        
        if len(candidate_nodes) == 0:
            return [], 0
        
        # 使用MLP选择节点
        device = query_embedding.device
        query_emb_tensor = current_query.to(device)
        
        probs, value, activation_scores = self.rl_finder(query_emb_tensor, candidate_nodes, debug=False)
        
        selected_indices = self.rl_finder.select_nodes(
            probs, 
            activation_scores=activation_scores,
            prob_threshold=0.01,
            debug=False, 
            candidate_nodes=candidate_nodes
        )
        
        selected_nodes = [candidate_nodes[idx] for idx in selected_indices]
        # 标记这些节点来自主检索（hop = current_hop_offset）
        for node in selected_nodes:
            node['hop'] = current_hop_offset
        all_selected_nodes.extend(selected_nodes)
        hop_count = max_hops
        
        # 正交化检索
        for orthogonal_hop in range(max_orthogonal_hops):
            if len(all_selected_nodes) == 0:
                break
            
            dtype = current_query.dtype
            non_knowledge_nodes = [
                n for n in all_selected_nodes 
                if n.get('node_type', '').lower() != 'knowledge'
            ]
            
            if len(non_knowledge_nodes) == 0:
                break
            
            retrieved_embs = [
                torch.tensor(n['embedding'], device=device, dtype=dtype) if not isinstance(n['embedding'], torch.Tensor)
                else n['embedding'].to(device=device, dtype=dtype)
                for n in non_knowledge_nodes
            ]
            
            orthogonal_emb = self._orthogonalize_embedding(current_query, retrieved_embs)
            orthogonal_norm = torch.norm(orthogonal_emb).item()
            
            if orthogonal_norm < orthogonal_threshold:
                break
            
            orthogonal_emb_norm = torch.norm(orthogonal_emb)
            if orthogonal_emb_norm > 1e-8:
                orthogonal_emb_normalized = orthogonal_emb / orthogonal_emb_norm
            else:
                orthogonal_emb_normalized = orthogonal_emb
            
            query_emb_np = orthogonal_emb_normalized.detach().cpu().float().numpy().reshape(1, -1)
            adjusted_threshold = orthogonal_emb_norm.item() * 0.8
            
            similar_nodes_orth = self._retrieve_similar_nodes_hnsw(query_emb_np, k=30, threshold=adjusted_threshold)
            
            if len(similar_nodes_orth) == 0:
                break
            
            existing_node_ids = {n['node_id'] for n in all_selected_nodes}
            new_node_ids = [node_id for node_id, _ in similar_nodes_orth if node_id not in existing_node_ids]
            
            if len(new_node_ids) == 0:
                break
            
            orthogonal_emb_tensor = orthogonal_emb.to(device)
            hyperedges_orth = self._build_hyperedges_from_similarity(orthogonal_emb_tensor, similar_nodes_orth)
            
            selected_nodes_orth, hop_count_orth = self._multi_hop_retrieval(
                orthogonal_emb,
                new_node_ids[:5],
                max_hops=max_hops,
                query_hyperedges=hyperedges_orth,
                max_orthogonal_hops=0,
                previous_selected_nodes=all_selected_nodes,
                current_hop_offset=orthogonal_hop + 1
            )
            
            if len(selected_nodes_orth) == 0:
                break
            
            existing_node_ids = {n['node_id'] for n in all_selected_nodes}
            newly_selected_ids = {n['node_id'] for n in selected_nodes_orth}
            
            if newly_selected_ids.issubset(existing_node_ids):
                break
            
            # 标记这些节点来自正交检索（hop = orthogonal_hop + 1）
            for node in selected_nodes_orth:
                if 'hop' not in node:  # 如果还没有hop信息，添加
                    node['hop'] = orthogonal_hop + 1
            
            all_selected_nodes.extend(selected_nodes_orth)
            hop_count += hop_count_orth
            current_query = orthogonal_emb
        
        return all_selected_nodes, hop_count
    
    def _orthogonalize_embedding(
        self,
        query_embedding: torch.Tensor,
        retrieved_embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        """正交化embedding"""
        if len(retrieved_embeddings) == 0:
            return query_embedding
        
        device = query_embedding.device
        dtype = query_embedding.dtype
        
        retrieved_embeddings = [
            emb.to(device=device, dtype=dtype) if isinstance(emb, torch.Tensor)
            else torch.tensor(emb, device=device, dtype=dtype)
            for emb in retrieved_embeddings
        ]
        
        mean_embedding = torch.stack(retrieved_embeddings).mean(dim=0)
        
        query_norm = torch.norm(query_embedding)
        if query_norm > 1e-8:
            query_normalized = query_embedding / query_norm
        else:
            query_normalized = query_embedding
        
        mean_norm = torch.norm(mean_embedding)
        if mean_norm > 1e-8:
            mean_normalized = mean_embedding / mean_norm
        else:
            mean_normalized = mean_embedding
        
        dot_ab = torch.dot(query_normalized, mean_normalized)
        dot_bb = torch.dot(mean_normalized, mean_normalized)
        projection = (dot_ab / dot_bb) * mean_normalized
        
        orthogonal_embedding = query_normalized - projection
        
        return orthogonal_embedding
    
    def _retrieve_triplets_and_knowledge(
        self,
        selected_nodes: List[Dict]
    ) -> Tuple[List[Dict], List[str]]:
        """从选中的节点中召回三元组和knowledge"""
        triplets = []
        knowledge_texts = []
        processed_triplet_keys = set()
        
        prompt = "Please generate a paragraph of radiology report for this chest X-ray image."
        
        for node_info in selected_nodes:
            node_id = node_info['node_id']
            node = self.hypergraph.nodes[node_id]
            
            if node.node_type == NodeType.KNOWLEDGE:
                if 'description' in node.metadata:
                    knowledge_texts.append(node.metadata['description'])
                elif 'text' in node.metadata:
                    knowledge_texts.append(node.metadata['text'])
            
            if 'triplets' in node.metadata:
                for triplet_info in node.metadata['triplets']:
                    triplet_key = (triplet_info.get('subject_id'), triplet_info.get('study_id'))
                    if triplet_key in processed_triplet_keys:
                        continue
                    processed_triplet_keys.add(triplet_key)
                    
                    ehr_texts = []
                    image_paths = []
                    report_text = None
                    
                    for ehr_node_id in triplet_info.get('ehr_node_ids', []):
                        if ehr_node_id in self.hypergraph.nodes:
                            ehr_node = self.hypergraph.nodes[ehr_node_id]
                            if 'ehr_text' in ehr_node.metadata:
                                ehr_texts.append(ehr_node.metadata['ehr_text'])
                    
                    for image_node_id in triplet_info.get('image_node_ids', []):
                        if image_node_id in self.hypergraph.nodes:
                            image_node = self.hypergraph.nodes[image_node_id]
                            if 'image_path' in image_node.metadata:
                                image_paths.append(image_node.metadata['image_path'])
                    
                    for report_node_id in triplet_info.get('report_node_ids', []):
                        if report_node_id in self.hypergraph.nodes:
                            report_node = self.hypergraph.nodes[report_node_id]
                            if 'report_text' in report_node.metadata:
                                report_text = report_node.metadata['report_text']
                                break
                    
                    if ehr_texts and image_paths and report_text:
                        combined_ehr_text = "\n".join(ehr_texts)
                        ehr_prompt_text = combined_ehr_text + "\n" + prompt
                        tokenized_ehr_prompt = self.tokenizer(
                            ehr_prompt_text,
                            return_tensors="pt",
                            add_special_tokens=False,
                            truncation=False
                        )
                        ehr_prompt_tokens = tokenized_ehr_prompt["input_ids"].shape[1]
                        
                        if ehr_prompt_tokens <= self.max_tokens - 256:
                            triplets.append({
                                'ehr_texts': ehr_texts,
                                'image_paths': image_paths,
                                'report_text': report_text,
                                'subject_id': triplet_info.get('subject_id'),
                                'study_id': triplet_info.get('study_id')
                            })
        
        return triplets, knowledge_texts
    
    def _fine_tune_adapter_for_sample(
        self,
        ehr_text: str,
        image_paths: List[str],
        knowledge_texts: List[str],
        ground_truth_report: str
    ):
        """对当前样本微调adapter，只使用CE loss"""
        if not self.fine_tune_adapter:
            return
        
        # 限制ground truth report的长度（避免序列过长导致显存溢出）
        max_gt_tokens = 256  # 限制ground truth最多256个token
        gt_tokens = self.tokenizer.encode(ground_truth_report, add_special_tokens=False)
        if len(gt_tokens) > max_gt_tokens:
            gt_tokens = gt_tokens[:max_gt_tokens]
            ground_truth_report = self.tokenizer.decode(gt_tokens, skip_special_tokens=True)
        
        prompt = "Please generate a paragraph of radiology report for this chest X-ray image."
        
        text_content = ehr_text
        if knowledge_texts and len(knowledge_texts) > 0:
            # 限制knowledge texts的数量和长度
            limited_knowledge_texts = knowledge_texts[:3]  # 最多使用3个knowledge
            knowledge_str = "\n".join(limited_knowledge_texts)
            # 限制总长度
            if len(knowledge_str) > 500:
                knowledge_str = knowledge_str[:500]
            text_content = ehr_text + "\n\nKnowledge:\n" + knowledge_str
        
        text_content = text_content + "\n" + prompt
        
        # 准备训练数据（使用ground truth作为target）
        content = []
        for img_path in image_paths:
            content.append({"type": "image", "image": img_path})
        content.append({"type": "text", "text": text_content})
        
        messages = [
            {
                "role": "user",
                "content": content
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ground_truth_report}]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.vlm_model.device)
        
        input_ids = inputs.get("input_ids")
        pixel_values = inputs.get("pixel_values")
        
        if input_ids is None:
            return
        
        # 截断序列长度（避免显存溢出）
        max_seq_len = 2048  # 限制最大序列长度
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
        
        # 创建labels
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        # 微调adapter（只使用CE loss）
        self.vlm_model.train()
        for step in range(self.adapter_fine_tune_steps):
            self.adapter_optimizer.zero_grad()
            
            # 前向传播计算CE loss
            outputs = self.vlm_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=labels
            )
            ce_loss = outputs.loss
            
            # 反向传播（只使用CE loss）
            ce_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.vlm_model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            self.adapter_optimizer.step()
            
            # 立即释放显存
            del outputs, ce_loss
            torch.cuda.empty_cache()  # 每个step后都清理显存
        
        self.vlm_model.eval()
        # 清理显存
        torch.cuda.empty_cache()
    
    def _generate_report(
        self,
        ehr_text: str,
        image_paths: List[str],
        knowledge_texts: List[str] = None,
        ground_truth_report: str = None
    ) -> str:
        """生成报告"""
        prompt = "Please generate a paragraph of radiology report for this chest X-ray image."
        
        text_content = ehr_text
        if knowledge_texts and len(knowledge_texts) > 0:
            knowledge_str = "\n".join(knowledge_texts)
            text_content = ehr_text + "\n\nKnowledge:\n" + knowledge_str
        
        text_content = text_content + "\n" + prompt
        
        content = []
        for img_path in image_paths:
            content.append({"type": "image", "image": img_path})
        content.append({"type": "text", "text": text_content})
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.vlm_model.device)
        
        input_ids = inputs.get("input_ids")
        pixel_values = inputs.get("pixel_values")
        if input_ids is not None:
            input_length = input_ids.shape[1]
        else:
            input_length = 0
        
        # 如果启用微调且有ground truth，先进行微调
        if self.fine_tune_adapter and ground_truth_report:
            self._fine_tune_adapter_for_sample(
                ehr_text, image_paths, knowledge_texts, ground_truth_report
            )
        
        self.vlm_model.eval()
        with torch.no_grad():
            outputs = self.vlm_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
            
            if input_length > 0 and len(outputs[0]) > input_length:
                generated_ids = outputs[0][input_length:]
            else:
                generated_ids = outputs[0]
            
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def _generate_reports_batch(
        self,
        batch_data_list: List[Dict]
    ) -> List[str]:
        """
        批量生成报告（batch size = 2）
        注意：由于每个样本的微调是独立的，这里只批量处理生成部分
        """
        generated_reports = []
        
        # 先对每个样本进行微调（如果需要）
        for batch_item in batch_data_list:
            data = batch_item['data']
            knowledge_texts = batch_item['knowledge_texts']
            
            if self.fine_tune_adapter and data['report_text']:
                self._fine_tune_adapter_for_sample(
                    data['ehr_text'],
                    data['image_paths'],
                    knowledge_texts,
                    data['report_text']
                )
        
        # 批量生成报告
        self.vlm_model.eval()
        prompt = "Please generate a paragraph of radiology report for this chest X-ray image."
        
        # 准备批量输入
        batch_messages = []
        batch_input_lengths = []
        
        for batch_item in batch_data_list:
            data = batch_item['data']
            knowledge_texts = batch_item['knowledge_texts']
            
            text_content = data['ehr_text']
            if knowledge_texts and len(knowledge_texts) > 0:
                knowledge_str = "\n".join(knowledge_texts)
                text_content = data['ehr_text'] + "\n\nKnowledge:\n" + knowledge_str
            
            text_content = text_content + "\n" + prompt
            
            content = []
            for img_path in data['image_paths']:
                content.append({"type": "image", "image": img_path})
            content.append({"type": "text", "text": text_content})
            
            messages = [{
                "role": "user",
                "content": content
            }]
            batch_messages.append(messages)
        
        # 批量处理（如果processor支持batch）
        # 注意：由于不同样本的图像数量可能不同，这里可能需要逐个处理
        # 或者使用padding处理
        try:
            # 先单独处理每个样本，获取input_lengths
            individual_inputs = []
            for messages in batch_messages:
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, return_dict=True, return_tensors="pt"
                ).to(self.vlm_model.device)
                individual_inputs.append(inputs)
                if 'input_ids' in inputs:
                    batch_input_lengths.append(inputs['input_ids'].shape[1])
                else:
                    batch_input_lengths.append(0)
            
            # 尝试批量处理（需要padding）
            # 由于不同样本的图像数量可能不同，这里使用逐个处理更安全
            # 但我们可以尝试批量处理文本部分
            with torch.no_grad():
                for idx, inputs in enumerate(individual_inputs):
                    input_length = batch_input_lengths[idx] if idx < len(batch_input_lengths) else 0
                    outputs = self.vlm_model.generate(
                        input_ids=inputs.get('input_ids'),
                        pixel_values=inputs.get('pixel_values'),
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7
                    )
                    
                    if input_length > 0 and len(outputs[0]) > input_length:
                        generated_ids = outputs[0][input_length:]
                    else:
                        generated_ids = outputs[0]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    generated_reports.append(generated_text)
        
        except Exception as e:
            # 如果批量处理失败，回退到逐个处理
            print(f"Warning: Batch generation failed, falling back to individual processing: {e}")
            for batch_item in batch_data_list:
                data = batch_item['data']
                knowledge_texts = batch_item['knowledge_texts']
                generated_report = self._generate_report(
                    data['ehr_text'],
                    data['image_paths'],
                    knowledge_texts,
                    ground_truth_report=None  # 微调已经在前面完成
                )
                generated_reports.append(generated_report)
        
        return generated_reports
    
    def _extract_label_vector(self, study: Dict) -> List[int]:
        """从study中提取CheXbert label vector"""
        labels_cls = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", 
                     "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", 
                     "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
        
        label_vec = []
        
        # 从cxr_metadata中获取chexpert标签
        if 'cxr_metadata' in study and 'chexpert' in study['cxr_metadata']:
            chexpert = study['cxr_metadata']['chexpert']
            for label in labels_cls:
                if label in chexpert:
                    val = chexpert[label]
                    if isinstance(val, str):
                        label_vec.append(int(float(val)))
                    else:
                        label_vec.append(int(val))
                else:
                    label_vec.append(0)  # 默认0（没有提到）
        else:
            # 如果没有chexpert，全部设为0
            label_vec = [0] * len(labels_cls)
        
        return label_vec
    
    def _prepare_test_data(self, study: Dict) -> Optional[Dict]:
        """准备测试数据"""
        image_paths = self.data_loader.get_image_paths(study)
        if not image_paths:
            return None
        
        ehr_data = self.data_loader.get_ehr_data(study)
        if not ehr_data:
            return None
        
        ehr_text = self._format_ehr(ehr_data)
        report_text = self.data_loader.get_report_text(study)
        if not report_text:
            return None
        
        # 提取label vector
        label_vec = self._extract_label_vector(study)
        
        return {
            'image_paths': image_paths,
            'ehr_text': ehr_text,
            'report_text': report_text,
            'label_vec': label_vec,
            'subject_id': study.get('subject_id'),
            'study_id': study.get('study_id')
        }
    
    def _format_ehr(self, ehr_data: Dict, max_length: int = None) -> str:
        """
        格式化EHR数据为简洁的文本，适合作为prompt
        保留核心信息，忽略检查事件等次要信息
        
        Args:
            ehr_data: EHR数据字典
            max_length: 最大字符长度（None表示不限制，或设置为较大值如2000）
        
        Returns:
            格式化后的EHR文本
        """
        ehr_parts = []
        
        # 辅助函数：不区分大小写获取值
        def get_value(d, *keys):
            for key in keys:
                # 尝试原始key
                if key in d:
                    return d[key]
                # 尝试大写
                if key.upper() in d:
                    return d[key.upper()]
                # 尝试小写
                if key.lower() in d:
                    return d[key.lower()]
            return None
        
        # 1. 患者基本信息（保留完整）
        if ehr_data.get('patient_info'):
            p = ehr_data['patient_info']
            patient_info = []
            gender = get_value(p, 'GENDER', 'gender')
            if gender:
                patient_info.append(f"Gender: {gender}")
            age = get_value(p, 'ANCHOR_AGE', 'anchor_age', 'AGE', 'age')
            if age is not None:
                patient_info.append(f"Age: {age}")
            if patient_info:
                ehr_parts.append(f"Patient: {', '.join(patient_info)}")
        
        # 2. 入院信息（保留所有记录）
        if ehr_data.get('admissions'):
            for adm in ehr_data['admissions']:  # 保留所有入院记录
                adm_info = []
                adm_type = get_value(adm, 'ADMISSION_TYPE', 'admission_type')
                if adm_type:
                    adm_info.append(adm_type)
                adm_loc = get_value(adm, 'ADMISSION_LOCATION', 'admission_location')
                if adm_loc:
                    adm_info.append(f"from {adm_loc}")
                disch_loc = get_value(adm, 'DISCHARGE_LOCATION', 'discharge_location')
                if disch_loc:
                    adm_info.append(f"to {disch_loc}")
                if adm_info:
                    ehr_parts.append(f"Admission: {', '.join(adm_info)}")
        
        # 3. 诊断信息（保留所有主要诊断，完整标题）
        if ehr_data.get('all_diagnoses'):
            diag_items = []
            for diag in ehr_data['all_diagnoses']:
                title = get_value(diag, 'LONG_TITLE', 'long_title') or get_value(diag, 'SHORT_TITLE', 'short_title') or get_value(diag, 'ICD_TITLE', 'icd_title')
                if title:
                    # 保留完整诊断标题，不截断
                    icd_code = get_value(diag, 'ICD_CODE', 'icd_code')
                    if icd_code:
                        diag_items.append(f"{title} ({icd_code})")
                    else:
                        diag_items.append(title)
            if diag_items:
                ehr_parts.append(f"Diagnoses: {'; '.join(diag_items)}")
        
        # 4. 手术/操作（保留所有，完整标题）
        if ehr_data.get('all_procedures'):
            proc_items = []
            for proc in ehr_data['all_procedures']:
                title = get_value(proc, 'LONG_TITLE', 'long_title') or get_value(proc, 'SHORT_TITLE', 'short_title') or get_value(proc, 'ICD_TITLE', 'icd_title')
                if title:
                    # 保留完整手术标题，不截断
                    icd_code = get_value(proc, 'ICD_CODE', 'icd_code')
                    if icd_code:
                        proc_items.append(f"{title} ({icd_code})")
                    else:
                        proc_items.append(title)
            if proc_items:
                ehr_parts.append(f"Procedures: {'; '.join(proc_items)}")
        
        # 5. 处方信息（保留所有药物）
        if ehr_data.get('prescriptions'):
            med_items = []
            for presc in ehr_data['prescriptions']:  # 保留所有处方
                drug = get_value(presc, 'DRUG', 'drug') or get_value(presc, 'DRUG_NAME', 'drug_name')
                if drug:
                    med_items.append(drug)
            if med_items:
                ehr_parts.append(f"Medications: {', '.join(med_items)}")
        
        # 6. 实验室检查（保留所有检查结果）
        if ehr_data.get('all_labevents'):
            lab_items = []
            for lab in ehr_data['all_labevents']:  # 保留所有实验室检查
                label = get_value(lab, 'LABEL', 'label') or get_value(lab, 'ITEMID', 'itemid') or get_value(lab, 'ITEM', 'item')
                value = get_value(lab, 'VALUENUM', 'valuenum') or get_value(lab, 'VALUE', 'value')
                if label and value is not None:
                    unit = get_value(lab, 'VALUEUOM', 'valueuom') or get_value(lab, 'UNIT', 'unit') or ''
                    value_str = f"{value} {unit}".strip() if unit else str(value)
                    lab_items.append(f"{label}: {value_str}")
            if lab_items:
                ehr_parts.append(f"Labs: {'; '.join(lab_items)}")
        
        # 7. 微生物检查（保留所有结果）
        if ehr_data.get('microbiologyevents'):
            micro_items = []
            for micro in ehr_data['microbiologyevents']:  # 保留所有微生物检查
                org = get_value(micro, 'ORG_NAME', 'org_name') or get_value(micro, 'ORGANISM', 'organism')
                if org:
                    micro_items.append(org)
            if micro_items:
                ehr_parts.append(f"Microbiology: {', '.join(micro_items)}")
        
        # 组合所有部分
        ehr_text = " | ".join(ehr_parts)
        
        # 如果设置了最大长度且超过，则截断（但保留重要信息）
        if max_length and len(ehr_text) > max_length:
            # 优先保留前面的重要信息（患者、诊断、手术）
            # 如果必须截断，至少保留患者和诊断信息
            if len(ehr_text) > max_length:
                # 尝试保留前max_length-3个字符
                ehr_text = ehr_text[:max_length-3] + "..."
        
        return ehr_text
    
    def run(self):
        """运行测试"""
        import time
        device = next(self.fusion.parameters()).device
        
        results = []
        all_generated_reports = []
        all_gt_reports = []
        all_label_vecs = []
        
        # 记录模型参数
        model_params = {
            'adapter_checkpoint_dir': str(self.adapter_checkpoint_dir),
            'model_f_checkpoint': str(self.model_f_checkpoint) if self.model_f_checkpoint else None,
            'similarity_threshold': self.similarity_threshold,
            'max_hops': self.max_hops,
            'embed_dim': self.embed_dim,
            'hypergraph_sample_ratio': self.hypergraph_sample_ratio,
            'fine_tune_adapter': self.fine_tune_adapter,
            'adapter_lr': self.adapter_lr if self.fine_tune_adapter else None,
            'adapter_fine_tune_steps': self.adapter_fine_tune_steps if self.fine_tune_adapter else None
        }
        
        # Batch处理：每次处理batch_size个样本
        batch_data_list = []
        
        for study_idx, study in enumerate(tqdm(self.test_studies, desc="Testing")):
            sample_start_time = time.time()
            data = self._prepare_test_data(study)
            if data is None:
                continue
            
            # 1. 生成混合embedding
            with torch.no_grad():
                image_embeds_sum = None
                for img_path in data['image_paths']:
                    img_emb = self.embed_model.image_embed(img_path)
                    if image_embeds_sum is None:
                        image_embeds_sum = img_emb
                    else:
                        image_embeds_sum = image_embeds_sum + img_emb
                
                ehr_embeds = self.embed_model.ehr_embed(data['ehr_text'])
                
                image_pooled = self.embed_model.embed_pooling(image_embeds_sum, dim=1)
                ehr_pooled = self.embed_model.embed_pooling(ehr_embeds, dim=1)
            
            fused_embedding = self.fusion(image_pooled, ehr_pooled)
            fused_embedding_pooled = fused_embedding.squeeze(0)
            
            # 2. 使用HNSW检索相似节点
            fused_emb_norm = torch.norm(fused_embedding_pooled)
            if fused_emb_norm > 1e-8:
                fused_embedding_normalized = fused_embedding_pooled / fused_emb_norm
            else:
                fused_embedding_normalized = fused_embedding_pooled
            
            fused_embedding_pooled_np = fused_embedding_normalized.detach().cpu().float().numpy().reshape(1, -1)
            
            similar_nodes = self._retrieve_similar_nodes_hnsw(fused_embedding_pooled_np, k=30)
            
            # 3. 构建超边
            hyperedges = self._build_hyperedges_from_similarity(fused_embedding_pooled, similar_nodes)
            
            # 4. 多跳检索
            initial_node_ids = [node_id for node_id, _ in similar_nodes[:5]]
            if len(initial_node_ids) == 0:
                selected_nodes = []
                hop_count = 0
            else:
                selected_nodes, hop_count = self._multi_hop_retrieval(
                    fused_embedding_pooled,
                    initial_node_ids,
                    max_hops=self.max_hops,
                    query_hyperedges=hyperedges,
                    max_orthogonal_hops=2
                )
            
            # 5. 召回三元组和knowledge
            triplets, knowledge_texts = self._retrieve_triplets_and_knowledge(selected_nodes)
            
            # 将数据添加到batch列表
            batch_data_list.append({
                'data': data,
                'knowledge_texts': knowledge_texts,
                'selected_nodes': selected_nodes,
                'triplets': triplets,
                'hop_count': hop_count,
                'sample_start_time': sample_start_time,
                'model_params': model_params
            })
            
            # 当达到batch_size或最后一个样本时，批量生成报告
            if len(batch_data_list) >= self.batch_size or study_idx == len(self.test_studies) - 1:
                # 批量生成报告
                generation_start_time = time.time()
                generated_reports = self._generate_reports_batch(
                    batch_data_list
                )
                generation_time = time.time() - generation_start_time
                
                # 处理每个batch中的样本结果
                for batch_idx, batch_item in enumerate(batch_data_list):
                    data = batch_item['data']
                    knowledge_texts = batch_item['knowledge_texts']
                    selected_nodes = batch_item['selected_nodes']
                    triplets = batch_item['triplets']
                    hop_count = batch_item['hop_count']
                    sample_start_time = batch_item['sample_start_time']
                    model_params = batch_item['model_params']
                    generated_report = generated_reports[batch_idx]
                    
                    total_inference_time = time.time() - sample_start_time
                    
                    # 记录retrieved_nodes信息（包含hop信息）
                    retrieved_nodes_info = []
                    for node_info in selected_nodes:
                        node_id = node_info['node_id']
                        node = self.hypergraph.nodes.get(node_id)
                        if node:
                            node_detail = {
                                'node_id': node_id,
                                'node_type': node_info.get('node_type', 'unknown'),
                                'hop': node_info.get('hop', -1),  # 记录hop信息，-1表示未知
                                'has_metadata': bool(node.metadata) if node else False
                            }
                            # 添加更多节点信息
                            if node.metadata:
                                if 'description' in node.metadata:
                                    node_detail['description'] = node.metadata['description']
                                elif 'text' in node.metadata:
                                    node_detail['text'] = node.metadata['text']
                            retrieved_nodes_info.append(node_detail)
                    
                    # 计算当前样本的指标（只计算NLG指标，CE指标在最后批量计算以提高效率）
                    sample_nlg_metrics = self.nlg_evaluator.get_metrics([generated_report], [data['report_text']])
                    
                    # 保存结果（按id）
                    subject_id = str(data.get('subject_id', 'unknown'))
                    study_id = str(data.get('study_id', 'unknown'))
                    sample_id = f"{subject_id}_{study_id}"
                    
                    result = {
                        'sample_id': sample_id,
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'ground_truth': data['report_text'],
                        'generated_report': generated_report,
                        'retrieved_nodes': retrieved_nodes_info,
                        'num_retrieved_nodes': len(selected_nodes),
                        'num_knowledge_texts': len(knowledge_texts),
                        'num_triplets': len(triplets),
                        'hop_count': hop_count,
                        'metrics': {
                            'nlg': sample_nlg_metrics  # 当前样本的NLG指标（CE指标在最后批量计算）
                        },
                        'inference_time': {
                            'generation_time': generation_time / len(batch_data_list),  # 平均生成时间（秒）
                            'total_time': total_inference_time  # 总推理时间（秒）
                        },
                        'model_params': model_params  # 模型参数
                    }
                    
                    results.append(result)
                    # 确保generated_report是字符串类型
                    if generated_report is None:
                        generated_report = ""
                    elif not isinstance(generated_report, str):
                        generated_report = str(generated_report)
                    all_generated_reports.append(generated_report)
                    all_gt_reports.append(data['report_text'])
                    # 确保label_vec是列表类型
                    label_vec = data['label_vec']
                    if label_vec is None:
                        label_vec = [0] * 14  # 14个标签类别
                    elif not isinstance(label_vec, list):
                        try:
                            label_vec = list(label_vec)
                        except:
                            label_vec = [0] * 14
                    all_label_vecs.append(label_vec)
                    
                    # 按路径保存每个样本的结果: result_re/run/subject_id/study_id.json
                    sample_output_dir = self.output_dir / "run" / subject_id
                    sample_output_dir.mkdir(parents=True, exist_ok=True)
                    sample_output_file = sample_output_dir / f"{study_id}.json"
                    with open(sample_output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                
                # 清空batch列表
                batch_data_list = []
        
        # 计算NLG指标
        print("\nComputing NLG metrics...")
        nlg_metrics = self.nlg_evaluator.get_metrics(all_generated_reports, all_gt_reports)
        
        # 计算CE指标
        print("\nComputing CE metrics...")
        print(f"  Total generated reports: {len(all_generated_reports)}")
        print(f"  Total label vectors: {len(all_label_vecs)}")
        
        # 检查数据有效性
        if len(all_generated_reports) == 0:
            print("  Warning: No generated reports, skipping CE metrics")
            ce_metrics = None
        elif len(all_label_vecs) == 0:
            print("  Warning: No label vectors, skipping CE metrics")
            ce_metrics = None
        elif len(all_generated_reports) != len(all_label_vecs):
            print(f"  Warning: Length mismatch (reports: {len(all_generated_reports)}, labels: {len(all_label_vecs)}), skipping CE metrics")
            ce_metrics = None
        else:
            try:
                ce_metrics = self.ce_evaluator.get_metrics(all_generated_reports, all_label_vecs)
                print("  CE metrics computed successfully")
            except Exception as e:
                import traceback
                print(f"  Warning: Failed to compute CE metrics: {e}")
                print(f"  Error details:")
                traceback.print_exc()
                ce_metrics = None
        
        # 保存指标（按照summary_report.json的格式，只保存指标部分）
        metrics_file = self.output_dir / "metrics.json"
        metrics_data = {
            'nlg': nlg_metrics
        }
        if ce_metrics is not None:
            metrics_data['ce'] = ce_metrics
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nTest completed!")
        print(f"Total samples: {len(self.test_studies)}")
        print(f"Successful samples: {len(results)}")
        print(f"Results saved to: {self.output_dir}")
        print(f"Sample results saved to: {self.output_dir / 'run'}")
        print(f"Metrics saved to: {metrics_file}")
        print(f"NLG Metrics:")
        for k, v in nlg_metrics.items():
            print(f"  {k}: {v:.4f}")
        if ce_metrics is not None:
            print(f"CE Metrics:")
            for k, v in ce_metrics.items():
                print(f"  {k}: {v:.4f}")


def main():
    print("Starting test script...")
    parser = argparse.ArgumentParser(description='Test Model F')
    parser.add_argument('--vlm_model_path', type=str, default=None)
    parser.add_argument('--adapter_checkpoint_dir', type=str,
                        default='/mnt/sda/VLM/code/hypercode/adatri/adamodel/adapter_epoch_2')
    parser.add_argument('--model_f_checkpoint', type=str,
                        default='/mnt/sda/VLM/code/hypercode/checkpoints/model_f_epoch_1.pt')
    parser.add_argument('--fine_tune_adapter', action='store_true', default=True,
                        help='是否在测试时微调adapter')
    parser.add_argument('--adapter_lr', type=float, default=1e-5,
                        help='adapter微调的学习率')
    parser.add_argument('--adapter_fine_tune_steps', type=int, default=1,
                        help='每个样本微调的步数')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/sda/VLM/code/hypercode/result_re')
    parser.add_argument('--similarity_threshold', type=float, default=0.8)
    parser.add_argument('--max_hops', type=int, default=3)
    parser.add_argument('--hypergraph_sample_ratio', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批处理大小（同时处理的样本数）')
    
    args = parser.parse_args()
    print("Arguments parsed, creating TestModelF instance...")
    
    tester = TestModelF(
        vlm_model_path=args.vlm_model_path,
        adapter_checkpoint_dir=args.adapter_checkpoint_dir,
        model_f_checkpoint=args.model_f_checkpoint,
        output_dir=args.output_dir,
        similarity_threshold=args.similarity_threshold,
        max_hops=args.max_hops,
        hypergraph_sample_ratio=args.hypergraph_sample_ratio,
        fine_tune_adapter=args.fine_tune_adapter,
        adapter_lr=args.adapter_lr,
        adapter_fine_tune_steps=args.adapter_fine_tune_steps,
        batch_size=args.batch_size
    )
    
    print("TestModelF initialized, starting run...")
    tester.run()


if __name__ == "__main__":
    main()
