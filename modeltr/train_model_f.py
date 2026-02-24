"""
Step 3: Train Model F (Reinforcement Learning)
训练强化学习模型F，用于寻找合适的样本进行adapter微调
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm
import json
import numpy as np

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# 导入模块
from embed_models.image_ehr_embed import ImageEHREmbed
from embed_models.film_fusion import FiLMFusion
from rl_finder.rl_path_finder import RLPathFinder, RewardFunction
from hyperbuild.hypergraph import Hypergraph, NodeType
from hyperbuild.data_loader import DataLoader
from transformers import Gemma3ForConditionalGeneration, AutoProcessor, AutoTokenizer
import hnswlib
from peft import PeftModel


class TrainModelF:
    """训练强化学习模型F"""
    def __init__(
        self,
        vlm_model_path: str = None,
        hypergraph_path: str = None,
        hypergraph: Hypergraph = None,
        adapter_checkpoint_dir: str = "/mnt/sda/VLM/code/hypercode/adatri/adamodel/adapter_epoch_6",
        output_dir: str = "/mnt/sda/VLM/code/hypercode/modeltr/checkpoints",
        num_epochs: int = 20,
        embed_dim: int = 1024,
        similarity_threshold: float = 0.8,
        max_hops: int = 3
    ):
        """
        Args:
            vlm_model_path: VLM模型路径
            hypergraph_path: 超图保存路径（如果提供hypergraph对象则不需要）
            hypergraph: 超图对象（如果提供则直接使用，跳过加载）
            adapter_checkpoint_dir: Adapter checkpoint目录（第10轮）
            output_dir: 模型F保存目录
            num_epochs: 训练轮数
            embed_dim: embedding维度
            similarity_threshold: 相似度阈值（用于构建超边）
            max_hops: 最大hop数量
        """
        self.vlm_model_path = vlm_model_path
        self.adapter_checkpoint_dir = Path(adapter_checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件
        self.log_file = self.output_dir / "training_log.jsonl"
        self.max_tokens = 12000  # 最大token数限制
        
        self.num_epochs = num_epochs
        self.embed_dim = embed_dim
        self.similarity_threshold = similarity_threshold
        self.max_hops = max_hops
        
        # 加载超图：优先使用传入的超图对象，否则从文件加载
        if hypergraph is not None:
            print("Using provided hypergraph object (skipping file load)...")
            self.hypergraph = hypergraph
        elif hypergraph_path is not None:
            print("Loading hypergraph from file...")
            self.hypergraph_path = Path(hypergraph_path)
            hypergraph_file = str(self.hypergraph_path / "hypergraph.json")
            self.hypergraph = Hypergraph.load(hypergraph_file)
        else:
            raise ValueError("Either hypergraph or hypergraph_path must be provided")
        
        # 加载embedding模型
        print("Loading embedding model...")
        self.embed_model = ImageEHREmbed(model_path=self.vlm_model_path)
        device = next(self.embed_model.model.parameters()).device
        
        # 动态获取实际的embedding维度（从超图中的节点embedding）
        # 或者从embed_model获取一个样本的embedding维度
        actual_embed_dim = embed_dim
        if len(self.hypergraph.nodes) > 0:
            # 从超图中获取第一个节点的embedding维度
            first_node = next(iter(self.hypergraph.nodes.values()))
            if first_node.embedding is not None:
                if isinstance(first_node.embedding, torch.Tensor):
                    actual_embed_dim = first_node.embedding.shape[-1]
                else:
                    actual_embed_dim = len(first_node.embedding) if hasattr(first_node.embedding, '__len__') else embed_dim
                print(f"Detected actual embedding dimension: {actual_embed_dim}")
        
        # 更新embed_dim为实际值
        self.embed_dim = actual_embed_dim
        
        # 加载FiLM融合模块（使用实际维度）
        print(f"Loading FiLM fusion module (dim={actual_embed_dim})...")
        self.fusion = FiLMFusion(dim=actual_embed_dim).to(device).to(torch.bfloat16)
        
        # 加载强化学习路径寻找器（使用实际维度）
        print(f"Loading RL path finder (embed_dim={actual_embed_dim})...")
        self.rl_finder = RLPathFinder(
            embed_dim=actual_embed_dim,
            hidden_dim=512,
            max_hops=max_hops,
            similarity_threshold=similarity_threshold,
            temperature=0.01  # 温度参数：<1让分布更尖锐，放大差异（0.01非常尖锐）
        ).to(device).to(torch.bfloat16)
        
        # 加载奖励函数
        self.reward_fn = RewardFunction()
        
        # 确定基础模型路径（用于加载processor和tokenizer）
        if self.vlm_model_path:
            base_model_path = str(Path(self.vlm_model_path).resolve())
        else:
            base_model_path = "/mnt/sda/VLM/code/model_cache/models--google--medgemma-4b-it/snapshots/290cda5eeccbee130f987c4ad74a59ae6f196408"
        
        # 加载processor和tokenizer（从基础模型路径）
        print("Loading processor and tokenizer from base model...")
        self.processor = AutoProcessor.from_pretrained(base_model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
        
        # 先加载基础模型
        print("Loading base VLM model...")
        self.vlm_model = Gemma3ForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        
        # 加载adapter（第10轮权重）
        # 注意：即使checkpoint保存为完整模型格式，但权重结构仍然是LoRA adapter
        print(f"Loading adapter from {self.adapter_checkpoint_dir}...")
        self._load_adapter()
        
        # 构建HNSW索引
        print("Building HNSW index...")
        self._build_hnsw_index()
        
        # 数据加载器（延迟加载，只在训练时使用）
        # 注意：这里不立即加载所有训练数据，避免初始化时卡住
        # 训练数据会在 run() 方法中按需加载
        print("Initializing data loader (lazy loading, 0.5% sample)...")
        self.data_loader = DataLoader(train_sample_ratio=0.005)
        # 不在这里加载所有训练数据，避免卡住
        # self.train_studies 会在 run() 方法中按需加载
        self.train_studies = None
    
    def _load_adapter(self):
        """加载adapter（第10轮权重）"""
        import json
        adapter_path = Path(self.adapter_checkpoint_dir)
        
        # 检查是否有adapter_config.json
        adapter_config_file = adapter_path / "adapter_config.json"
        if adapter_config_file.exists():
            # 标准PEFT adapter格式
            self.vlm_model = PeftModel.from_pretrained(
                self.vlm_model,
                str(adapter_path),
                adapter_name="last_ffn",
                local_files_only=True
            )
        else:
            # 没有adapter_config.json，从training_info.json创建配置
            training_info_file = adapter_path / "training_info.json"
            if training_info_file.exists():
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
                self.vlm_model = get_peft_model(self.vlm_model, peft_config, adapter_name="last_ffn")
                
                # 直接加载本地safetensors文件
                import safetensors.torch
                adapter_weights = {}
                
                # 检查是否有分片的safetensors文件
                index_file = adapter_path / "model.safetensors.index.json"
                if index_file.exists():
                    # 分片权重文件
                    import json
                    with open(index_file, 'r') as f:
                        index_data = json.load(f)
                    
                    weight_map = index_data.get("weight_map", {})
                    for weight_name, shard_file in weight_map.items():
                        shard_path = adapter_path / shard_file
                        if shard_path.exists():
                            shard_weights = safetensors.torch.load_file(str(shard_path))
                            # 只加载LoRA相关的权重
                            for key, value in shard_weights.items():
                                if "lora" in key.lower() or "last_ffn" in key:
                                    adapter_weights[key] = value
                else:
                    # 单个safetensors文件或查找所有safetensors文件
                    safetensors_files = list(adapter_path.glob("*.safetensors"))
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
                set_peft_model_state_dict(self.vlm_model, adapter_weights, adapter_name="last_ffn")
            else:
                # 尝试直接加载（可能权重文件格式特殊）
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
        
        # 设置adapter为活跃状态
        self.vlm_model.set_adapter("last_ffn")
        print(f"Adapter loaded from {self.adapter_checkpoint_dir}")
    
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
        
        # 创建HNSW索引
        self.hnsw_index = hnswlib.Index(space='cosine', dim=embed_dim)
        self.hnsw_index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        indices = np.arange(len(embeddings))
        self.hnsw_index.add_items(embeddings, indices)
        self.hnsw_index.set_ef(50)
        
        # 建立映射
        self.node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        self.index_to_node_id = {i: node_id for i, node_id in enumerate(node_ids)}
        print(f"HNSW index built with {len(embeddings)} nodes")
    
    def _retrieve_similar_nodes_hnsw(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: float = None
    ) -> List[Tuple[str, float]]:
        """
        使用HNSW检索相似节点（贪心搜索：HNSW内部会在每层找最相似的点）
        Args:
            query_embedding: 查询embedding
            k: 检索的候选数量（HNSW会贪心搜索返回top-k）
            threshold: 相似度阈值，只返回相似度 >= threshold 的节点
        Returns:
            List of (node_id, similarity) tuples
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        # HNSW的knn_query本身就是贪心搜索，会在图的每一层找到最相似的节点
        labels, distances = self.hnsw_index.knn_query(query_embedding, k=k)
        results = []
        similarities_list = []
        for label, dist in zip(labels[0], distances[0]):
            similarity = 1.0 - dist  # cosine distance to similarity
            similarities_list.append(similarity)
            if similarity >= threshold:
                node_id = self.index_to_node_id[int(label)]
                results.append((node_id, similarity))
        
        
        return results
    
    def _build_hyperedges_from_similarity(
        self,
        query_embedding: torch.Tensor,
        similar_nodes: List[Tuple[str, float]]
    ) -> List[List[str]]:
        """
        基于相似度构建超边（HNSW层次结构策略）
        策略：第一层选择top-k个节点控制超边数量，每条超边不断向下检索HNSW的下一层，
              找到与上一层最相似的节点，直到找不到超过阈值的节点
        Args:
            query_embedding: (embed_dim,) 查询embedding
            similar_nodes: List of (node_id, similarity) tuples
        Returns:
            List of hyperedges (each hyperedge is a list of node_ids representing a path)
        """
        if len(similar_nodes) < 2:
            return []
        
        # HNSW层次结构策略：
        # 1. 第一层：选择top-k个节点（控制超边数量，比如top-10）
        # 2. 对每个第一层节点，不断向下检索HNSW的下一层
        # 3. 每层找到与上一层最相似的节点（k=1），添加到超边
        # 4. 直到找不到超过阈值的节点为止
        # 5. 每条超边是一条从第一层节点开始的"路径"
        
        top_k_first_layer = 10  # 第一层选择top-10个节点（控制超边数量）
        k_per_layer = 1  # 每层只找最相似的1个节点（与上一层最相似的）
        max_layers = 10  # 最大层数限制（防止无限循环）
        
        # 按相似度排序，选择第一层的top-k个节点
        similar_nodes_sorted = sorted(similar_nodes, key=lambda x: x[1], reverse=True)
        first_layer_nodes = similar_nodes_sorted[:top_k_first_layer]
        
        hyperedges = []
        
        # 对每个第一层节点，构建一条超边（路径）
        for first_node_id, first_sim in first_layer_nodes:
            if first_node_id not in self.hypergraph.nodes:
                continue
            
            # 当前超边：从第一层节点开始
            current_hyperedge = [first_node_id]
            visited_nodes = {first_node_id}  # 避免重复添加节点
            
            # 当前层的节点（用于下一层检索）
            current_layer_node_id = first_node_id
            
            # 不断向下检索，直到找不到超过阈值的节点
            for layer in range(max_layers):
                # 获取当前层节点的embedding
                current_node = self.hypergraph.nodes[current_layer_node_id]
                current_emb = current_node.embedding
                if isinstance(current_emb, torch.Tensor):
                    current_emb_np = current_emb.cpu().float().numpy()
                else:
                    current_emb_np = np.array(current_emb).astype('float32')
                
                # 使用HNSW检索下一层节点（找到与当前层节点最相似的）
                current_emb_np = current_emb_np.reshape(1, -1)
                next_layer_results = self._retrieve_similar_nodes_hnsw(
                    current_emb_np, 
                    k=k_per_layer + len(visited_nodes),  # k要大于已访问节点数，确保能找到新节点
                    threshold=self.similarity_threshold
                )
                
                # 找到下一个未访问的节点（与当前层节点最相似的）
                next_node_id = None
                for node_id, sim in next_layer_results:
                    if node_id not in visited_nodes:
                        next_node_id = node_id
                        break
                
                # 如果找不到新节点，停止检索
                if next_node_id is None:
                    break
                
                # 将下一层节点添加到超边
                current_hyperedge.append(next_node_id)
                visited_nodes.add(next_node_id)
                
                # 更新当前层节点，继续下一层检索
                current_layer_node_id = next_node_id
            
            # 如果超边至少包含2个节点，添加到结果中
            if len(current_hyperedge) >= 2:
                hyperedges.append(current_hyperedge)
        
        return hyperedges
    
    def _collect_candidate_nodes_recursive(
        self,
        initial_nodes: List[str],
        query_hyperedges: List[List[str]] = None,
        max_hops: int = 3
    ) -> List[Dict]:
        """
        递归收集候选节点：通过超边连接多跳收集
        1. 这些超边连接的节点
        2. 这些超边连接的节点的超边连接的节点
        3. 这些超边连接的节点的超边连接的节点的超边连接的节点
        ...
        Args:
            initial_nodes: 初始节点ID列表
            query_hyperedges: 查询时构建的超边列表
            max_hops: 最大跳数
        Returns:
            candidate_nodes: 所有候选节点列表
        """
        candidate_nodes = []
        visited_node_ids = set()  # 用于去重
        current_layer_nodes = set(initial_nodes)
        
        for hop in range(max_hops):
            if len(current_layer_nodes) == 0:
                break
            
            next_layer_nodes = set()
            
            # 遍历当前层的所有节点
            for node_id in current_layer_nodes:
                if node_id in visited_node_ids:
                    continue
                visited_node_ids.add(node_id)
                
                # 第一跳：优先使用查询时构建的超边
                if hop == 0 and query_hyperedges is not None and len(query_hyperedges) > 0:
                    for hyperedge in query_hyperedges:
                        if node_id in hyperedge:
                            # 添加超边中的所有节点
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
                
                # 从超图中获取该节点参与的超边
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
            
            # 更新当前层节点为下一层节点
            current_layer_nodes = next_layer_nodes
        
        return candidate_nodes
    
    def _multi_hop_retrieval(
        self,
        query_embedding: torch.Tensor,
        initial_nodes: List[str],
        max_hops: int = None,
        query_hyperedges: List[List[str]] = None,
        max_orthogonal_hops: int = 2,
        previous_selected_nodes: List[Dict] = None
    ) -> Tuple[List[Dict], int]:
        """
        多跳检索：先递归收集所有候选节点，然后用MLP选择激活的节点
        支持正交化检索：最多再正交2次（总共3次，包括第一次）
        Args:
            query_embedding: (embed_dim,) 查询embedding
            initial_nodes: 初始节点ID列表
            max_hops: 最大hop数量（用于收集候选节点）
            query_hyperedges: 查询时构建的超边列表（每个超边是一个节点ID列表）
            max_orthogonal_hops: 最大正交化跳数（除了第一次，最多再正交2次）
            previous_selected_nodes: 之前已选中的节点列表（用于正交化）
        Returns:
            (selected_nodes, hop_count): 选中的节点列表、hop数量
        """
        if max_hops is None:
            max_hops = self.max_hops
        
        all_selected_nodes = []
        current_query = query_embedding
        orthogonal_threshold = 0.3
        
        # 第一次检索（非正交化）
        # 1. 递归收集所有候选节点（通过超边连接多跳）
        candidate_nodes = self._collect_candidate_nodes_recursive(
            initial_nodes, 
            query_hyperedges, 
            max_hops=max_hops
        )
        
        # 1.5. 强制添加所有knowledge节点到候选节点中（总共只有13个）
        # 从超图中获取所有knowledge节点（使用NodeType枚举进行比较）
        knowledge_nodes = []
        all_node_types_count = {}
        
        # 遍历所有节点，统计类型并找到knowledge节点
        for node_id, node in self.hypergraph.nodes.items():
            # 统计所有节点类型
            node_type_str = str(node.node_type)
            all_node_types_count[node_type_str] = all_node_types_count.get(node_type_str, 0) + 1
            
            # 检查是否是knowledge节点（使用多种方式比较）
            is_knowledge = (
                node.node_type == NodeType.KNOWLEDGE or
                str(node.node_type) == 'NodeType.KNOWLEDGE' or
                (hasattr(node.node_type, 'value') and node.node_type.value == 'knowledge')
            )
            
            if is_knowledge:
                knowledge_nodes.append((node_id, node))
        
        # 检查候选节点中是否已经有这些knowledge节点
        existing_knowledge_ids = {node.get('node_id') for node in candidate_nodes if node.get('node_type', '').lower() == 'knowledge'}
        
        # 添加所有knowledge节点到候选节点中（如果还没有的话）
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
                    'node_type': 'knowledge'  # 确保类型是'knowledge'
                })
                added_count += 1
        
        if len(candidate_nodes) == 0:
            return [], 0
        
        # 2. 使用MLP选择激活的节点
        device = query_embedding.device
        query_emb_tensor = current_query.to(device)
        
        probs, value, activation_scores = self.rl_finder(query_emb_tensor, candidate_nodes, debug=False)
        
        # 3. 选择节点：使用softmax + 概率阈值，KNOWLEDGE节点强制召回
        selected_indices = self.rl_finder.select_nodes(
            probs, 
            activation_scores=activation_scores,
            prob_threshold=0.01,
            debug=False, 
            candidate_nodes=candidate_nodes
        )
        
        # 4. 添加选中的节点
        selected_nodes = [candidate_nodes[idx] for idx in selected_indices]
        all_selected_nodes.extend(selected_nodes)
        hop_count = max_hops
        
        # 正交化检索（最多再正交2次）
        for orthogonal_hop in range(max_orthogonal_hops):
            if len(all_selected_nodes) == 0:
                break
            
            # 计算正交化向量（排除knowledge节点）
            dtype = current_query.dtype
            # 排除knowledge节点，只使用非knowledge节点计算平均embedding
            non_knowledge_nodes = [
                n for n in all_selected_nodes 
                if n.get('node_type', '').lower() != 'knowledge'
            ]
            
            if len(non_knowledge_nodes) == 0:
                # 如果没有非knowledge节点，跳过正交化
                break
            
            retrieved_embs = [
                torch.tensor(n['embedding'], device=device, dtype=dtype) if not isinstance(n['embedding'], torch.Tensor)
                else n['embedding'].to(device=device, dtype=dtype)
                for n in non_knowledge_nodes  # 只使用非knowledge节点
            ]
            
            # 正交化：计算查询embedding相对于已选中节点平均embedding的正交差异向量
            query_norm_before = torch.norm(current_query).item()
            mean_emb = torch.stack(retrieved_embs).mean(dim=0)
            mean_norm_before = torch.norm(mean_emb).item()
            
            # 归一化查询向量和平均embedding（模长都为1）
            query_norm = torch.norm(current_query)
            if query_norm > 1e-8:
                query_normalized = current_query / query_norm
            else:
                query_normalized = current_query
            
            mean_norm = torch.norm(mean_emb)
            if mean_norm > 1e-8:
                mean_normalized = mean_emb / mean_norm
            else:
                mean_normalized = mean_emb
            
            # 计算点积和投影
            dot_ab = torch.dot(query_normalized, mean_normalized).item()
            dot_bb = torch.dot(mean_normalized, mean_normalized).item()  # 应该是1.0（因为已归一化）
            projection = (dot_ab / dot_bb) * mean_normalized
            proj_norm = torch.norm(projection).item()
            
            # 计算正交差异向量
            orthogonal_emb = self._orthogonalize_embedding(current_query, retrieved_embs)
            
            # 检查正交向量有效性（使用正交化后的模长）
            orthogonal_norm = torch.norm(orthogonal_emb).item()
            
            if orthogonal_norm < orthogonal_threshold:
                break
            
            # 使用正交化后的embedding进行HNSW检索
            # 注意：正交化后的embedding已经归一化（在_orthogonalize_embedding中归一化了query_normalized）
            # 但HNSW索引中的节点embedding可能没有归一化，所以需要归一化查询向量以确保一致性
            orthogonal_emb_norm = torch.norm(orthogonal_emb)
            if orthogonal_emb_norm > 1e-8:
                orthogonal_emb_normalized = orthogonal_emb / orthogonal_emb_norm
            else:
                orthogonal_emb_normalized = orthogonal_emb
            
            query_emb_np = orthogonal_emb_normalized.detach().cpu().float().numpy().reshape(1, -1)
            
            # 根据正交向量的实际模长调整阈值
            # 如果正交向量模长较小，说明大部分信息已经被投影掉了，阈值应该相应降低
            # 阈值 = 正交向量模长 * 基础阈值(0.8)
            adjusted_threshold = orthogonal_emb_norm.item() * 0.8
            
            # 使用调整后的阈值进行HNSW检索
            similar_nodes_orth = self._retrieve_similar_nodes_hnsw(query_emb_np, k=30, threshold=adjusted_threshold)
            
            if len(similar_nodes_orth) == 0:
                break
            
            # 检查是否有新节点
            existing_node_ids = {n['node_id'] for n in all_selected_nodes}
            new_node_ids = [node_id for node_id, _ in similar_nodes_orth if node_id not in existing_node_ids]
            
            if len(new_node_ids) == 0:
                break
            
            # 重新构建超边（使用正交化后的embedding和相似节点）
            orthogonal_emb_tensor = orthogonal_emb.to(device)
            hyperedges_orth = self._build_hyperedges_from_similarity(orthogonal_emb_tensor, similar_nodes_orth)
            
            # 使用新节点作为初始节点，重新进行多跳检索（不递归正交化，避免无限循环）
            selected_nodes_orth, hop_count_orth = self._multi_hop_retrieval(
                orthogonal_emb,
                new_node_ids[:5],  # 只取前5个作为初始节点
                max_hops=max_hops,
                query_hyperedges=hyperedges_orth,
                max_orthogonal_hops=0,  # 不再递归正交化
                previous_selected_nodes=all_selected_nodes
            )
            
            if len(selected_nodes_orth) == 0:
                break
            
            # 检查本次选中的节点是否都是之前已选中的
            existing_node_ids = {n['node_id'] for n in all_selected_nodes}
            newly_selected_ids = {n['node_id'] for n in selected_nodes_orth}
            
            # 如果所有新选中的节点都是之前已选中的，停止循环
            if newly_selected_ids.issubset(existing_node_ids):
                break
            
            # 合并选中的节点
            all_selected_nodes.extend(selected_nodes_orth)
            hop_count += hop_count_orth
            
            # 更新查询向量为当前正交向量
            current_query = orthogonal_emb
        
        return all_selected_nodes, hop_count
    
    def _retrieve_triplets_and_knowledge(
        self,
        selected_nodes: List[Dict]
    ) -> Tuple[List[Dict], List[str]]:
        """
        从选中的节点中召回三元组和knowledge
        Args:
            selected_nodes: 选中的节点列表
        Returns:
            (triplets, knowledge_texts): 
                triplets: 三元组列表，每个三元组包含 {'ehr_texts': [...], 'image_paths': [...], 'report_text': ...}
                knowledge_texts: knowledge文本列表（不限制token数）
        """
        triplets = []
        knowledge_texts = []
        processed_triplet_keys = set()  # 避免重复的三元组
        
        prompt = "Based on the above patient information and clinical history, Please generate a paragraph of radiology report for this chest X-ray image."
        
        for node_info in selected_nodes:
            node_id = node_info['node_id']
            node = self.hypergraph.nodes[node_id]
            
            if node.node_type == NodeType.KNOWLEDGE:
                # 从metadata中获取knowledge文本（不限制token数）
                if 'description' in node.metadata:
                    knowledge_texts.append(node.metadata['description'])
                elif 'text' in node.metadata:
                    knowledge_texts.append(node.metadata['text'])
            
            # 提取三元组（如果有）
            if 'triplets' in node.metadata:
                for triplet_info in node.metadata['triplets']:
                    # 使用subject_id和study_id作为唯一标识
                    triplet_key = (triplet_info.get('subject_id'), triplet_info.get('study_id'))
                    if triplet_key in processed_triplet_keys:
                        continue
                    processed_triplet_keys.add(triplet_key)
                    
                    # 从三元组的所有节点中提取原始数据
                    ehr_texts = []
                    image_paths = []
                    report_text = None
                    
                    # 提取EHR文本
                    for ehr_node_id in triplet_info.get('ehr_node_ids', []):
                        if ehr_node_id in self.hypergraph.nodes:
                            ehr_node = self.hypergraph.nodes[ehr_node_id]
                            if 'ehr_text' in ehr_node.metadata:
                                ehr_texts.append(ehr_node.metadata['ehr_text'])
                    
                    # 提取图像路径
                    for image_node_id in triplet_info.get('image_node_ids', []):
                        if image_node_id in self.hypergraph.nodes:
                            image_node = self.hypergraph.nodes[image_node_id]
                            if 'image_path' in image_node.metadata:
                                image_paths.append(image_node.metadata['image_path'])
                    
                    # 提取报告文本
                    for report_node_id in triplet_info.get('report_node_ids', []):
                        if report_node_id in self.hypergraph.nodes:
                            report_node = self.hypergraph.nodes[report_node_id]
                            if 'report_text' in report_node.metadata:
                                report_text = report_node.metadata['report_text']
                                break  # 通常只有一个报告节点
                    
                    # 如果有完整的三元组数据，检查 ehr + prompt 的 token 数（用于微调时的输入：image + ehr + prompt）
                    if ehr_texts and image_paths and report_text:
                        # 合并所有 ehr_texts
                        combined_ehr_text = "\n".join(ehr_texts)
                        # 检查 ehr + prompt 的 token 数（image 不占用文本 token，但需要预留空间）
                        ehr_prompt_text = combined_ehr_text + "\n" + prompt
                        tokenized_ehr_prompt = self.tokenizer(
                            ehr_prompt_text,
                            return_tensors="pt",
                            add_special_tokens=False,
                            truncation=False
                        )
                        ehr_prompt_tokens = tokenized_ehr_prompt["input_ids"].shape[1]
                        
                        # 如果超过限制（预留256 tokens给图像），跳过该三元组
                        # 微调时输入是 image + ehr + prompt，所以检查 ehr + prompt 的 token 数，预留空间给 image
                        if ehr_prompt_tokens <= self.max_tokens - 256:
                            triplets.append({
                                'ehr_texts': ehr_texts,
                                'image_paths': image_paths,
                                'report_text': report_text,
                                'subject_id': triplet_info.get('subject_id'),
                                'study_id': triplet_info.get('study_id')
                            })
        
        return triplets, knowledge_texts
    
    def _fine_tune_adapter(
        self,
        ehr_text: str,
        image_paths: List[str],
        report_text: str,
        selected_nodes: List[Dict] = None
    ):
        """
        使用当前样本和检索到的节点的unpooled embeddings微调adapter（只微调一步）
        Args:
            ehr_text: EHR文本
            image_paths: 图像路径列表（使用所有图像）
            report_text: 真实报告文本（作为标签）
            selected_nodes: 检索到的节点列表，用于获取embedding_unpooled
        """
        # 启用adapter训练
        self.vlm_model.train()
        for name, param in self.vlm_model.named_parameters():
            if 'lora' in name.lower() or 'adapter' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 优化器
        optimizer = torch.optim.AdamW(
            [p for p in self.vlm_model.parameters() if p.requires_grad],
            lr=1e-5
        )
        
        device = self.vlm_model.device
        
        # 从选中的节点中获取embedding_unpooled，用于微调
        unpooled_embeddings = []
        if selected_nodes:
            for node_info in selected_nodes:
                node_id = node_info['node_id']
                # 按需加载unpooled embedding
                unpooled_emb = self.hypergraph.load_node_unpooled(node_id)
                if unpooled_emb is not None:
                    # 转换为tensor并移到设备上
                    if isinstance(unpooled_emb, torch.Tensor):
                        unpooled_emb = unpooled_emb.to(device).float()
                    else:
                        unpooled_emb = torch.tensor(unpooled_emb, device=device, dtype=torch.float32)
                    unpooled_embeddings.append(unpooled_emb)
        
        # 构建输入：图像 + EHR + prompt（不放入knowledge）
        prompt = "Based on the above patient information and clinical history, Please generate a paragraph of radiology report for this chest X-ray image."
        
        # 检查 ehr_text + prompt 的 token 数，限制在 max_tokens 以内（预留空间给图像，约256 tokens）
        ehr_prompt_text = ehr_text + "\n" + prompt
        tokenized_ehr_prompt = self.tokenizer(
            ehr_prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False
        )
        ehr_prompt_tokens = tokenized_ehr_prompt["input_ids"].shape[1]
        
        # 如果超过限制，截断 ehr_text
        if ehr_prompt_tokens > self.max_tokens - 256:
            # 计算 prompt 的 token 数
            prompt_tokens = len(self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0])
            max_ehr_tokens = self.max_tokens - 256 - prompt_tokens - 10  # 预留一些空间
            
            # 截断 ehr_text
            tokenized_ehr = self.tokenizer(
                ehr_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_ehr_tokens
            )
            ehr_text = self.tokenizer.decode(tokenized_ehr["input_ids"][0], skip_special_tokens=True)
            ehr_prompt_text = ehr_text + "\n" + prompt
        
        # 构建包含所有图像的消息
        content = []
        for img_path in image_paths:
            content.append({"type": "image", "image": img_path})
        content.append({"type": "text", "text": ehr_prompt_text})
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(device)
        
        # 构建完整的输入（包含报告作为标签）
        # 需要将报告文本添加到输入中，并正确设置labels
        # 注意：这里使用上面已经截断过的 ehr_prompt_text
        full_text = ehr_prompt_text + "\n" + report_text
        # 构建包含所有图像的完整消息
        full_content = []
        for img_path in image_paths:
            full_content.append({"type": "image", "image": img_path})
        full_content.append({"type": "text", "text": full_text})
        
        full_messages = [{
            "role": "user",
            "content": full_content
        }]
        
        full_inputs = self.processor.apply_chat_template(
            full_messages, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(device)
        
        # 创建labels：只对报告部分计算loss，其他部分设为-100（忽略）
        input_ids = full_inputs["input_ids"]
        # 找到prompt结束的位置（报告开始的位置）
        prompt_text = ehr_text + "\n" + prompt + "\n"
        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to(device)
        
        # 报告部分的token
        report_ids = self.tokenizer(
            report_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=512
        )["input_ids"].to(device)
        
        # 创建labels：与input_ids相同形状，但只有报告部分有标签
        labels = input_ids.clone()
        labels.fill_(-100)  # 默认全部忽略
        
        # 找到报告在input_ids中的位置并设置labels
        # 简化处理：假设报告在输入的最后部分
        report_len = report_ids.shape[1]
        if report_len > 0 and report_len < labels.shape[1]:
            # 将报告部分的labels设置为实际的token ids
            labels[0, -report_len:] = report_ids[0]
        
        # 前向传播
        # 如果有unpooled embeddings，可以通过某种方式注入到模型中
        # 这里先使用标准的forward，后续可以根据需要修改
        outputs = self.vlm_model(
            input_ids=input_ids,
            pixel_values=full_inputs.get("pixel_values"),
            labels=labels
        )
        
        loss = outputs.loss
        
        # 如果有unpooled embeddings，可以添加额外的损失项
        # 例如：使用unpooled embeddings作为额外的监督信号
        if len(unpooled_embeddings) > 0:
            # 这里可以添加基于unpooled embeddings的辅助损失
            # 例如：确保生成的embedding与检索到的embedding相似
            # 暂时先不添加，保持原有逻辑
            pass
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 切换回eval模式
        self.vlm_model.eval()
        
        return loss.item()
    
    def _generate_report(
        self,
        ehr_text: str,
        image_paths: List[str],
        knowledge_texts: List[str] = None
    ) -> str:
        """
        使用微调后的adapter生成报告
        Args:
            ehr_text: EHR文本
            image_paths: 图像路径列表（使用所有图像）
            knowledge_texts: knowledge文本列表（如果有）
        Returns:
            generated_report: 生成的报告文本
        """
        self.vlm_model.eval()
        
        # 构建输入：图像 + EHR + knowledge（如果有）+ prompt
        # knowledge 不限制 token 数，直接添加到输入中
        prompt = "Based on the above patient information and clinical history, Please generate a paragraph of radiology report for this chest X-ray image."
        
        text_content = ehr_text
        if knowledge_texts and len(knowledge_texts) > 0:
            knowledge_str = "\n".join(knowledge_texts)
            text_content = ehr_text + "\n\nKnowledge:\n" + knowledge_str
        
        text_content = text_content + "\n" + prompt
        
        # 构建包含所有图像的消息
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
        
        # 获取输入长度（用于后续截取新生成的部分）
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            input_length = input_ids.shape[1]
        else:
            input_length = 0
        
        with torch.no_grad():
            # 生成报告
            outputs = self.vlm_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
            
            # 只解码新生成的部分（从input_length开始）
            # outputs[0] 包含 [输入部分 + 新生成部分]
            if input_length > 0 and len(outputs[0]) > input_length:
                generated_ids = outputs[0][input_length:]  # 只取新生成的部分
            else:
                generated_ids = outputs[0]  # 如果没有输入或输出长度异常，使用全部
            
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def _orthogonalize_embedding(
        self,
        query_embedding: torch.Tensor,
        retrieved_embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        正交化：计算查询embedding相对于已选中节点平均embedding的正交差异向量
        方法：
        1. 先归一化查询向量和平均embedding（模长都为1）
        2. 计算查询向量在平均embedding上的投影
        3. 计算正交差异向量 c = a - projection
        """
        if len(retrieved_embeddings) == 0:
            return query_embedding
        
        # 确保所有tensor在相同的设备和dtype上
        device = query_embedding.device
        dtype = query_embedding.dtype
        
        # 将retrieved_embeddings转换为正确的dtype和设备
        retrieved_embeddings = [
            emb.to(device=device, dtype=dtype) if isinstance(emb, torch.Tensor)
            else torch.tensor(emb, device=device, dtype=dtype)
            for emb in retrieved_embeddings
        ]
        
        # 先计算mean embedding（代表平均的检索方向）
        mean_embedding = torch.stack(retrieved_embeddings).mean(dim=0)  # (embed_dim,)
        
        # 1. 归一化查询向量和平均embedding（模长都为1）
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
        
        # 2. 计算查询向量在归一化后的平均embedding上的投影
        # 公式: proj = (a·b) / (b·b) * b
        # 由于b已经归一化，b·b = 1，所以 proj = (a·b) * b
        dot_ab = torch.dot(query_normalized, mean_normalized)  # a与b的点积
        dot_bb = torch.dot(mean_normalized, mean_normalized)    # b与b的点积（应该是1.0，因为已归一化）
        projection = (dot_ab / dot_bb) * mean_normalized
        
        # 3. 计算正交差异向量 c = a - projection
        orthogonal_embedding = query_normalized - projection
        
        return orthogonal_embedding
    
    def _prepare_training_data(self, study: Dict) -> Optional[Dict]:
        """准备训练数据，过滤掉token数超过限制的样本"""
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
        
        # 检查token数量，过滤掉过大的样本
        prompt = "Based on the above patient information and clinical history, Please generate a paragraph of radiology report for this chest X-ray image."
        full_text = ehr_text + "\n" + prompt + "\n" + report_text
        
        # 估算token数量（使用tokenizer）
        tokenized = self.tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False
        )
        num_tokens = tokenized["input_ids"].shape[1]
        
        # 如果token数超过限制，跳过该样本
        if num_tokens > self.max_tokens:
            return None
        
        return {
            'image_paths': image_paths,  # 使用所有图像
            'ehr_text': ehr_text,
            'report_text': report_text,
            'num_tokens': num_tokens
        }
    
    def _format_ehr(self, ehr_data: Dict) -> str:
        """格式化EHR数据为文本"""
        parts = []
        for field_name, field_data in ehr_data.items():
            formatted = self.data_loader.format_ehr_field(field_name, field_data)
            if formatted:
                parts.append(f"{field_name}: {formatted}")
        return " | ".join(parts)
    
    def run(self):
        """运行训练"""
        device = next(self.fusion.parameters()).device
        
        # 优化器
        optimizer = torch.optim.AdamW(
            list(self.fusion.parameters()) + list(self.rl_finder.parameters()),
            lr=1e-4
        )
        
        self.fusion.train()
        self.rl_finder.train()
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            epoch_loss = 0.0
            epoch_adapter_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_reward = 0.0
            num_samples = 0
            skipped_samples = 0
            
            # 延迟加载训练数据（如果还没有加载）
            if self.train_studies is None:
                print("Loading training studies (this may take a while)...")
                self.train_studies = self.data_loader.get_train_studies()
                print(f"Loaded {len(self.train_studies)} training studies")
            
            # 使用所有已采样的训练数据（DataLoader已经采样了1%的subjects）
            for study_idx, study in enumerate(tqdm(self.train_studies, desc=f"Epoch {epoch + 1}")):
                data = self._prepare_training_data(study)
                if data is None:
                    skipped_samples += 1
                    continue
                
                optimizer.zero_grad()
                
                # 1. 生成混合embedding
                with torch.no_grad():
                    # 对所有图像进行embedding，然后相加
                    image_embeds_sum = None
                    for img_path in data['image_paths']:
                        img_emb = self.embed_model.image_embed(img_path)  # (1, seq_len, embed_dim)
                        if image_embeds_sum is None:
                            image_embeds_sum = img_emb
                        else:
                            image_embeds_sum = image_embeds_sum + img_emb  # (1, seq_len, embed_dim)
                    
                    ehr_embeds = self.embed_model.ehr_embed(data['ehr_text'])  # (1, seq_len, embed_dim)
                    
                    # 先pooling，参考embed_model.py的方式
                    image_pooled = self.embed_model.embed_pooling(image_embeds_sum, dim=1)  # (1, embed_dim)
                    ehr_pooled = self.embed_model.embed_pooling(ehr_embeds, dim=1)  # (1, embed_dim)
                
                fused_embedding = self.fusion(image_pooled, ehr_pooled)  # (1, embed_dim)
                fused_embedding_pooled = fused_embedding.squeeze(0)  # (embed_dim,)
                
                # 2. 使用HNSW检索相似节点（只使用融合后的embedding）
                # 增大k值以获取更多候选节点
                # 归一化fused_embedding以确保与HNSW索引中的节点embedding一致（HNSW使用cosine距离）
                fused_emb_norm = torch.norm(fused_embedding_pooled)
                if fused_emb_norm > 1e-8:
                    fused_embedding_normalized = fused_embedding_pooled / fused_emb_norm
                else:
                    fused_embedding_normalized = fused_embedding_pooled
                
                fused_embedding_pooled_np = fused_embedding_normalized.detach().cpu().float().numpy().reshape(1, -1)
                
                similar_nodes = self._retrieve_similar_nodes_hnsw(fused_embedding_pooled_np, k=30)
                
                # 3. 构建超边（基于相似度）
                hyperedges = self._build_hyperedges_from_similarity(fused_embedding_pooled, similar_nodes)
                
                # 4. 多跳检索（使用构建的超边）
                initial_node_ids = [node_id for node_id, _ in similar_nodes[:5]]
                if len(initial_node_ids) == 0:
                    print(f"  Warning: No initial nodes found! similar_nodes={len(similar_nodes)}")
                    selected_nodes = []
                    hop_count = 0
                else:
                    # 将构建的超边传递给多跳检索函数（包含正交化，最多再正交2次）
                    selected_nodes, hop_count = self._multi_hop_retrieval(
                        fused_embedding_pooled,
                        initial_node_ids,
                        max_hops=self.max_hops,
                        query_hyperedges=hyperedges,  # 传递构建的超边
                        max_orthogonal_hops=2  # 除了第一次，最多再正交2次（总共3次）
                    )
                    if len(selected_nodes) == 0:
                        print(f"  Warning: _multi_hop_retrieval returned empty! initial_node_ids={len(initial_node_ids)}, hyperedges={len(hyperedges)}")
                    
                # 5. 召回三元组和knowledge（三元组中的 ehr+prompt 超过限制的会被跳过）
                triplets, knowledge_texts = self._retrieve_triplets_and_knowledge(selected_nodes)
                
                # 6. 使用三元组微调adapter（只微调一步，使用检索到的节点的unpooled embeddings）
                # 遍历所有三元组，对每个三元组进行微调
                adapter_loss = 0.0
                if triplets:
                    for triplet in triplets:
                        # 合并 ehr_texts
                        combined_ehr_text = "\n".join(triplet['ehr_texts'])
                        # 使用所有图像路径
                        triplet_image_paths = triplet['image_paths'] if triplet['image_paths'] else data['image_paths']
                        # 使用三元组的 report_text
                        triplet_report_text = triplet['report_text']
                        
                        # 对每个三元组进行微调
                        triplet_adapter_loss = self._fine_tune_adapter(
                            combined_ehr_text,
                            triplet_image_paths,
                            triplet_report_text,
                            selected_nodes=selected_nodes
                        )
                        adapter_loss += triplet_adapter_loss
                    adapter_loss = adapter_loss / len(triplets) if triplets else 0.0
                else:
                    # 如果没有三元组，使用当前样本微调
                    adapter_loss = self._fine_tune_adapter(
                        data['ehr_text'],
                        data['image_paths'],
                        data['report_text'],
                        selected_nodes=selected_nodes
                    )
                
                # 7. 生成报告（使用当前样本的 ehr+image，加上 knowledge）
                generated_report = self._generate_report(
                    data['ehr_text'],
                    data['image_paths'],
                    knowledge_texts
                )
                
                # 8. 计算真实报告的CE loss（使用微调后的adapter）
                # 构建输入：图像 + EHR + knowledge（如果有）+ prompt + report
                prompt = "Based on the above patient information and clinical history, Please generate a paragraph of radiology report for this chest X-ray image."
                text_content = data['ehr_text']
                if knowledge_texts and len(knowledge_texts) > 0:
                    knowledge_str = "\n".join(knowledge_texts)
                    text_content = data['ehr_text'] + "\n\nKnowledge:\n" + knowledge_str
                # 将报告也添加到输入中，用于计算loss
                full_text_content = text_content + "\n" + prompt + "\n" + data['report_text']
                
                # 检查总token数是否超过限制
                tokenized_full = self.tokenizer(
                    full_text_content,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=False
                )
                full_tokens = tokenized_full["input_ids"].shape[1]
                if full_tokens > self.max_tokens:
                    # 如果超过限制，跳过该样本
                    skipped_samples += 1
                    continue
                
                # 构建包含所有图像的消息
                content = []
                for img_path in data['image_paths']:
                    content.append({"type": "image", "image": img_path})
                content.append({"type": "text", "text": full_text_content})
                
                messages = [{
                    "role": "user",
                    "content": content
                }]
                
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, return_dict=True, return_tensors="pt"
                ).to(device)
                
                # 创建labels：只对报告部分计算loss
                input_ids = inputs["input_ids"]
                labels = input_ids.clone()
                labels.fill_(-100)  # 默认全部忽略
                
                # 正确找到报告在input_ids中的位置
                # 方法：tokenize prompt+report，然后在input_ids中查找匹配的位置
                prompt_text = text_content + "\n" + prompt + "\n"
                
                # tokenize prompt部分（不包含报告）
                prompt_ids = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=False
                )["input_ids"].to(device)
                
                # tokenize完整文本（包含报告）
                full_text_ids = self.tokenizer(
                    full_text_content,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=False
                )["input_ids"].to(device)
                
                # 在input_ids中找到prompt结束的位置（报告开始的位置）
                # 由于apply_chat_template可能添加特殊token，我们需要在input_ids中查找prompt_ids
                prompt_len = prompt_ids.shape[1]
                input_len = input_ids.shape[1]
                
                # 尝试在input_ids中找到prompt的结束位置
                # 简化方法：假设报告在最后，从后往前匹配
                report_ids_only = self.tokenizer(
                    data['report_text'],
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=512
                )["input_ids"].to(device)
                
                report_len = report_ids_only.shape[1]
                
                # 更准确的方法：在input_ids中查找report_ids的位置
                # 如果找不到精确匹配，则使用简化方法（假设在最后）
                found_match = False
                if report_len > 0 and report_len < input_len:
                    # 尝试在input_ids中查找report_ids
                    for start_idx in range(max(0, input_len - report_len - 10), input_len - report_len + 1):
                        if start_idx + report_len <= input_len:
                            candidate = input_ids[0, start_idx:start_idx + report_len]
                            # 检查是否匹配（允许部分匹配）
                            if torch.equal(candidate, report_ids_only[0, :min(report_len, candidate.shape[0])]):
                                labels[0, start_idx:start_idx + report_len] = report_ids_only[0, :min(report_len, candidate.shape[0])]
                                found_match = True
                                break
                
                # 如果没找到匹配，使用简化方法（假设报告在最后）
                if not found_match and report_len > 0 and report_len < input_len:
                    labels[0, -report_len:] = report_ids_only[0]
                
                # 计算真实报告的CE loss（需要参与反向传播）
                loss_outputs = self.vlm_model(
                    input_ids=input_ids,
                    pixel_values=inputs.get("pixel_values"),
                    labels=labels
                )
                real_ce_loss = loss_outputs.loss
                
                # 计算reward（在正交化检索之后，使用最终的selected_nodes）
                selected_node_dicts = []
                for n in selected_nodes:
                    # 确保embedding是tensor格式
                    if isinstance(n['embedding'], torch.Tensor):
                        emb = n['embedding']
                    else:
                        emb = torch.tensor(n['embedding'], device=device)
                    
                    # 确保embedding在正确的设备上
                    if emb.device != device:
                        emb = emb.to(device)
                    
                    # 确保embedding的dtype正确（float32）
                    if emb.dtype != torch.float32:
                        emb = emb.float()
                    
                    selected_node_dicts.append({
                        'embedding': emb,
                        'node_type': n.get('node_type', 'unknown')
                    })
                
                # 调试信息：检查selected_nodes是否为空
                if len(selected_node_dicts) == 0:
                    print(f"  Warning: No nodes selected! initial_nodes={len(initial_node_ids)}, similar_nodes={len(similar_nodes)}")
                
                reward = self.reward_fn.compute_reward(
                    fused_embedding_pooled,
                    selected_node_dicts,
                    hop_count,
                    depth=hop_count
                )
                
                # 记录召回节点信息（简单信息，不包含embedding）
                retrieved_nodes_info = []
                for node_info in selected_nodes:
                    node_id = node_info['node_id']
                    node = self.hypergraph.nodes.get(node_id)
                    if node:
                        retrieved_nodes_info.append({
                            'node_id': node_id,
                            'node_type': node_info.get('node_type', 'unknown'),
                            'has_metadata': bool(node.metadata) if node else False
                        })
                
                # Loss = -reward*10 + CE loss（最大化reward + 最小化CE loss）
                loss = -reward * 10 + real_ce_loss
                
                loss.backward()
                optimizer.step()
                
                # 记录损失和召回信息
                loss_value = loss.item()
                reward_value = reward.item() if isinstance(reward, torch.Tensor) else reward
                ce_loss_value = real_ce_loss.item() if isinstance(real_ce_loss, torch.Tensor) else real_ce_loss
                adapter_loss_value = adapter_loss if isinstance(adapter_loss, (int, float)) else adapter_loss
                
                epoch_loss += loss_value
                epoch_adapter_loss += adapter_loss_value
                epoch_ce_loss += ce_loss_value
                epoch_reward += reward_value
                num_samples += 1
                
                # 写入日志（不包含报告，只记录指标）
                log_entry = {
                    'epoch': epoch + 1,
                    'sample_idx': study_idx,
                    'loss': loss_value,
                    'adapter_loss': adapter_loss_value,
                    'ce_loss': ce_loss_value,
                    'reward': reward_value,
                    'hop_count': hop_count,
                    'num_retrieved_nodes': len(selected_nodes),
                    'num_knowledge_texts': len(knowledge_texts),
                    'num_triplets': len(triplets),
                    'retrieved_nodes': retrieved_nodes_info,
                    'num_tokens': data.get('num_tokens', 0)
                }
                
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            avg_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
            avg_adapter_loss = epoch_adapter_loss / num_samples if num_samples > 0 else 0.0
            avg_ce_loss = epoch_ce_loss / num_samples if num_samples > 0 else 0.0
            avg_reward = epoch_reward / num_samples if num_samples > 0 else 0.0
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Processed samples: {num_samples}")
            print(f"  Skipped samples (token > {self.max_tokens}): {skipped_samples}")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Average adapter loss: {avg_adapter_loss:.4f}")
            print(f"  Average CE loss: {avg_ce_loss:.4f}")
            print(f"  Average reward: {avg_reward:.4f}")
            
            # 记录epoch总结到日志
            epoch_summary = {
                'epoch': epoch + 1,
                'type': 'epoch_summary',
                'num_samples': num_samples,
                'skipped_samples': skipped_samples,
                'avg_loss': avg_loss,
                'avg_adapter_loss': avg_adapter_loss,
                'avg_ce_loss': avg_ce_loss,
                'avg_reward': avg_reward
            }
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(epoch_summary, ensure_ascii=False) + '\n')
            
            # 每轮保存一次checkpoint（只保存融合模型和RL模型）
            self._save_checkpoint(epoch + 1, avg_loss)
        
        print("Training completed!")
    
    def _save_checkpoint(self, epoch: int, loss: float):
        """保存checkpoint（只保存融合模型和RL模型）"""
        checkpoint_path = self.output_dir / f"model_f_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'fusion_state_dict': self.fusion.state_dict(),
            'rl_finder_state_dict': self.rl_finder.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Model F (RL)')
    parser.add_argument('--vlm_model_path', type=str, default=None)
    parser.add_argument('--hypergraph_path', type=str,
                        default='/mnt/sda/VLM/code/hypercode/hyperbuild/hypergraph')
    parser.add_argument('--adapter_checkpoint_dir', type=str,
                        default='/mnt/sda/VLM/code/hypercode/adatri/adamodel/adapter_epoch_6')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/sda/VLM/code/hypercode/modeltr/checkpoints')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--similarity_threshold', type=float, default=0.3)
    parser.add_argument('--max_hops', type=int, default=3)
    
    args = parser.parse_args()
    
    trainer = TrainModelF(
        vlm_model_path=args.vlm_model_path,
        hypergraph_path=args.hypergraph_path,
        adapter_checkpoint_dir=args.adapter_checkpoint_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        similarity_threshold=args.similarity_threshold,
        max_hops=args.max_hops
    )
    
    trainer.run()
