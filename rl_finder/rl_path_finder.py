"""
Reinforcement Learning Path Finder
强化学习路径寻找器：从超边中选择节点
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Set


class RLPathFinder(nn.Module):
    """
    强化学习路径寻找器
    使用简单的MLP选择激活的节点
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        max_hops: int = 3,
        similarity_threshold: float = 0.6,
        temperature: float = 0.01
    ):
        """
        Args:
            embed_dim: embedding维度
            hidden_dim: 隐藏层维度
            max_hops: 最大跳数
            similarity_threshold: 相似度阈值（用于构建超边）
            temperature: 温度参数，用于softmax（<1会让分布更尖锐，>1会让分布更平滑）
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_hops = max_hops
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        
        # 简单的MLP：用于计算节点激活概率
        # 输入：query_embedding + node_embedding
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # query + node
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出激活分数
        )
        
        # 值网络：估计状态价值
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 状态投影层：将状态embedding投影到hidden_dim（用于值网络）
        # 注意：这个层在forward中会被移动到正确的设备和dtype
        self._temp_state_proj = nn.Linear(1, hidden_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重：使用Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        candidate_nodes: List[Dict],
        debug: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：使用简单MLP计算节点激活概率
        Args:
            query_embedding: (embed_dim,) 查询embedding
            candidate_nodes: 候选节点列表，每个元素包含 {'node_id', 'embedding', 'node_type'}
            debug: 是否输出调试信息
        Returns:
            probs: (num_candidates,) 每个候选节点的激活概率（softmax后）
            value: (1,) 状态价值估计
            activation_scores: (num_candidates,) 激活分数（softmax前），用于KNOWLEDGE节点强制召回
        """
        if len(candidate_nodes) == 0:
            empty_tensor = torch.tensor([], device=query_embedding.device, dtype=query_embedding.dtype)
            return empty_tensor, torch.tensor([0.0], device=query_embedding.device), empty_tensor
        
        # 1. 构建节点特征矩阵（确保在正确的设备和dtype上）
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype  # 获取模型的dtype（可能是bfloat16）
        
        node_embeddings = torch.stack([
            torch.tensor(n['embedding'], device=device, dtype=dtype) if not isinstance(n['embedding'], torch.Tensor)
            else n['embedding'].to(device=device, dtype=dtype)
            for n in candidate_nodes
        ])  # (num_candidates, embed_dim)
        
        # 2. 确保query_embedding在正确的设备和dtype上
        query_embedding = query_embedding.to(device=device, dtype=dtype)
        
        # 3. 拼接query和node embedding
        query_expanded = query_embedding.unsqueeze(0).expand(len(candidate_nodes), -1)  # (num_candidates, embed_dim)
        combined = torch.cat([query_expanded, node_embeddings], dim=-1)  # (num_candidates, embed_dim * 2)
        
        # 4. 通过MLP计算激活分数
        activation_scores = self.mlp(combined).squeeze(-1)  # (num_candidates,)
        
        # 5. 使用温度缩放的softmax转换为概率分布
        # Temperature scaling: softmax(x / T)
        # T < 1: 让分布更尖锐，放大差异
        # T > 1: 让分布更平滑
        scaled_scores = activation_scores / self.temperature
        probs = F.softmax(scaled_scores, dim=0)  # (num_candidates,)
        
        # 6. 计算状态价值（基于激活分数的平均值）
        state_embedding = activation_scores.mean().unsqueeze(0).unsqueeze(0)  # (1, 1)
        # 投影到hidden_dim（确保在正确的设备和dtype上）
        state_proj = F.relu(self._temp_state_proj(state_embedding.to(device=device, dtype=dtype)))  # (1, hidden_dim)
        value = self.value_net(state_proj).squeeze(-1)  # (1,)
        
        if debug:
            # 输出激活分数（softmax前）的统计信息
            scores_cpu = activation_scores.detach().cpu().float().numpy()
            probs_cpu = probs.detach().cpu().float().numpy()
            print(f"    [DEBUG] 使用简单MLP，候选节点数={len(candidate_nodes)}")
            print(f"    [DEBUG] 激活分数（softmax前）:")
            print(f"      分数范围: [{scores_cpu.min():.4f}, {scores_cpu.max():.4f}]")
            print(f"      平均分数: {scores_cpu.mean():.4f}")
            print(f"      标准差: {scores_cpu.std():.4f}")
            print(f"      温度参数: {self.temperature}")
            print(f"    [DEBUG] 温度缩放后的概率分布:")
            print(f"      概率范围: [{probs_cpu.min():.4f}, {probs_cpu.max():.4f}]")
            print(f"      最大概率: {probs_cpu.max():.4f} (对应分数: {scores_cpu[np.argmax(probs_cpu)]:.4f})")
            if scores_cpu.std() < 0.01:
                print(f"      ⚠️ 警告：激活分数标准差很小，可能导致softmax后概率几乎相同")
        
        return probs, value, activation_scores  # 同时返回激活分数用于KNOWLEDGE节点强制召回
    
    def select_nodes(
        self,
        probs: torch.Tensor,
        activation_scores: torch.Tensor = None,
        top_k: int = None,
        prob_threshold: float = 0.01,
        debug: bool = False,
        candidate_nodes: List[Dict] = None
    ) -> List[int]:
        """
        根据概率选择节点（使用softmax概率分布）
        Args:
            probs: (num_candidates,) 激活概率（softmax后）
            activation_scores: (num_candidates,) 激活分数（softmax前），用于KNOWLEDGE节点强制召回
            top_k: 选择top k个节点（如果为None，则使用prob_threshold）
            prob_threshold: 概率阈值，选择概率 >= prob_threshold 的节点
            debug: 是否输出调试信息
            candidate_nodes: 候选节点列表（用于调试）
        Returns:
            selected_indices: 选中的节点索引列表
        """
        if len(probs) == 0:
            return []
        
        if debug:
            # 转换为float32再转numpy（因为numpy不支持bfloat16）
            probs_cpu = probs.detach().cpu().float().numpy()
            print(f"    [DEBUG] 节点激活概率分布:")
            print(f"      候选节点数: {len(probs)}")
            print(f"      概率范围: [{probs_cpu.min():.4f}, {probs_cpu.max():.4f}]")
            print(f"      平均概率: {probs_cpu.mean():.4f}")
            print(f"      中位数概率: {np.median(probs_cpu):.4f}")
            print(f"      概率阈值: {prob_threshold}")
            # 统计超过阈值的节点数
            above_threshold = (probs_cpu >= prob_threshold).sum()
            print(f"      超过阈值({prob_threshold})的节点数: {above_threshold}/{len(probs)}")
            # 输出前10个节点的概率（如果有candidate_nodes）
            if candidate_nodes and len(candidate_nodes) > 0:
                print(f"      前10个节点的概率:")
                top_indices = np.argsort(probs_cpu)[-10:][::-1]
                for idx in top_indices:
                    node_id = candidate_nodes[idx].get('node_id', f'node_{idx}')
                    node_type = candidate_nodes[idx].get('node_type', 'unknown')
                    print(f"        {node_id[:50]} ({node_type}): {probs_cpu[idx]:.4f}")
        
        # 选择策略：如果指定了top_k，选择top-k；否则严格使用概率阈值
        if top_k is not None:
            _, indices = torch.topk(probs, min(top_k, len(probs)))
            selected = indices.tolist()
        else:
            # 严格选择概率 >= prob_threshold 的节点
            selected = (probs >= prob_threshold).nonzero(as_tuple=True)[0].tolist()
        
        # 对于KNOWLEDGE节点，在softmax前选择一个激活分数最高的，强制召回
        if activation_scores is not None and candidate_nodes is not None:
            knowledge_indices = []
            for idx, node_info in enumerate(candidate_nodes):
                node_type = node_info.get('node_type', '').lower()  # 转换为小写进行比较
                if node_type == 'knowledge':
                    knowledge_indices.append(idx)
            
            if debug and len(knowledge_indices) == 0:
                # 输出所有候选节点的类型，用于调试
                all_types = set()
                for node_info in candidate_nodes:
                    all_types.add(node_info.get('node_type', 'unknown'))
                print(f"      [DEBUG] 候选节点中没有knowledge类型，所有类型: {all_types}")
            
            if len(knowledge_indices) > 0:
                # 找到激活分数最高的KNOWLEDGE节点
                knowledge_scores = activation_scores[knowledge_indices]
                best_knowledge_idx_in_list = torch.argmax(knowledge_scores).item()
                best_knowledge_idx = knowledge_indices[best_knowledge_idx_in_list]
                
                # 如果这个节点不在已选中的列表中，强制添加
                if best_knowledge_idx not in selected:
                    selected.append(best_knowledge_idx)
                    if debug:
                        node_id = candidate_nodes[best_knowledge_idx].get('node_id', f'node_{best_knowledge_idx}')
                        score = activation_scores[best_knowledge_idx].item()
                        print(f"      强制召回KNOWLEDGE节点: {node_id[:50]}, 激活分数={score:.4f}")
            elif debug:
                print(f"      [DEBUG] 候选节点中没有knowledge类型的节点")
        
        if debug:
            print(f"      最终选中节点数: {len(selected)}")
            # 输出所有选中节点的ID和类型（包括knowledge节点）
            if candidate_nodes and len(selected) > 0:
                print(f"      最终选中的节点ID:")
                # 确保probs_cpu可用
                probs_cpu_debug = probs.detach().cpu().float().numpy()
                for idx in selected:
                    node_id = candidate_nodes[idx].get('node_id', f'node_{idx}')
                    node_type = candidate_nodes[idx].get('node_type', 'unknown')
                    prob_val = probs_cpu_debug[idx] if idx < len(probs_cpu_debug) else 0.0
                    print(f"        {node_id[:60]} ({node_type}): 概率={prob_val:.4f}")
        
        return selected


class RewardFunction:
    """
    奖励函数：包含四个角度
    1. 寻找深度（不能太深）
    2. 寻找准确性（相似度）
    3. 寻找多样性（互相间相似度不能太高）
    4. hop数量（尽可能少轮次）
    """
    def __init__(
        self,
        depth_penalty_weight: float = 0.1,
        accuracy_weight: float = 0.4,
        diversity_weight: float = 0.3,
        hop_penalty_weight: float = 0.2,
        max_depth: int = 5
    ):
        """
        Args:
            depth_penalty_weight: 深度惩罚权重
            accuracy_weight: 准确性权重
            diversity_weight: 多样性权重
            hop_penalty_weight: hop数量惩罚权重
            max_depth: 最大深度
        """
        self.depth_penalty_weight = depth_penalty_weight
        self.accuracy_weight = accuracy_weight
        self.diversity_weight = diversity_weight
        self.hop_penalty_weight = hop_penalty_weight
        self.max_depth = max_depth
    
    def compute_reward(
        self,
        query_embedding: torch.Tensor,
        selected_nodes: List[Dict],
        hop_count: int,
        depth: int
    ) -> float:
        """
        计算奖励
        Args:
            query_embedding: (embed_dim,) 查询embedding
            selected_nodes: 选中的节点列表，每个元素包含 {'embedding', ...}
            hop_count: hop数量
            depth: 搜索深度
        Returns:
            reward: 奖励值
        """
        if len(selected_nodes) == 0:
            return -1.0  # 惩罚：没有选中任何节点
        
        # 1. 深度惩罚：深度越深，惩罚越大
        depth_penalty = -self.depth_penalty_weight * (depth / self.max_depth)
        
        # 2. 准确性奖励：与查询embedding的相似度
        # 确保embedding是tensor格式，并且在同一设备上
        node_emb_list = []
        for n in selected_nodes:
            emb = n['embedding']
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, device=query_embedding.device)
            elif emb.device != query_embedding.device:
                emb = emb.to(query_embedding.device)
            node_emb_list.append(emb)
        
        node_embeddings = torch.stack(node_emb_list)  # (num_selected, embed_dim)
        query_expanded = query_embedding.unsqueeze(0).expand(len(selected_nodes), -1)  # (num_selected, embed_dim)
        similarities = F.cosine_similarity(query_expanded, node_embeddings, dim=1)  # (num_selected,)
        accuracy_reward = self.accuracy_weight * similarities.mean().item()
        
        # 3. 多样性奖励：节点间相似度越低越好
        if len(selected_nodes) > 1:
            # 计算所有节点对之间的相似度
            pairwise_similarities = []
            for i in range(len(selected_nodes)):
                for j in range(i + 1, len(selected_nodes)):
                    sim = F.cosine_similarity(
                        node_embeddings[i:i+1],
                        node_embeddings[j:j+1],
                        dim=1
                    ).item()
                    pairwise_similarities.append(sim)
            avg_pairwise_sim = np.mean(pairwise_similarities) if pairwise_similarities else 0.0
            diversity_reward = self.diversity_weight * (1.0 - avg_pairwise_sim)  # 相似度越低，奖励越高
        else:
            diversity_reward = 0.0
        
        # 4. hop数量惩罚：hop越少越好
        hop_penalty = -self.hop_penalty_weight * (hop_count / 5.0)  # 假设最大5个hops
        
        # 总奖励
        reward = depth_penalty + accuracy_reward + diversity_reward + hop_penalty
        
        return reward

