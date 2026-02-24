"""
Hypergraph Data Structure
超图数据结构：节点、超边、关联矩阵
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
import json


class NodeType(Enum):
    """节点类型"""
    EHR = "ehr"
    IMAGE = "image"
    REPORT = "report"  # 报告节点
    KNOWLEDGE = "knowledge"  # 知识节点（先验知识）
    MIXED = "mixed"  # 混合节点（临时节点，训练时使用，最终不保存）


@dataclass
class Node:
    """超图节点"""
    node_id: str
    node_type: NodeType
    embedding: torch.Tensor  # (embed_dim,)
    metadata: Dict  # 存储额外信息（如EHR数据、图像路径、报告等）
    
    def __post_init__(self):
        if isinstance(self.embedding, np.ndarray):
            self.embedding = torch.from_numpy(self.embedding)


@dataclass
class Hyperedge:
    """超边"""
    edge_id: str
    node_ids: List[str]  # 连接的节点ID列表
    weight: float = 1.0  # 超边权重
    edge_type: str = "similarity"  # 超边类型：3类 - "similarity_based", "id_based", "disease_based"
    metadata: Dict = None  # 存储额外信息
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Hypergraph:
    """
    超图数据结构
    维护节点集合、超边集合和关联矩阵H
    """
    def __init__(self, embed_dim: int = 1024):
        self.embed_dim = embed_dim
        self.nodes: Dict[str, Node] = {}  # {node_id: Node}
        self.hyperedges: Dict[str, Hyperedge] = {}  # {edge_id: Hyperedge}
        self.incidence_matrix: Optional[torch.Tensor] = None  # (|V|, |E|)
        self.node_id_to_idx: Dict[str, int] = {}  # 节点ID到矩阵索引的映射
        self.edge_id_to_idx: Dict[str, int] = {}  # 超边ID到矩阵索引的映射
        self._matrix_dirty: bool = False  # 标记矩阵是否需要全量重建
        
    def add_node(self, node: Node) -> str:
        """
        添加节点到超图
        Args:
            node: 节点对象
        Returns:
            node_id: 节点ID
        """
        if node.node_id in self.nodes:
            # 如果节点已存在，更新embedding
            self.nodes[node.node_id].embedding = node.embedding
            self.nodes[node.node_id].metadata.update(node.metadata)
            return node.node_id
        
        # 添加新节点
        self.nodes[node.node_id] = node
        
        # 增量更新关联矩阵：添加新行
        self._add_node_to_matrix(node.node_id)
        
        return node.node_id
    
    def add_hyperedge(self, hyperedge: Hyperedge) -> str:
        """
        添加超边到超图
        Args:
            hyperedge: 超边对象
        Returns:
            edge_id: 超边ID
        """
        # 验证所有节点都存在
        for node_id in hyperedge.node_ids:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} not found in hypergraph")
        
        if hyperedge.edge_id in self.hyperedges:
            # 如果超边已存在，更新（需要更新对应的列）
            old_hyperedge = self.hyperedges[hyperedge.edge_id]
            self.hyperedges[hyperedge.edge_id] = hyperedge
            # 更新该超边对应的列
            self._update_hyperedge_in_matrix(hyperedge.edge_id)
        else:
            # 添加新超边
            self.hyperedges[hyperedge.edge_id] = hyperedge
            # 增量更新关联矩阵：添加新列
            self._add_hyperedge_to_matrix(hyperedge.edge_id)
        
        return hyperedge.edge_id
    
    def _update_incidence_matrix(self, force_rebuild: bool = False):
        """
        更新关联矩阵 H（全量重建）
        H[i, j] = 节点i在超边j中的权重
        Args:
            force_rebuild: 是否强制全量重建（用于删除操作后）
        """
        num_nodes = len(self.nodes)
        num_edges = len(self.hyperedges)
        
        # 重新构建索引映射（确保索引连续且正确）
        self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.nodes.keys())}
        self.edge_id_to_idx = {edge_id: idx for idx, edge_id in enumerate(self.hyperedges.keys())}
        
        if num_nodes == 0 or num_edges == 0:
            self.incidence_matrix = torch.zeros((num_nodes, num_edges))
            self._matrix_dirty = False
            return
        
        # 初始化矩阵
        H = torch.zeros((num_nodes, num_edges))
        
        # 填充矩阵
        for edge_id, hyperedge in self.hyperedges.items():
            if edge_id not in self.edge_id_to_idx:
                continue
            edge_idx = self.edge_id_to_idx[edge_id]
            for node_id in hyperedge.node_ids:
                if node_id in self.node_id_to_idx:
                    node_idx = self.node_id_to_idx[node_id]
                    # 确保索引在有效范围内
                    if node_idx < num_nodes and edge_idx < num_edges:
                        # 权重可以是超边权重，也可以根据相似度计算
                        H[node_idx, edge_idx] = hyperedge.weight
        
        self.incidence_matrix = H
        self._matrix_dirty = False
    
    def _add_node_to_matrix(self, node_id: str):
        """
        增量更新：添加新节点到矩阵（添加新行）
        """
        if self.incidence_matrix is None:
            # 如果矩阵不存在，初始化
            self._update_incidence_matrix()
            return
        
        num_nodes = len(self.nodes)
        num_edges = len(self.hyperedges)
        
        # 更新节点索引映射（新节点在末尾）
        if node_id not in self.node_id_to_idx:
            self.node_id_to_idx[node_id] = num_nodes - 1
        
        # 确保矩阵列数与超边数一致
        current_cols = self.incidence_matrix.shape[1] if self.incidence_matrix.numel() > 0 else 0
        if current_cols < num_edges:
            # 需要扩展列数
            if self.incidence_matrix.numel() > 0:
                padding = torch.zeros((self.incidence_matrix.shape[0], num_edges - current_cols))
                self.incidence_matrix = torch.cat([self.incidence_matrix, padding], dim=1)
            else:
                self.incidence_matrix = torch.zeros((0, num_edges))
        
        # 在矩阵末尾添加新行（全零）
        new_row = torch.zeros((1, num_edges))
        if self.incidence_matrix.numel() == 0:
            self.incidence_matrix = new_row
        else:
            self.incidence_matrix = torch.cat([self.incidence_matrix, new_row], dim=0)
        
        # 更新该节点参与的所有超边
        for edge_id, hyperedge in self.hyperedges.items():
            if node_id in hyperedge.node_ids:
                self._update_matrix_entry(node_id, edge_id, hyperedge.weight)
    
    def _add_hyperedge_to_matrix(self, edge_id: str):
        """
        增量更新：添加新超边到矩阵（添加新列）
        """
        if self.incidence_matrix is None:
            # 如果矩阵不存在，初始化
            self._update_incidence_matrix()
            return
        
        num_nodes = len(self.nodes)
        num_edges = len(self.hyperedges)
        
        # 更新超边索引映射（新超边在末尾）
        if edge_id not in self.edge_id_to_idx:
            self.edge_id_to_idx[edge_id] = num_edges - 1
        
        hyperedge = self.hyperedges[edge_id]
        
        # 确保矩阵行数与节点数一致
        current_rows = self.incidence_matrix.shape[0] if self.incidence_matrix.numel() > 0 else 0
        if current_rows < num_nodes:
            # 需要扩展行数
            if self.incidence_matrix.numel() > 0:
                current_cols = self.incidence_matrix.shape[1]
                padding = torch.zeros((num_nodes - current_rows, current_cols))
                self.incidence_matrix = torch.cat([self.incidence_matrix, padding], dim=0)
            else:
                self.incidence_matrix = torch.zeros((num_nodes, 0))
        
        # 在矩阵末尾添加新列（全零）
        new_col = torch.zeros((num_nodes, 1))
        if self.incidence_matrix.numel() == 0:
            self.incidence_matrix = new_col
        else:
            # 添加新列
            self.incidence_matrix = torch.cat([self.incidence_matrix, new_col], dim=1)
        
        # 更新该超边连接的所有节点
        for node_id in hyperedge.node_ids:
            if node_id in self.node_id_to_idx:
                self._update_matrix_entry(node_id, edge_id, hyperedge.weight)
    
    def _update_hyperedge_in_matrix(self, edge_id: str):
        """
        增量更新：更新超边在矩阵中的值（更新整列）
        """
        if edge_id not in self.edge_id_to_idx or edge_id not in self.hyperedges:
            return
        
        hyperedge = self.hyperedges[edge_id]
        edge_idx = self.edge_id_to_idx[edge_id]
        
        # 先将该列清零
        if self.incidence_matrix is not None and edge_idx < self.incidence_matrix.shape[1]:
            self.incidence_matrix[:, edge_idx] = 0.0
        
        # 更新该超边连接的所有节点
        for node_id in hyperedge.node_ids:
            if node_id in self.node_id_to_idx:
                self._update_matrix_entry(node_id, edge_id, hyperedge.weight)
    
    def _update_matrix_entry(self, node_id: str, edge_id: str, weight: float):
        """
        更新矩阵中的单个元素
        """
        if (node_id not in self.node_id_to_idx or 
            edge_id not in self.edge_id_to_idx or 
            self.incidence_matrix is None):
            return
        
        node_idx = self.node_id_to_idx[node_id]
        edge_idx = self.edge_id_to_idx[edge_id]
        
        if (node_idx < self.incidence_matrix.shape[0] and 
            edge_idx < self.incidence_matrix.shape[1]):
            self.incidence_matrix[node_idx, edge_idx] = weight
    
    def get_connected_nodes(self, edge_id: str) -> List[str]:
        """获取超边连接的所有节点ID"""
        if edge_id not in self.hyperedges:
            return []
        return self.hyperedges[edge_id].node_ids
    
    def get_node_edges(self, node_id: str) -> List[str]:
        """获取节点参与的所有超边ID"""
        if node_id not in self.nodes:
            return []
        
        edge_ids = []
        for edge_id, hyperedge in self.hyperedges.items():
            if node_id in hyperedge.node_ids:
                edge_ids.append(edge_id)
        
        return edge_ids
    
    @classmethod
    def load(cls, json_path: str, load_unpooled: bool = False):
        """
        从JSON文件加载超图（流式加载，节省内存）
        Args:
            json_path: JSON文件路径
            load_unpooled: 是否加载embedding_unpooled（默认False，节省内存）
        """
        import ijson
        from pathlib import Path
        from tqdm import tqdm
        
        json_path = Path(json_path)
        
        # 显示文件信息
        file_size = json_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  Reading JSON file: {json_path.name} ({file_size:.1f} MB)")
        print("  Using stream parsing (ijson) to reduce memory usage...")
        
        hypergraph = None
        embed_dim = 1024
        file_size_bytes = json_path.stat().st_size
        
        print("  Parsing JSON structure (streaming)...")
        with open(json_path, 'rb') as f:
            parser = ijson.parse(f)
            current_path = None
            current_node_id = None
            current_node_data = {}
            current_edge_id = None
            current_edge_data = {}
            node_count = 0
            edge_count = 0
            in_nodes = False
            in_hyperedges = False
            
            # 估算总数（粗略）
            estimated_total = max(10000, int(file_size_bytes / (1024 * 100)))
            pbar = tqdm(total=estimated_total, desc="    Processing", mininterval=1.0, ncols=80)
            
            for prefix, event, value in parser:
                # 读取embed_dim
                if prefix == 'embed_dim' and event == 'number':
                    embed_dim = int(value)
                    if hypergraph is None:
                        hypergraph = cls(embed_dim=embed_dim)
                        hypergraph._json_path = str(json_path)
                        hypergraph._load_unpooled = load_unpooled
                
                # 处理nodes
                elif prefix.startswith('nodes.'):
                    if event == 'start_map' and '.' not in prefix[6:]:
                        # 新节点开始
                        current_node_id = prefix.split('.')[-1]
                        current_node_data = {}
                        in_nodes = True
                    elif in_nodes and current_node_id:
                        if prefix.endswith('.node_id'):
                            current_node_data['node_id'] = value
                        elif prefix.endswith('.node_type'):
                            current_node_data['node_type'] = value
                        elif prefix.endswith('.embedding'):
                            if event == 'start_array':
                                current_node_data['embedding'] = []
                            elif event == 'number':
                                current_node_data['embedding'].append(value)
                        elif prefix.endswith('.metadata'):
                            if event == 'end_map':
                                # 节点完成
                                if 'node_id' in current_node_data and 'embedding' in current_node_data:
                                    node_count += 1
                                    pbar.update(1)
                                    
                                    embedding = torch.tensor(current_node_data['embedding'], dtype=torch.float32)
                                    metadata = current_node_data.get('metadata', {})
                                    
                                    if 'embedding_unpooled' in metadata and not load_unpooled:
                                        metadata['_has_unpooled'] = True
                                        del metadata['embedding_unpooled']
                                    
                                    node = Node(
                                        node_id=current_node_data['node_id'],
                                        node_type=NodeType(current_node_data['node_type']),
                                        embedding=embedding,
                                        metadata=metadata
                                    )
                                    hypergraph.nodes[current_node_data['node_id']] = node
                                    hypergraph.node_id_to_idx[current_node_data['node_id']] = len(hypergraph.nodes) - 1
                                
                                current_node_id = None
                                current_node_data = {}
                                in_nodes = False
                
                # 处理hyperedges（类似处理）
                elif prefix.startswith('hyperedges.'):
                    if event == 'start_map' and '.' not in prefix[11:]:
                        current_edge_id = prefix.split('.')[-1]
                        current_edge_data = {}
                        in_hyperedges = True
                    elif in_hyperedges and current_edge_id:
                        if prefix.endswith('.edge_id'):
                            current_edge_data['edge_id'] = value
                        elif prefix.endswith('.node_ids'):
                            if event == 'start_array':
                                current_edge_data['node_ids'] = []
                            elif event == 'string':
                                current_edge_data['node_ids'].append(value)
                        elif prefix.endswith('.weight'):
                            current_edge_data['weight'] = value
                        elif prefix.endswith('.edge_type'):
                            current_edge_data['edge_type'] = value
                        elif prefix.endswith('.metadata'):
                            if event == 'end_map':
                                # 超边完成
                                if 'edge_id' in current_edge_data:
                                    edge_count += 1
                                    pbar.update(1)
                                    
                                    hyperedge = Hyperedge(
                                        edge_id=current_edge_data['edge_id'],
                                        node_ids=current_edge_data.get('node_ids', []),
                                        weight=current_edge_data.get('weight', 1.0),
                                        edge_type=current_edge_data.get('edge_type', 'similarity'),
                                        metadata=current_edge_data.get('metadata', {})
                                    )
                                    hypergraph.hyperedges[current_edge_data['edge_id']] = hyperedge
                                    hypergraph.edge_id_to_idx[current_edge_data['edge_id']] = len(hypergraph.hyperedges) - 1
                                
                                current_edge_id = None
                                current_edge_data = {}
                                in_hyperedges = False
            
            pbar.close()
        
        # 重建关联矩阵
        print("  Building incidence matrix...")
        hypergraph._update_incidence_matrix(force_rebuild=True)
        print(f"  Hypergraph loaded successfully: {node_count} nodes, {edge_count} hyperedges")
        
        return hypergraph
    
    def load_node_unpooled(self, node_id: str) -> Optional[torch.Tensor]:
        """
        按需加载特定节点的embedding_unpooled
        Args:
            node_id: 节点ID
        Returns:
            embedding_unpooled tensor，如果不存在或已加载则返回None
        """
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        
        # 如果已经加载，直接返回
        if 'embedding_unpooled' in node.metadata:
            return node.metadata['embedding_unpooled']
        
        # 如果标记为未加载，从JSON文件中加载
        if node.metadata.get('_has_unpooled', False):
            import json
            with open(self._json_path, 'r') as f:
                data = json.load(f)
            
            node_data = data.get('nodes', {}).get(node_id, {})
            metadata_data = node_data.get('metadata', {})
            
            if 'embedding_unpooled' in metadata_data:
                unpooled_value = metadata_data['embedding_unpooled']
                if isinstance(unpooled_value, list) and len(unpooled_value) > 0:
                    unpooled_tensor = torch.tensor(unpooled_value, dtype=torch.float32)
                    node.metadata['embedding_unpooled'] = unpooled_tensor
                    node.metadata.pop('_has_unpooled', None)  # 移除标记
                    return unpooled_tensor
        
        return None

