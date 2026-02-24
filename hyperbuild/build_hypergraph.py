"""
使用2%训练数据构建超图
使用HNSW优化构建速度
"""

import torch
import torch.nn as nn
import numpy as np
import json
import random
import math
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm
import hnswlib
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入本地模块（使用绝对导入，从当前目录）
import sys
from pathlib import Path
_current_dir = Path(__file__).parent
sys.path.insert(0, str(_current_dir))

from hypergraph import Hypergraph, Node, NodeType, Hyperedge
from data_loader import DataLoader
from embed_model import EmbedModel


class HypergraphBuilder:
    """
    超图构建器
    使用2%的训练数据构建超图，利用HNSW优化构建速度
    """
    def __init__(
        self,
        model: EmbedModel,
        embed_dim: int = 1024,
        k: int = 5,  # 相似度检索的top-k
        train_sample_ratio: float = 0.02,  # 2%的训练数据
        batch_size: int = 8  # 批量处理大小，增大可提升速度
    ):
        """
        Args:
            model: EmbedModel模型，用于生成embedding
            embed_dim: embedding维度
            k: 相似度检索的top-k
            train_sample_ratio: 训练数据采样比例（2%）
            batch_size: 批量处理大小，用于加速embedding生成
        """
        self.model = model
        self.embed_dim = embed_dim
        self.k = k
        self.batch_size = batch_size
        
        # 初始化超图
        self.hypergraph = Hypergraph(embed_dim=embed_dim)
        
        # HNSW向量索引：用于快速检索相似节点
        self._vector_indices = {}  # {NodeType: HNSW index}
        self._node_id_mappings = {}  # {NodeType: [node_ids]} 索引位置到node_id的映射
        self._index_built = False
        
        # 数据加载器（使用2%的训练数据）
        self.data_loader = DataLoader(train_sample_ratio=train_sample_ratio)
        
        # 设备
        self.device = next(self.model.model.parameters()).device
    
    def build(self, output_path: Optional[str] = None, max_samples: Optional[int] = None):
        """
        构建超图
        Args:
            output_path: 保存超图的路径（可选）
            max_samples: 最大样本数量（可选，None表示使用全部数据）
        """
        print("=" * 80)
        if max_samples:
            print(f"Building hypergraph (using at most {max_samples} samples)")
        else:
            print("Building hypergraph (using 2% training data)")
        print("=" * 80)
        
        # 1. 加载训练数据
        print("\n[Step 1] Loading training data...")
        train_studies = self.data_loader.get_train_studies()
        if max_samples and max_samples > 0:
            train_studies = train_studies[:max_samples]
        print(f"Loaded {len(train_studies)} training samples")
        
        # 2. 添加节点并构建id_based超边
        print("\n[Step 2] Adding nodes and building id_based hyperedges...")
        self._add_nodes_and_id_based_edges(train_studies)
        
        # 输出各类型节点数量（剪枝前）
        print("\nNode count by type (before pruning):")
        node_counts_by_type = defaultdict(int)
        for node in self.hypergraph.nodes.values():
            node_counts_by_type[node.node_type] += 1
        for node_type, count in sorted(node_counts_by_type.items(), key=lambda x: x[0].value):
            print(f"  {node_type.value}: {count} nodes")
        
        # 2.6. EHR节点剪枝：相同类型+相同ID+相似度>0.9的节点进行合并
        print("\n[Step 2.6] Pruning EHR nodes...")
        self._prune_ehr_nodes()
        
        # 输出各类型节点数量（剪枝后）
        print("\nNode count by type (after pruning):")
        node_counts_by_type = defaultdict(int)
        for node in self.hypergraph.nodes.values():
            node_counts_by_type[node.node_type] += 1
        for node_type, count in sorted(node_counts_by_type.items(), key=lambda x: x[0].value):
            print(f"  {node_type.value}: {count} nodes")
        
        # 3. 构建HNSW索引
        print("\n[Step 3] Building HNSW vector indices...")
        self._build_vector_indices()
        
        # 4. 构建similarity_based超边
        print("\n[Step 4] Building similarity_based hyperedges...")
        self._build_similarity_based_edges()
        
        # 5. 构建disease_based超边
        print("\n[Step 5] Building disease_based hyperedges...")
        self._build_disease_based_edges()
        
        # 6. 保存超图
        if output_path:
            print(f"\n[Step 6] Saving hypergraph to {output_path}...")
            self._save_hypergraph(output_path)
        
        print("\n" + "=" * 80)
        print("Hypergraph construction completed!")
        print(f"Number of nodes: {len(self.hypergraph.nodes)}")
        print(f"Number of hyperedges: {len(self.hypergraph.hyperedges)}")
        print("=" * 80)
        
        return self.hypergraph
    
    def _add_nodes_and_id_based_edges(self, studies: List[Dict]):
        """添加节点并构建id_based超边（流式处理以节省内存）"""
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
        
        # 流式处理：按study分组，处理完一个study就释放内存
        print("  Processing studies and creating nodes (streaming mode)...")
        import gc
        for study_idx, study in enumerate(tqdm(studies, desc="  Processing studies")):
            ehr_data = self.data_loader.get_ehr_data(study)
            image_paths = self.data_loader.get_image_paths(study)
            report = self.data_loader.get_report_text(study)
            subject_id = study.get('subject_id')
            study_id = study.get('study_id')
            
            if not ehr_data or not image_paths:
                continue
            
            sample_node_ids = []
            
            # 1. 批量处理当前study的EHR字段
            ehr_items = []  # (field_name, field_text, ehr_data)
            for field_name in ehr_field_names:
                if field_name in ehr_data and ehr_data[field_name]:
                    field_text = self._format_ehr_field(field_name, ehr_data[field_name])
                    if field_text:
                        ehr_items.append((field_name, field_text, ehr_data[field_name]))
            
            if ehr_items:
                # 批量处理EHR embedding
                ehr_texts = [item[1] for item in ehr_items]
                batch_embeds = self.model.batch_ehr_embed(ehr_texts)
                
                # 立即创建节点并释放
                for j, (field_name, field_text, field_ehr_data) in enumerate(ehr_items):
                    field_embeds = batch_embeds[j]  # 在CUDA上
                    # 进行pooling并移到CPU，并转换为float16以节省内存
                    field_embedding = self.model.embed_pooling(field_embeds, dim=1).squeeze(0).cpu().half()
                    # 存储未pooling的embedding（转换为float16以节省内存）
                    field_embedding_unpooled = field_embeds.squeeze(0).cpu().half()
                    
                    # 立即释放 CUDA tensor
                    del field_embeds
                    
                    ehr_node_id = f"ehr_{subject_id}_{study_id}_{field_name}"
                    ehr_node = Node(
                        node_id=ehr_node_id,
                        node_type=NodeType.EHR,
                        embedding=field_embedding,
                        metadata={
                            'field_name': field_name,
                            'subject_id': subject_id,
                            'study_id': study_id,
                            'embedding_unpooled': field_embedding_unpooled,
                            'ehr_text': field_text  # 保存原始EHR文本
                        }
                    )
                    self.hypergraph.add_node(ehr_node)
                    sample_node_ids.append(ehr_node_id)
                    
                    # 释放临时变量
                    del field_embedding, field_embedding_unpooled, ehr_node
                
                # 释放batch embedding和临时变量
                del batch_embeds, ehr_items, ehr_texts
            
            # 2. 批量处理当前study的图像
            if image_paths:
                batch_embeds = self.model.batch_image_embed(image_paths)
                
                # 立即创建节点并释放
                for img_idx, image_path in enumerate(image_paths):
                    image_embeds = batch_embeds[img_idx]  # 在CUDA上
                    # 进行pooling并移到CPU，并转换为float16以节省内存
                    image_embedding = self.model.embed_pooling(image_embeds, dim=1).squeeze(0).cpu().half()
                    # 存储未pooling的embedding（转换为float16以节省内存）
                    image_embedding_unpooled = image_embeds.squeeze(0).cpu().half()
                    
                    # 立即释放 CUDA tensor
                    del image_embeds
                    
                    image_node_id = f"image_{subject_id}_{study_id}_{img_idx}"
                    image_node = Node(
                        node_id=image_node_id,
                        node_type=NodeType.IMAGE,
                        embedding=image_embedding,
                        metadata={
                            'image_path': image_path,
                            'subject_id': subject_id,
                            'study_id': study_id,
                            'image_idx': img_idx,
                            'embedding_unpooled': image_embedding_unpooled
                        }
                    )
                    self.hypergraph.add_node(image_node)
                    sample_node_ids.append(image_node_id)
                    
                    # 释放临时变量
                    del image_embedding, image_embedding_unpooled, image_node
                
                # 释放batch embedding
                del batch_embeds
            
            # 3. 处理当前study的报告
            if report:
                report_embeds = self.model.ehr_embed(report)  # 在CUDA上
                # 进行pooling并移到CPU，并转换为float16以节省内存
                report_embedding = self.model.embed_pooling(report_embeds, dim=1).squeeze(0).cpu().half()
                # 存储未pooling的embedding（转换为float16以节省内存）
                report_embedding_unpooled = report_embeds.squeeze(0).cpu().half()
                
                # 立即释放 CUDA tensor
                del report_embeds
                
                report_node_id = f"report_{subject_id}_{study_id}"
                report_node = Node(
                    node_id=report_node_id,
                    node_type=NodeType.REPORT,
                    embedding=report_embedding,
                    metadata={
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'embedding_unpooled': report_embedding_unpooled,
                        'report_text': report  # 保存原始报告文本
                    }
                )
                self.hypergraph.add_node(report_node)
                sample_node_ids.append(report_node_id)
                
                # 释放临时变量
                del report_embedding, report_embedding_unpooled, report_node
            
            # 4. 构建id_based超边（连接同一study的所有节点）
            if len(sample_node_ids) >= 2:
                self._create_id_based_hyperedge(
                    node_ids=sample_node_ids,
                    id_type='study',
                    id_value=f"{subject_id}_{study_id}"
                )
            
            # 4.5. 创建三元组（或多节点组）：将同一study的EHR节点、图像节点、报告节点组合
            # 三元组可能包含多个节点（因为EHR可能有多个field，图像可能有多个）
            ehr_node_ids = [nid for nid in sample_node_ids if nid.startswith('ehr_')]
            image_node_ids = [nid for nid in sample_node_ids if nid.startswith('image_')]
            report_node_ids = [nid for nid in sample_node_ids if nid.startswith('report_')]
            
            # 如果有EHR、图像和报告节点，创建三元组
            if ehr_node_ids and image_node_ids and report_node_ids:
                # 将三元组信息保存到每个节点的metadata中
                triplet_node_ids = ehr_node_ids + image_node_ids + report_node_ids
                for node_id in triplet_node_ids:
                    node = self.hypergraph.nodes[node_id]
                    if 'triplets' not in node.metadata:
                        node.metadata['triplets'] = []
                    # 保存三元组信息：包含所有相关节点ID
                    node.metadata['triplets'].append({
                        'ehr_node_ids': ehr_node_ids,
                        'image_node_ids': image_node_ids,
                        'report_node_ids': report_node_ids,
                        'subject_id': subject_id,
                        'study_id': study_id
                    })
            
            # 释放当前study的所有临时变量
            del ehr_data, image_paths, report, sample_node_ids
            
            # 每处理100个study进行一次垃圾回收，防止内存累积
            if (study_idx + 1) % 100 == 0:
                gc.collect()
        
        # 5. 添加KNOWLEDGE节点
        print("\n[Step 2.5] Adding KNOWLEDGE nodes...")
        self._add_knowledge_nodes()
    
    def _format_ehr_field(self, field_name: str, field_data: Dict) -> str:
        """格式化EHR字段为文本"""
        if not field_data:
            return ""
        
        if field_name == 'patient_info':
            # 格式化患者基本信息
            info = field_data
            parts = []
            if 'gender' in info:
                parts.append(f"Gender: {info['gender']}")
            if 'anchor_age' in info:
                parts.append(f"Age: {info['anchor_age']}")
            if 'anchor_year_group' in info:
                parts.append(f"Year Group: {info['anchor_year_group']}")
            return " | ".join(parts)
        elif field_name in ['admissions', 'all_diagnoses', 'all_procedures', 
                           'prescriptions', 'all_labevents', 'microbiologyevents',
                           'transfers', 'icustays', 'chartevents', 'inputevents', 
                           'outputevents']:
            # 格式化列表类型的数据
            if not isinstance(field_data, list):
                return ""
            
            formatted_items = []
            for item in field_data:  # 使用完整数据
                if isinstance(item, dict):
                    # 将字典转换为键值对字符串
                    item_str = ", ".join([f"{k}: {v}" for k, v in item.items() if v is not None])
                    formatted_items.append(item_str)
                else:
                    formatted_items.append(str(item))
            
            return " | ".join(formatted_items)
        else:
            # 默认转换为字符串
            return str(field_data)
    
    def _create_id_based_hyperedge(self, node_ids: List[str], id_type: str, id_value: Optional[str] = None):
        """创建基于ID的超边"""
        node_ids = list(dict.fromkeys(node_ids))  # 去重
        if len(node_ids) < 2:
            return
        
        edge_id = f"id_{id_type}_{len(self.hypergraph.hyperedges)}"
        metadata = {'id_type': id_type}
        if id_value:
            metadata['id_value'] = id_value
        
        hyperedge = Hyperedge(
            edge_id=edge_id,
            node_ids=node_ids,
            weight=1.0,
            edge_type="id_based",
            metadata=metadata
        )
        self.hypergraph.add_hyperedge(hyperedge)
    
    def _add_knowledge_nodes(self):
        """添加KNOWLEDGE节点：从knowledge.yaml读取，使用冒号前的病种名称进行ehr_embed"""
        knowledge_file = Path(__file__).parent / "hypergraph" / "knowledge.yaml"
        print(f"  Looking for knowledge file at: {knowledge_file}")
        print(f"  File exists: {knowledge_file.exists()}")
        
        if not knowledge_file.exists():
            print(f"  Knowledge file not found: {knowledge_file}")
            # 尝试其他可能的路径
            alt_paths = [
                Path(__file__).parent.parent / "hyperbuild" / "hypergraph" / "knowledge.yaml",
                Path("/mnt/sda/VLM/code/hypercode/hyperbuild/hypergraph/knowledge.yaml"),
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    print(f"  Found knowledge file at alternative path: {alt_path}")
                    knowledge_file = alt_path
                    break
            else:
                print("  No knowledge file found")
                return
        
        # 读取knowledge.yaml
        print(f"  Reading knowledge file: {knowledge_file}")
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            knowledge_data = yaml.safe_load(f)
        
        if not knowledge_data:
            print("  No knowledge data found in file")
            return
        
        print(f"  Found {len(knowledge_data)} knowledge entries in YAML file")
        
        knowledge_count = 0
        for disease_name, description in tqdm(knowledge_data.items(), desc="  Adding knowledge nodes"):
            # 使用冒号前的病种名称（一个单词）进行ehr_embed
            # disease_name已经是冒号前的部分（如"Cardiomegaly"）
            disease_text = disease_name.strip()
            
            if not disease_text:
                continue
            
            # 生成embedding（使用ehr_embed方法，只用病种名称如"Cardiomegaly"）
            disease_embeds = self.model.ehr_embed(disease_text)  # (1, seq_len, embed_dim)
            # 生成pooling后的embedding（用于搜索），转换为float16并移到CPU以节省内存
            disease_embedding = self.model.embed_pooling(disease_embeds, dim=1).squeeze(0).cpu().half()  # (embed_dim,)
            # 存储未pooling的embedding（转换为float16以节省内存）
            disease_embedding_unpooled = disease_embeds.squeeze(0).cpu().half()
            # 清理显存
            del disease_embeds
            
            # 创建KNOWLEDGE节点
            knowledge_node_id = f"knowledge_{disease_name}"
            knowledge_node = Node(
                node_id=knowledge_node_id,
                node_type=NodeType.KNOWLEDGE,
                embedding=disease_embedding,  # 已经在CPU上，float16格式
                metadata={
                    'disease_name': disease_name,
                    'description': description.strip() if description else '',  # 存储原始描述数据
                    'embedding_unpooled': disease_embedding_unpooled
                }
            )
            self.hypergraph.add_node(knowledge_node)
            knowledge_count += 1
        
        print(f"  Added {knowledge_count} KNOWLEDGE nodes")
        
        # 验证节点是否真的被添加
        knowledge_nodes_in_graph = [
            (node_id, node) for node_id, node in self.hypergraph.nodes.items()
            if node.node_type == NodeType.KNOWLEDGE
        ]
        print(f"  Verified: Found {len(knowledge_nodes_in_graph)} KNOWLEDGE nodes in hypergraph")
        if len(knowledge_nodes_in_graph) > 0:
            print(f"  Knowledge node IDs: {[nid for nid, _ in knowledge_nodes_in_graph[:5]]}...")
    
    def _prune_ehr_nodes(self):
        """
        EHR节点剪枝：如果类型一样（field_name相同）+ ID一样（subject_id相同）+ study_id相同 + 相似度>0.95，则合并
        优化：按study_id分组，只对同一study_id的节点进行剪枝
        """
        ehr_nodes = [
            (node_id, node) for node_id, node in self.hypergraph.nodes.items()
            if node.node_type == NodeType.EHR
        ]
        
        if len(ehr_nodes) <= 1:
            print("  No EHR nodes to prune")
            return
        
        # 按study_id分组（不同study_id不需要剪枝）
        nodes_by_study = defaultdict(list)
        for node_id, node in ehr_nodes:
            study_id = node.metadata.get('study_id')
            nodes_by_study[study_id].append((node_id, node))
        
        # 对每个study_id内的节点进行剪枝
        nodes_to_remove = set()
        merged_count = 0
        
        for study_id, study_nodes in tqdm(nodes_by_study.items(), desc="  Pruning by study_id"):
            if len(study_nodes) <= 1:
                continue
            
            # 按field_name和subject_id分组（同一study_id内）
            nodes_by_field_subject = defaultdict(list)
            for node_id, node in study_nodes:
                field_name = node.metadata.get('field_name', 'unknown')
                subject_id = node.metadata.get('subject_id')
                key = (field_name, subject_id)
                nodes_by_field_subject[key].append((node_id, node))
            
            # 对每个(field_name, subject_id)组合的节点进行相似度检查和合并
            for (field_name, subject_id), field_nodes in nodes_by_field_subject.items():
                if len(field_nodes) <= 1:
                    continue
                
                # 计算节点之间的相似度
                processed = set()
                for i, (node_id_i, node_i) in enumerate(field_nodes):
                    if node_id_i in processed or node_id_i in nodes_to_remove:
                        continue
                    
                    embedding_i = node_i.embedding.cpu().float()
                    
                    # 找到与node_i相似度>0.95的节点
                    similar_nodes = []
                    for j, (node_id_j, node_j) in enumerate(field_nodes):
                        if i >= j or node_id_j in processed or node_id_j in nodes_to_remove:
                            continue
                        
                        embedding_j = node_j.embedding.cpu().float()
                        
                        # 计算余弦相似度
                        similarity = torch.cosine_similarity(
                            embedding_i.unsqueeze(0),
                            embedding_j.unsqueeze(0),
                            dim=1
                        ).item()
                        
                        if similarity > 0.95:  # 相似度阈值提高到0.95
                            similar_nodes.append((node_id_j, node_j, similarity))
                    
                    # 如果有相似节点，合并它们（保留第一个，删除其他的）
                    if similar_nodes:
                        # 将所有相似节点的metadata合并到node_i
                        for node_id_j, node_j, sim in similar_nodes:
                            # 合并metadata（保留更完整的信息）
                            for key, value in node_j.metadata.items():
                                if key not in node_i.metadata or node_i.metadata[key] is None:
                                    node_i.metadata[key] = value
                            
                            # 标记为待删除
                            nodes_to_remove.add(node_id_j)
                            processed.add(node_id_j)
                            merged_count += 1
                    
                    processed.add(node_id_i)
        
        # 删除被合并的节点
        if nodes_to_remove:
            print(f"  Merged {merged_count} duplicate EHR nodes")
            for node_id in nodes_to_remove:
                if node_id in self.hypergraph.nodes:
                    # 删除包含该节点的超边
                    edges_to_remove = []
                    for edge_id, edge in self.hypergraph.hyperedges.items():
                        if node_id in edge.node_ids:
                            # 从超边中移除该节点
                            edge.node_ids = [nid for nid in edge.node_ids if nid != node_id]
                            # 如果超边只剩下一个节点，删除该超边
                            if len(edge.node_ids) < 2:
                                edges_to_remove.append(edge_id)
                    
                    # 删除无效的超边
                    for edge_id in edges_to_remove:
                        del self.hypergraph.hyperedges[edge_id]
                        if edge_id in self.hypergraph.edge_id_to_idx:
                            del self.hypergraph.edge_id_to_idx[edge_id]
                    
                    # 删除节点
                    del self.hypergraph.nodes[node_id]
                    if node_id in self.hypergraph.node_id_to_idx:
                        del self.hypergraph.node_id_to_idx[node_id]
            
            # 重建关联矩阵
            self.hypergraph._update_incidence_matrix()
        else:
            print(f"  No duplicate EHR nodes found")
    
    def _build_vector_indices(self):
        """构建HNSW向量索引（使用pooling后的embedding）"""
        # 按照三种组合方式构建索引：
        # 1. ehr+image分层 - EHR和IMAGE节点一起
        # 2. report+knowledge分层 - REPORT和KNOWLEDGE节点一起
        # 3. ehr+image+knowledge+report分层 - 所有节点一起
        
        # 收集各类型节点
        ehr_image_nodes = []
        ehr_image_node_ids = []
        report_knowledge_nodes = []
        report_knowledge_node_ids = []
        all_nodes = []
        all_node_ids = []
        
        for node_id, node in self.hypergraph.nodes.items():
            # 使用pooling后的embedding构建索引（用于搜索）
            # 先转换为float32，再转为numpy（处理bfloat16类型）
            embedding = node.embedding.cpu().float().numpy().astype('float32')
            
            # 1. ehr+image分层
            if node.node_type in [NodeType.EHR, NodeType.IMAGE]:
                ehr_image_nodes.append(embedding)
                ehr_image_node_ids.append(node_id)
            
            # 2. report+knowledge分层
            if node.node_type in [NodeType.REPORT, NodeType.KNOWLEDGE]:
                report_knowledge_nodes.append(embedding)
                report_knowledge_node_ids.append(node_id)
            
            # 3. ehr+image+knowledge+report分层（所有节点）
            all_nodes.append(embedding)
            all_node_ids.append(node_id)
        
        # 构建三个HNSW索引
        indices_config = [
            ('ehr+image', ehr_image_nodes, ehr_image_node_ids),
            ('report+knowledge', report_knowledge_nodes, report_knowledge_node_ids),
            ('ehr+image+knowledge+report', all_nodes, all_node_ids)
        ]
        
        for index_name, embeddings, node_ids in indices_config:
            if len(embeddings) == 0:
                continue
            
            embeddings_array = np.array(embeddings).astype('float32')
            embed_dim = embeddings_array.shape[1]
            num_nodes = len(embeddings_array)
            
            # 设置max_elements为当前节点数的2倍，预留增长空间
            index = hnswlib.Index(space='cosine', dim=embed_dim)
            index.init_index(max_elements=max(num_nodes * 2, 1000), ef_construction=200, M=16)
            index.add_items(embeddings_array, np.arange(num_nodes))
            index.set_ef(50)
            
            self._vector_indices[index_name] = index
            self._node_id_mappings[index_name] = node_ids
        
        self._index_built = True
        print(f"HNSW index construction completed, containing {len(self._vector_indices)} index types")
        
        # 输出每个索引的层数信息
        print("\nHNSW index layer information:")
        for index_name, index in self._vector_indices.items():
            num_nodes = len(self._node_id_mappings[index_name])
            # HNSW层数估算：log2(N) + 1
            if num_nodes > 0:
                estimated_layers = int(math.log2(num_nodes)) + 1
                print(f"  {index_name}: {num_nodes} nodes → estimated {estimated_layers} layers")
            else:
                print(f"  {index_name}: 0 nodes")
    
    def _retrieve_similar_nodes(
        self,
        query_node_id: str,
        index_name: str,  # 'report', 'image+ehr', or 'image+ehr+report'
        top_k: int = 10,
        exclude_nodes: List[str] = None,
        threshold: float = 0.8,
    ) -> List[str]:
        """使用HNSW快速检索相似节点（使用pooling后的embedding）"""
        if query_node_id not in self.hypergraph.nodes:
            return []
        
        if exclude_nodes is None:
            exclude_nodes = []
        
        if index_name not in self._vector_indices:
            return [query_node_id]
        
        query_node = self.hypergraph.nodes[query_node_id]
        # 使用pooling后的embedding进行搜索
        query_embedding = query_node.embedding.cpu().float().numpy().astype('float32').reshape(1, -1)
        
        index = self._vector_indices[index_name]
        node_id_mapping = self._node_id_mappings[index_name]
        
        labels, distances = index.knn_query(query_embedding, k=min(top_k, len(node_id_mapping)))
        similarities = 1 - distances[0]
        indices = labels[0]
        
        similar_node_ids = [query_node_id]
        for i, idx in enumerate(indices):
            if similarities[i] < threshold:
                continue
            node_id = node_id_mapping[idx]
            if node_id != query_node_id and node_id not in exclude_nodes:
                similar_node_ids.append(node_id)
        
        return similar_node_ids
    
    def _build_similarity_based_edges(self):
        """构建基于相似度的超边（只处理EHR和IMAGE节点）"""
        # 使用HNSW贪心策略：统一处理EHR和IMAGE节点，使用ehr+image索引
        nodes_of_type = [
            (node_id, node) for node_id, node in self.hypergraph.nodes.items()
            if node.node_type in [NodeType.EHR, NodeType.IMAGE]
        ]
        
        print(f"  Building similarity hyperedges for ehr+image nodes...")
        
        # 使用HNSW贪心策略构建超边
        self._build_similarity_edges_with_hnsw_greedy(nodes_of_type, 'ehr+image')
    
    def _build_similarity_edges_with_hnsw_greedy(self, nodes_of_type: List[Tuple[str, Node]], index_name: str):
        """
        使用HNSW贪心策略构建相似度超边
        策略：每次只取top K去计算相似度，看那些点相连的下一层的topK，最后链接那些大于阈值的点
        """
        if index_name not in self._vector_indices:
            return
        
        index = self._vector_indices[index_name]
        node_id_mapping = self._node_id_mappings[index_name]
        
        # 构建node_id到索引位置的映射
        node_id_to_idx = {}
        for idx, node_id in enumerate(node_id_mapping):
            node_id_to_idx[node_id] = idx
        
        processed = set()
        # 根据索引类型设置阈值：report+knowledge使用0.7，其他使用0.8
        threshold = 0.7 if 'report' in index_name else 0.8
        
        # 根据索引类型确定进度条描述
        if 'report' in index_name:
            desc = "  report+knowledge"
        else:
            desc = "  ehr+image"
        for node_id, node in tqdm(nodes_of_type, desc=desc):
            if node_id in processed or node_id not in node_id_to_idx:
                continue
            
            # 找到同一sample的节点（排除）
            sample_nodes = []
            for edge_id, edge in self.hypergraph.hyperedges.items():
                if edge.edge_type == "id_based" and node_id in edge.node_ids:
                    sample_nodes.extend(edge.node_ids)
            sample_nodes = set(sample_nodes)
            
            # 使用HNSW检索top K相似节点
            query_embedding = node.embedding.cpu().float().numpy().astype('float32').reshape(1, -1)
            query_idx = node_id_to_idx[node_id]
            
            # 第一步：获取top K相似节点
            top_k = self.k * 2
            labels, distances = index.knn_query(query_embedding, k=min(top_k, len(node_id_mapping)))
            similarities = 1 - distances[0]
            top_k_indices = labels[0]
            
            # 收集相似节点（相似度>阈值且不在同一sample）
            similar_node_ids = [node_id]
            top_k_node_ids = set()
            
            for i, idx in enumerate(top_k_indices):
                if similarities[i] < threshold:
                    continue
                candidate_node_id = node_id_mapping[idx]
                if candidate_node_id != node_id and candidate_node_id not in sample_nodes:
                    similar_node_ids.append(candidate_node_id)
                    top_k_node_ids.add(candidate_node_id)
            
            # 第二步：对top K中的每个节点，获取它们的top K（下一层）
            # 然后检查是否有共同的相似节点，形成连接
            if len(top_k_node_ids) > 0:
                # 对每个top K节点，获取它们的top K相似节点
                second_layer_nodes = set()
                for top_k_node_id in top_k_node_ids:
                    if top_k_node_id not in node_id_to_idx:
                        continue
                    
                    top_k_node = self.hypergraph.nodes[top_k_node_id]
                    top_k_embedding = top_k_node.embedding.cpu().float().numpy().astype('float32').reshape(1, -1)
                    
                    labels2, distances2 = index.knn_query(top_k_embedding, k=min(top_k, len(node_id_mapping)))
                    similarities2 = 1 - distances2[0]
                    top_k_indices2 = labels2[0]
                    
                    # 检查第二层的节点是否也与查询节点相似
                    for i, idx2 in enumerate(top_k_indices2):
                        if similarities2[i] < threshold:
                            continue
                        second_node_id = node_id_mapping[idx2]
                        if second_node_id != node_id and second_node_id not in sample_nodes:
                            # 检查第二层节点与查询节点的相似度
                            second_node = self.hypergraph.nodes[second_node_id]
                            second_embedding = second_node.embedding.cpu().float()
                            query_embedding_tensor = node.embedding.cpu().float()
                            
                            similarity = torch.cosine_similarity(
                                query_embedding_tensor.unsqueeze(0),
                                second_embedding.unsqueeze(0),
                                dim=1
                            ).item()
                            
                            if similarity > threshold and second_node_id not in similar_node_ids:
                                similar_node_ids.append(second_node_id)
                                second_layer_nodes.add(second_node_id)
            
            # 第三步：创建超边（如果找到相似节点）
            if len(similar_node_ids) > 1:
                # 根据索引类型决定创建哪种超边
                if 'report' in index_name:
                    self._create_disease_based_hyperedge(similar_node_ids)
                else:
                    self._create_similarity_based_hyperedge(similar_node_ids)
                processed.update(similar_node_ids)
    
    def _create_similarity_based_hyperedge(self, node_ids: List[str]):
        """创建基于相似度的超边"""
        node_ids = list(dict.fromkeys(node_ids))  # 去重
        if len(node_ids) < 2:
            return
        
        edge_id = f"similarity_{len(self.hypergraph.hyperedges)}"
        hyperedge = Hyperedge(
            edge_id=edge_id,
            node_ids=node_ids,
            weight=1.0,
            edge_type="similarity_based",
            metadata={}
        )
        self.hypergraph.add_hyperedge(hyperedge)
    
    def _build_disease_based_edges(self):
        """构建基于病种的超边（根据REPORT节点的相似度，同时连接KNOWLEDGE节点）"""
        report_nodes = [
            (node_id, node) for node_id, node in self.hypergraph.nodes.items()
            if node.node_type == NodeType.REPORT
        ]
        
        print(f"  Building disease hyperedges for REPORT nodes (with KNOWLEDGE connections)...")
        
        # 使用HNSW贪心策略构建超边（使用report+knowledge索引）
        self._build_similarity_edges_with_hnsw_greedy(report_nodes, 'report+knowledge')
        
        # 确保每个REPORT节点至少连接一个KNOWLEDGE节点
        self._ensure_report_knowledge_connections()
    
    def _create_disease_based_hyperedge(self, node_ids: List[str]):
        """创建基于病种的超边"""
        node_ids = list(dict.fromkeys(node_ids))  # 去重
        if len(node_ids) < 2:
            return
        
        edge_id = f"disease_{len(self.hypergraph.hyperedges)}"
        hyperedge = Hyperedge(
            edge_id=edge_id,
            node_ids=node_ids,
            weight=1.0,
            edge_type="disease_based",
            metadata={}
        )
        self.hypergraph.add_hyperedge(hyperedge)
    
    def _ensure_report_knowledge_connections(self):
        """确保每个REPORT节点至少连接一个KNOWLEDGE节点"""
        # 收集所有REPORT节点
        report_nodes = [
            (node_id, node) for node_id, node in self.hypergraph.nodes.items()
            if node.node_type == NodeType.REPORT
        ]
        
        # 收集所有KNOWLEDGE节点
        knowledge_nodes = [
            (node_id, node) for node_id, node in self.hypergraph.nodes.items()
            if node.node_type == NodeType.KNOWLEDGE
        ]
        
        print(f"  Found {len(knowledge_nodes)} KNOWLEDGE nodes")
        if not knowledge_nodes:
            print("  No KNOWLEDGE nodes found, skipping report-knowledge connections")
            # 调试：检查超图中所有节点类型
            all_node_types = {}
            for node_id, node in self.hypergraph.nodes.items():
                node_type_str = str(node.node_type)
                all_node_types[node_type_str] = all_node_types.get(node_type_str, 0) + 1
            print(f"  All node types in hypergraph: {all_node_types}")
            return
        
        # 检查每个REPORT节点是否已连接到KNOWLEDGE节点
        report_to_knowledge = defaultdict(set)
        for edge_id, edge in self.hypergraph.hyperedges.items():
            if edge.edge_type == "disease_based":
                report_ids = [nid for nid in edge.node_ids if nid in [rn[0] for rn in report_nodes]]
                knowledge_ids = [nid for nid in edge.node_ids if nid in [kn[0] for kn in knowledge_nodes]]
                for report_id in report_ids:
                    report_to_knowledge[report_id].update(knowledge_ids)
        
        # 为没有连接到KNOWLEDGE的REPORT节点添加连接
        threshold = 0.7  # 与disease_based_edges使用相同的阈值
        connected_count = 0
        
        for report_id, report_node in tqdm(report_nodes, desc="  Ensuring report-knowledge connections"):
            if report_id in report_to_knowledge and len(report_to_knowledge[report_id]) > 0:
                continue  # 已经连接了KNOWLEDGE节点
            
            # 找到最相似的KNOWLEDGE节点
            report_embedding = report_node.embedding.cpu().float()
            best_knowledge_id = None
            best_similarity = -1.0
            
            for knowledge_id, knowledge_node in knowledge_nodes:
                knowledge_embedding = knowledge_node.embedding.cpu().float()
                similarity = torch.cosine_similarity(
                    report_embedding.unsqueeze(0),
                    knowledge_embedding.unsqueeze(0),
                    dim=1
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_knowledge_id = knowledge_id
            
            # 如果相似度达到阈值，创建连接
            if best_knowledge_id and best_similarity >= threshold:
                # 检查是否已存在包含这两个节点的disease_based超边
                existing_edge = None
                for edge_id, edge in self.hypergraph.hyperedges.items():
                    if edge.edge_type == "disease_based":
                        if report_id in edge.node_ids and best_knowledge_id in edge.node_ids:
                            existing_edge = edge
                            break
                
                if existing_edge:
                    # 已存在，更新记录
                    report_to_knowledge[report_id].add(best_knowledge_id)
                else:
                    # 创建新的超边
                    self._create_disease_based_hyperedge([report_id, best_knowledge_id])
                    connected_count += 1
            else:
                # 即使相似度不够，也强制连接最相似的（确保每个report至少连接一个knowledge）
                if best_knowledge_id:
                    self._create_disease_based_hyperedge([report_id, best_knowledge_id])
                    connected_count += 1
        
        if connected_count > 0:
            print(f"  Added {connected_count} report-knowledge connections")
    
    def _save_hypergraph(self, output_path: str):
        """保存超图到文件（直接保存，优化版本）"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("  Saving hypergraph...")
        node_items = list(self.hypergraph.nodes.items())
        total_nodes = len(node_items)
        
        # 直接写入，使用大buffer加速
        with open(output_path, 'w', buffering=2*1024*1024) as f:  # 2MB buffer
            f.write('{\n')
            f.write(f'  "embed_dim": {self.embed_dim},\n')
            f.write('  "nodes": {\n')
            
            # 直接处理节点，不并行
            node_count = 0
            for node_id, node in node_items:
                # 处理节点数据
                embedding_np = node.embedding.cpu().float().numpy()
                node_data = {
                    'node_id': node.node_id,
                    'node_type': node.node_type.value,
                    'embedding': embedding_np.tolist(),
                    'metadata': {}
                }
                
                # 处理metadata
                for key, value in node.metadata.items():
                    if isinstance(value, torch.Tensor):
                        tensor_np = value.cpu().float().numpy()
                        node_data['metadata'][key] = tensor_np.tolist()
                    else:
                        node_data['metadata'][key] = value
                
                # 直接写入JSON
                node_json = json.dumps(node_data, ensure_ascii=False)
                f.write(f'    "{node_id}": {node_json}')
                if node_count < total_nodes - 1:
                    f.write(',')
                f.write('\n')
                
                node_count += 1
                
                # 每1000个节点打印一次进度
                if node_count % 1000 == 0:
                    print(f"    Processed {node_count}/{total_nodes} nodes")
            
            f.write('  },\n')
            f.write('  "hyperedges": {\n')
            
            # 直接处理超边
            edge_items = list(self.hypergraph.hyperedges.items())
            total_edges = len(edge_items)
            edge_count = 0
            
            for edge_id, edge in edge_items:
                edge_data = {
                    'edge_id': edge.edge_id,
                    'node_ids': edge.node_ids,
                    'weight': edge.weight,
                    'edge_type': edge.edge_type,
                    'metadata': edge.metadata
                }
                
                edge_json = json.dumps(edge_data, ensure_ascii=False)
                f.write(f'    "{edge_id}": {edge_json}')
                if edge_count < total_edges - 1:
                    f.write(',')
                f.write('\n')
                
                edge_count += 1
                
                # 每5000个超边打印一次进度
                if edge_count % 5000 == 0:
                    print(f"    Processed {edge_count}/{total_edges} hyperedges")
            
            f.write('  }\n')
            f.write('}\n')
        
        print(f"Hypergraph saved to {output_path}")
    
    def _print_hyperedge_statistics(self, edge_type: str):
        """输出指定类型超边的统计信息"""
        edges_of_type = [
            edge for edge in self.hypergraph.hyperedges.values()
            if edge.edge_type == edge_type
        ]
        
        if not edges_of_type:
            print(f"  {edge_type}: 0个超边")
            return
        
        # 统计每个超边连接的节点数
        node_counts = [len(edge.node_ids) for edge in edges_of_type]
        total_edges = len(edges_of_type)
        total_nodes_connected = sum(node_counts)
        avg_nodes_per_edge = total_nodes_connected / total_edges if total_edges > 0 else 0
        min_nodes = min(node_counts) if node_counts else 0
        max_nodes = max(node_counts) if node_counts else 0
        
        print(f"  {edge_type}:")
        print(f"    超边数量: {total_edges}")
        print(f"    总连接数: {total_nodes_connected} (所有超边连接的节点总数)")
        print(f"    平均每个超边连接: {avg_nodes_per_edge:.2f} 个节点")
        print(f"    最少连接: {min_nodes} 个节点")
        print(f"    最多连接: {max_nodes} 个节点")
        
        # 统计连接数分布
        count_distribution = Counter(node_counts)
        print(f"    连接数分布:")
        for count in sorted(count_distribution.keys()):
            num_edges = count_distribution[count]
            percentage = (num_edges / total_edges) * 100
            print(f"      {count}个节点: {num_edges}个超边 ({percentage:.1f}%)")
    
    def _print_all_hyperedge_statistics(self):
        """输出所有超边的统计信息"""
        # 按类型分组统计
        edge_types = {}
        for edge in self.hypergraph.hyperedges.values():
            edge_type = edge.edge_type
            if edge_type not in edge_types:
                edge_types[edge_type] = []
            edge_types[edge_type].append(edge)
        
        for edge_type in sorted(edge_types.keys()):
            edges_of_type = edge_types[edge_type]
            total_edges = len(edges_of_type)
            node_counts = [len(edge.node_ids) for edge in edges_of_type]
            total_nodes_connected = sum(node_counts)
            avg_nodes_per_edge = total_nodes_connected / total_edges if total_edges > 0 else 0
            
            print(f"  {edge_type}:")
            print(f"    超边数量: {total_edges}")
            print(f"    总连接数: {total_nodes_connected}")
            print(f"    平均每个超边连接: {avg_nodes_per_edge:.2f} 个节点")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build hypergraph')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum number of samples (for testing, default None means use all data)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: hypergraph/hypergraph.json)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for embedding generation (default: 8, increase if you have enough GPU memory)')
    args = parser.parse_args()
    
    # 初始化模型
    print("Initializing model...")
    model = EmbedModel()
    model.model.eval()
    
    # 创建超图构建器
    builder = HypergraphBuilder(
        model=model,
        embed_dim=1024,
        k=5,
        train_sample_ratio=0.003,  # 0.3%的训练数据
        batch_size=args.batch_size
    )
    
    # 构建超图
    if args.output:
        output_path = args.output
    else:
        output_dir = Path("/mnt/sda/VLM/code/hypercode/hyperbuild/hypergraph")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "hypergraph.json"
    
    hypergraph = builder.build(output_path=str(output_path), max_samples=args.max_samples)
    
    print("\nHypergraph construction completed!")


if __name__ == "__main__":
    main()

