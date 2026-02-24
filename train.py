"""
直接构建超图并训练Model F（跳过保存和加载步骤）
"""
import os
import argparse

# 设置离线模式，强制只使用本地文件
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from hyperbuild.build_hypergraph import HypergraphBuilder
from hyperbuild.embed_model import EmbedModel
from hyperbuild.data_loader import DataLoader
from modeltr.train_model_f import TrainModelF

def main():
    parser = argparse.ArgumentParser(description='Build hypergraph and train Model F directly')
    
    # 超图构建参数
    parser.add_argument('--vlm_model_path', type=str, default=None,
                        help='VLM模型路径（用于构建超图和训练）')
    parser.add_argument('--train_sample_ratio', type=float, default=0.01,
                        help='训练数据采样比例（用于构建超图，默认1%）')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数量（用于构建超图）')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批量处理大小（用于构建超图）')
    
    # Model F训练参数
    parser.add_argument('--adapter_checkpoint_dir', type=str,
                        default='/mnt/sda/VLM/code/hypercode/adatri/adamodel/adapter_epoch_6',
                        help='Adapter checkpoint目录（第6轮）')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/sda/VLM/code/hypercode/checkpoints',
                        help='模型F保存目录')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--similarity_threshold', type=float, default=0.6,
                        help='相似度阈值（用于构建超边）')
    parser.add_argument('--max_hops', type=int, default=3,
                        help='最大hop数量')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Step 1: Building hypergraph (direct mode, no saving)")
    print("=" * 80)
    
    # 1. 加载embedding模型
    print("\nLoading embedding model...")
    embed_model = EmbedModel(model_path=args.vlm_model_path)
    
    # 2. 构建超图（不保存）
    print("\nBuilding hypergraph...")
    builder = HypergraphBuilder(
        model=embed_model,
        embed_dim=1024,
        k=5,
        train_sample_ratio=args.train_sample_ratio,
        batch_size=args.batch_size
    )
    
    # 构建超图，不保存（output_path=None）
    hypergraph = builder.build(output_path=None, max_samples=args.max_samples)
    
    print("\n" + "=" * 80)
    print("Step 2: Training Model F (using hypergraph object directly)")
    print("=" * 80)
    
    # 3. 直接使用超图对象训练Model F
    trainer = TrainModelF(
        vlm_model_path=args.vlm_model_path,
        hypergraph=hypergraph,  # 直接传入超图对象
        adapter_checkpoint_dir=args.adapter_checkpoint_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        similarity_threshold=args.similarity_threshold,
        max_hops=args.max_hops
    )
    
    # 5. 开始训练
    trainer.run()
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

