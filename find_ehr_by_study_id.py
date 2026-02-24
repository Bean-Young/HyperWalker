#!/usr/bin/env python3
"""
根据study_id在ehrdata目录中查找对应的完整数据路径
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Set

def extract_study_ids_from_node_ids(node_ids: List[str]) -> Set[str]:
    """从节点ID中提取study_id
    
    节点ID格式: image_None_55024693_0
    其中55024693是study_id（第3部分，索引为2）
    """
    study_ids = set()
    for node_id in node_ids:
        parts = node_id.split('_')
        # 格式: image_None_55024693_0
        if len(parts) >= 4 and parts[0] == 'image' and parts[1] == 'None':
            study_id = parts[2]  # 第3部分就是study_id
            if study_id.isdigit():
                study_ids.add(study_id)
    return study_ids

def find_ehr_file_by_study_id(ehrdata_dir: Path, study_id: str) -> Dict:
    """根据study_id查找EHR文件
    
    Returns:
        {
            'study_id': study_id,
            'found': bool,
            'file_path': str or None,
            'subject_id': str or None,
            'data': dict or None
        }
    """
    result = {
        'study_id': study_id,
        'found': False,
        'file_path': None,
        'subject_id': None,
        'data': None
    }
    
    # 遍历所有subject_id目录
    for subject_dir in ehrdata_dir.iterdir():
        if not subject_dir.is_dir():
            continue
        
        # 查找 study_id.json 文件
        ehr_file = subject_dir / f"{study_id}.json"
        if ehr_file.exists():
            try:
                with open(ehr_file, 'r', encoding='utf-8') as f:
                    ehr_data = json.load(f)
                
                result['found'] = True
                result['file_path'] = str(ehr_file)
                result['subject_id'] = subject_dir.name
                result['data'] = ehr_data
                return result
            except Exception as e:
                print(f"  Warning: Failed to load {ehr_file}: {e}")
                continue
    
    return result

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='根据study_id查找EHR数据文件')
    parser.add_argument('--result_file', type=str,
                        help='召回结果JSON文件路径（可选，如果提供则从节点ID中提取study_id）')
    parser.add_argument('--study_id', type=str,
                        help='直接指定study_id（如：55024693）')
    parser.add_argument('--ehrdata_dir', type=str,
                        default='/mnt/sda/VLM/matched_data/ehrdata',
                        help='EHR数据目录')
    parser.add_argument('--output_file', type=str,
                        help='输出文件路径（可选）')
    
    args = parser.parse_args()
    
    ehrdata_dir = Path(args.ehrdata_dir)
    if not ehrdata_dir.exists():
        print(f"Error: EHR data directory does not exist: {ehrdata_dir}")
        return
    
    study_ids = set()
    
    # 获取study_id列表
    current_sample_study_id = None
    
    if args.study_id:
        study_ids.add(args.study_id)
        print(f"Looking for study_id: {args.study_id}")
    elif args.result_file:
        print(f"Loading result file: {args.result_file}")
        with open(args.result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # 添加当前样本的study_id
        current_sample_study_id = result.get('study_id')
        if current_sample_study_id:
            study_ids.add(current_sample_study_id)
            print(f"Current sample study_id: {current_sample_study_id}")
        
        # 从节点ID中提取study_id
        node_ids = [node['node_id'] for node in result.get('retrieved_nodes', [])]
        extracted_ids = extract_study_ids_from_node_ids(node_ids)
        study_ids.update(extracted_ids)
        print(f"Extracted {len(extracted_ids)} study_ids from node IDs: {sorted(extracted_ids)}")
    else:
        print("Error: Must provide either --result_file or --study_id")
        return
    
    # 查找每个study_id对应的EHR文件
    print(f"\nSearching in {ehrdata_dir}...")
    results = {}
    
    for study_id in sorted(study_ids):
        print(f"\n  Study ID: {study_id}")
        result = find_ehr_file_by_study_id(ehrdata_dir, study_id)
        results[study_id] = result
        
        if result['found']:
            print(f"    ✓ Found: {result['file_path']}")
            print(f"    Subject ID: {result['subject_id']}")
            
            # 显示EHR数据摘要
            if result['data']:
                data = result['data']
                if 'ehr_data' in data:
                    ehr = data['ehr_data']
                    if 'patient_info' in ehr:
                        p = ehr['patient_info']
                        print(f"    Patient: Gender={p.get('gender')}, Age={p.get('anchor_age')}")
                    print(f"    Diagnoses: {len(ehr.get('all_diagnoses', []))}")
                    print(f"    Procedures: {len(ehr.get('all_procedures', []))}")
                    print(f"    Prescriptions: {len(ehr.get('prescriptions', []))}")
                    print(f"    Lab Events: {len(ehr.get('all_labevents', []))}")
        else:
            print(f"    ✗ Not found")
    
    # 输出结果
    output_data = {
        'ehrdata_dir': str(ehrdata_dir),
        'current_sample_study_id': current_sample_study_id,
        'study_ids_searched': sorted(list(study_ids)),
        'found_count': sum(1 for r in results.values() if r['found']),
        'total_count': len(results),
        'results': results
    }
    
    # 如果当前样本的study_id存在，单独标记
    if current_sample_study_id and current_sample_study_id in results:
        output_data['current_sample_ehr'] = results[current_sample_study_id]
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {args.output_file}")
    else:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        if current_sample_study_id:
            print(f"Current sample study_id: {current_sample_study_id}")
            if current_sample_study_id in results and results[current_sample_study_id]['found']:
                print(f"  ✓ Current sample EHR: {results[current_sample_study_id]['file_path']}")
            else:
                print(f"  ✗ Current sample EHR: Not found")
        print(f"\nFound: {output_data['found_count']}/{output_data['total_count']}")
        print(f"\nFile paths:")
        for study_id, result in sorted(results.items()):
            if result['found']:
                marker = "★" if study_id == current_sample_study_id else " "
                print(f"  {marker} {study_id}: {result['file_path']}")

if __name__ == "__main__":
    main()
