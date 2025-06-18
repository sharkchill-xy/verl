#!/usr/bin/env python3
"""
多数据源编程数据集准备脚本 - 支持KodCode和LeetCode数据集
结合code-r1的方法，为Python解释器训练准备数据
"""

import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, concatenate_datasets
import requests
from pathlib import Path


def download_leetcode_dataset(data_dir="/home/lixy/workspace/verl/LeetCodeDataset"):
    """
    下载LeetCode数据集
    """
    print("正在下载LeetCode数据集...")
    
    os.makedirs(f"{data_dir}/data", exist_ok=True)
    
    # LeetCode数据文件的URL (这些是公开可用的数据集)
    urls = {
        "LeetCodeDataset-v2-test-problems.jsonl": "https://huggingface.co/datasets/greengerong/leetcode/resolve/main/data/test.jsonl",
        "LeetCodeDataset-v2-rl-problems.jsonl": "https://huggingface.co/datasets/greengerong/leetcode/resolve/main/data/train.jsonl",
        "LeetCodeDataset-v2-sft-problems.jsonl": "https://huggingface.co/datasets/greengerong/leetcode/resolve/main/data/train.jsonl"
    }
    
    for filename, url in urls.items():
        filepath = f"{data_dir}/data/{filename}"
        if os.path.exists(filepath):
            print(f"文件已存在: {filename}")
            continue
            
        try:
            print(f"下载 {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ {filename} 下载完成")
        except Exception as e:
            print(f"❌ 下载 {filename} 失败: {e}")
            # 创建一个空文件避免后续错误
            with open(filepath, 'w') as f:
                pass
    
    return data_dir


def prepare_kodcode_data():
    """
    准备KodCode-Light-RL-10K数据集
    """
    print("=== 加载KodCode-Light-RL-10K数据集 ===")
    
    try:
        dataset = load_dataset("KodCode/KodCode-Light-RL-10K")
        print(f"KodCode数据集加载成功，包含 {len(dataset['train'])} 个训练样本")
    except Exception as e:
        print(f"KodCode加载失败: {e}")
        return None, None
    
    # 过滤数据 - 移除需要特殊库的问题
    block_libs = [
        "torch", "scipy", "sklearn", "cv2", "imageio", "matplotlib", 
        "pillow", "seaborn", "pandas", "bs4", "flask", "keras", "numpy"
    ]
    
    def should_keep_example(example):
        solution = example.get('solution', '')
        test = example.get('test', '')
        for lib in block_libs:
            if lib in solution.lower() or lib in test.lower():
                return False
        return True
    
    # 过滤数据集
    filtered_dataset = dataset['train'].filter(should_keep_example)
    print(f"KodCode过滤后剩余 {len(filtered_dataset)} 个样本")
    
    # 转换数据格式
    def transform_kodcode_example(example, idx):
        question = example['question'].strip()
        test_info = example.get('test_info', [])
        function_declaration = ""
        if test_info and len(test_info) > 0:
            function_declaration = test_info[0].get('function_declaration', '').strip()
        
        prompt = f"""Please solve the following programming problem step by step.

Problem:
{question}"""
        
        if function_declaration:
            prompt += f"\n\nFunction Declaration: {function_declaration}"
        
        prompt += f"""

Instructions:
1. Analyze the problem and understand what's required
2. Use <python_interpreter> tags to test your understanding and explore solutions
3. Develop your solution incrementally, testing each part
4. Provide the final working solution

Use the Python interpreter throughout your problem-solving process to verify your approach."""
        
        test_code = "from solution import *\n" + example['test'].strip()
        
        return {
            "data_source": "kodcode",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "coding",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({"pytest": test_code})
            },
            "extra_info": {
                "split": "train",
                "index": f"kodcode_{idx}",
                "reference": example['solution'],
                "dataset": "KodCode-Light-RL-10K",
                "original_question": question,
                "function_declaration": function_declaration,
                "prompt": prompt,
                "task_id": f"kodcode_{idx}",
                "entry_point": function_declaration,
                "difficulty": "Unknown"
            }
        }
    
    transformed_dataset = filtered_dataset.map(transform_kodcode_example, with_indices=True)
    return transformed_dataset, "kodcode"


def prepare_leetcode_data(leetcode_dir="/home/lixy/workspace/VerlCoder/code-r1/LeetCodeDataset"):
    """
    准备LeetCode数据集，参考code-r1的方法
    """
    print("=== 加载LeetCode数据集 ===")
    
    try:
        # 直接使用本地LeetCodeDataset数据文件
        test_dataset = load_dataset("json",
                                    data_files=f"{leetcode_dir}/data/LeetCodeDataset-v2-test-problems.jsonl")["train"]
        print("LeetCode测试集:", len(test_dataset))
        
        # 加载训练数据文件
        train_files = [
            f"{leetcode_dir}/data/LeetCodeDataset-v2-rl-problems.jsonl",
            f"{leetcode_dir}/data/LeetCodeDataset-v2-sft-problems.jsonl"
        ]
        
        train_datasets = []
        for filepath in train_files:
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                try:
                    ds = load_dataset("json", data_files=filepath)["train"]
                    train_datasets.append(ds)
                    print(f"加载 {os.path.basename(filepath)}: {len(ds)} 样本")
                except Exception as e:
                    print(f"跳过损坏的文件 {filepath}: {e}")
        
        if not train_datasets:
            print("❌ 没有成功加载任何LeetCode训练数据")
            return None, None
            
        train_dataset = concatenate_datasets(train_datasets)
        
        # 去重 - 避免重复的question_id
        try:
            test_question_ids = set([d["meta"]["question_id"] for d in test_dataset])
            train_dataset = train_dataset.filter(
                lambda example: example["meta"]["question_id"] not in test_question_ids
            )
            
            # 进一步去重
            seen_question_ids = set()
            first_time_idx = []
            for i, example in enumerate(train_dataset):
                question_id = example["meta"]["question_id"]
                if question_id not in seen_question_ids:
                    first_time_idx.append(i)
                    seen_question_ids.add(question_id)
            train_dataset = train_dataset.select(first_time_idx)
            
            print(f"LeetCode去重后训练集: {len(train_dataset)} 样本")
        except Exception as e:
            print(f"去重过程中出现错误: {e}, 使用原始数据")
            print(f"LeetCode训练集: {len(train_dataset)} 样本")
        
    except Exception as e:
        print(f"LeetCode数据集加载失败: {e}")
        return None, None
    
    # 系统提示词 - 专注于Python解释器调用
    SYSTEM_PROMPT = """You are a helpful programming assistant that can execute Python code using an interpreter.

When you need to run Python code, use the format:
<python_interpreter>
your_python_code_here
</python_interpreter>

The interpreter will execute your code and return the results. You can use this to:
- Test your understanding of problems
- Explore different approaches
- Verify your solutions step by step
- Debug and refine your code

Always explain your approach and use the Python interpreter to solve problems incrementally."""
    
    # 转换数据格式
    def transform_leetcode_example(example, idx):
        # LeetCodeDataset v2格式
        if 'meta' in example and 'query' in example['meta']:
            problem_query = example['meta']['query'].strip()
            original_question = problem_query
            
            prompt = f"""Please solve the following programming problem step by step.

Problem:
{problem_query}

Instructions:
1. Analyze the problem and understand what's required
2. Use <python_interpreter> tags to test your understanding and explore solutions
3. Develop your solution incrementally, testing each part
4. Provide the final working solution

Use the Python interpreter throughout your problem-solving process to verify your approach."""
            
            # 构建测试代码 - 使用LeetCode的测试格式
            test_code = f"{example['test']}\n\ncheck({example['entry_point'].strip()})"
            reference = example.get('completion', '')
            
        else:
            # 备选格式
            problem_text = example.get('src', example.get('problem', 'No problem description'))
            original_question = problem_text
            
            prompt = f"""Please solve the following programming problem step by step.

Problem:
{problem_text}

Instructions:
1. Analyze the problem and understand what's required
2. Use <python_interpreter> tags to test your understanding and explore solutions  
3. Develop your solution incrementally, testing each part
4. Provide the final working solution

Use the Python interpreter throughout your problem-solving process to verify your approach."""
            
            test_code = example.get('test', 'pass')
            reference = example.get('completion', example.get('solution', ''))
        
        return {
            "data_source": "leetcode",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "ability": "coding",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({"functional": test_code})
            },
            "extra_info": {
                "split": "train",
                "index": f"leetcode_{idx}",
                "reference": reference,
                "dataset": "LeetCodeDataset-v2",
                "prompt": prompt,
                "task_id": example.get('task_id', f'unknown_{idx}'),
                "entry_point": example.get('entry_point', ''),
                "difficulty": example.get('meta', {}).get('difficulty', 'Unknown') if 'meta' in example else 'Unknown',
                "function_declaration": example.get('entry_point', ''),
                "original_question": original_question
            }
        }
    
    transformed_train = train_dataset.map(transform_leetcode_example, with_indices=True)
    transformed_test = test_dataset.map(
        lambda example, idx: transform_leetcode_example(example, f"test_{idx}"), 
        with_indices=True
    )
    
    return transformed_train, transformed_test


def prepare_multi_source_data(output_dir="/data2/lixy/coding", use_kodcode=True, use_leetcode=True, test_size=500):
    """
    准备多数据源编程数据集
    """
    print("正在准备多数据源编程数据集...")
    
    train_datasets = []
    test_datasets = []
    source_names = []
    
    # 准备KodCode数据
    if use_kodcode:
        kodcode_data, kodcode_name = prepare_kodcode_data()
        if kodcode_data is not None:
            # 划分训练测试集
            splits = kodcode_data.train_test_split(test_size=min(test_size//2, len(kodcode_data)//10), seed=42)
            train_datasets.append(splits['train'])
            test_datasets.append(splits['test'])
            source_names.append("kodcode")
            print(f"✅ KodCode数据准备完成: 训练集 {len(splits['train'])}, 测试集 {len(splits['test'])}")
    
    # 准备LeetCode数据
    if use_leetcode:
        leetcode_train, leetcode_test = prepare_leetcode_data()
        if leetcode_train is not None:
            # 限制数据量避免过大
            if len(leetcode_train) > 5000:
                leetcode_train = leetcode_train.shuffle(seed=42).select(range(5000))
            if leetcode_test is not None and len(leetcode_test) > test_size//2:
                leetcode_test = leetcode_test.shuffle(seed=42).select(range(test_size//2))
            
            train_datasets.append(leetcode_train)
            if leetcode_test is not None:
                test_datasets.append(leetcode_test)
            source_names.append("leetcode")
            print(f"✅ LeetCode数据准备完成: 训练集 {len(leetcode_train)}, 测试集 {len(leetcode_test) if leetcode_test else 0}")
    
    # 检查是否有可用数据
    if not train_datasets:
        print("❌ 没有成功准备任何数据集，使用简单示例...")
        return create_simple_examples(output_dir)
    
    # 合并数据集
    final_train = concatenate_datasets(train_datasets).shuffle(seed=42)
    final_test = concatenate_datasets(test_datasets) if test_datasets else None
    
    # 如果测试集为空或太小，从训练集中分出一部分
    if final_test is None or len(final_test) < 50:
        test_size_actual = min(test_size, len(final_train) // 10)
        splits = final_train.train_test_split(test_size=test_size_actual, seed=42)
        final_train = splits['train']
        final_test = splits['test']
    
    print(f"最终合并数据集大小：训练集 {len(final_train)}, 测试集 {len(final_test)}")
    print(f"数据源: {', '.join(source_names)}")
    
    # 保存数据
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    final_train.to_parquet(train_path)
    final_test.to_parquet(test_path)
    
    print(f"数据集已保存到:")
    print(f"  训练集: {train_path}")
    print(f"  测试集: {test_path}")
    
    # 打印数据源统计
    print("\n=== 数据源统计 ===")
    for source in source_names:
        train_count = len([x for x in final_train if x['extra_info']['dataset'].lower().startswith(source)])
        test_count = len([x for x in final_test if x['extra_info']['dataset'].lower().startswith(source)])
        print(f"{source}: 训练集 {train_count}, 测试集 {test_count}")
    
    # 打印示例
    print("\n=== 数据样本示例 ===")
    sample = final_train[0]
    print("数据源:", sample['extra_info']['dataset'])
    print("Prompt:", sample['prompt'][-1]['content'][:200] + "...")
    if 'reference' in sample['extra_info']:
        print("Reference solution:", sample['extra_info']['reference'][:100] + "...")
    
    return train_path, test_path


def create_simple_examples(output_dir):
    """
    创建简单编程示例作为备选方案
    """
    print("创建简单编程示例...")
    
    examples = [
        {
            "problem": "Write a function that returns the sum of two numbers.",
            "solution": "def add_numbers(a, b):\n    return a + b",
            "test": "assert add_numbers(2, 3) == 5\nassert add_numbers(-1, 1) == 0"
        },
        {
            "problem": "Write a function that checks if a number is even.",
            "solution": "def is_even(n):\n    return n % 2 == 0", 
            "test": "assert is_even(4) == True\nassert is_even(3) == False"
        },
        {
            "problem": "Write a function that finds the maximum number in a list.",
            "solution": "def find_max(numbers):\n    return max(numbers)",
            "test": "assert find_max([1, 3, 2]) == 3\nassert find_max([-1, -5, -2]) == -1"
        }
    ]
    
    # 转换为训练格式
    train_data = []
    for i, example in enumerate(examples[:2]):
        train_data.append({
            "data_source": "simple",
            "prompt": [{"role": "user", "content": f"Problem: {example['problem']}\nUse Python interpreter to solve step by step."}],
            "ability": "coding",
            "reward_model": {"style": "rule", "ground_truth": json.dumps({"pytest": example['test']})},
            "extra_info": {"split": "train", "index": i, "reference": example['solution'], "dataset": "simple_examples"}
        })
    
    test_data = [train_data[-1]]  # 使用最后一个作为测试
    
    from datasets import Dataset
    os.makedirs(output_dir, exist_ok=True)
    
    Dataset.from_list(train_data).to_parquet(os.path.join(output_dir, "train.parquet"))
    Dataset.from_list(test_data).to_parquet(os.path.join(output_dir, "test.parquet"))
    
    return os.path.join(output_dir, "train.parquet"), os.path.join(output_dir, "test.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备多数据源编程数据集用于Python解释器训练")
    parser.add_argument("--output_dir", default="/data2/lixy/coding", help="输出目录")
    parser.add_argument("--test_size", type=int, default=500, help="测试集大小")
    parser.add_argument("--use_kodcode", action="store_true", default=True, help="使用KodCode数据集")
    parser.add_argument("--use_leetcode", action="store_true", default=True, help="使用LeetCode数据集")
    parser.add_argument("--no_kodcode", action="store_true", help="不使用KodCode数据集")
    parser.add_argument("--no_leetcode", action="store_true", help="不使用LeetCode数据集") 
    
    args = parser.parse_args()
    
    # 处理参数
    use_kodcode = args.use_kodcode and not args.no_kodcode
    use_leetcode = args.use_leetcode and not args.no_leetcode
    
    try:
        train_path, test_path = prepare_multi_source_data(
            output_dir=args.output_dir,
            use_kodcode=use_kodcode,
            use_leetcode=use_leetcode,
            test_size=args.test_size
        )
        print(f"\n✅ 多数据源数据准备完成!")
        print(f"训练集: {train_path}")
        print(f"测试集: {test_path}")
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        import traceback
        traceback.print_exc()