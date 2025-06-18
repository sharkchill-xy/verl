#!/usr/bin/env python3
"""
简化的编程数据集准备脚本 - 为Python解释器训练准备数据
使用KodCode-Light-RL-10K数据集，该数据集专门为强化学习设计
"""

import os
import json
import argparse
from datasets import load_dataset


def prepare_coding_data(output_dir="/data2/lixy/coding", test_size=500):
    """
    准备编程数据集用于训练模型使用Python解释器
    """
    print("正在加载KodCode-Light-RL-10K数据集...")
    
    try:
        dataset = load_dataset("KodCode/KodCode-Light-RL-10K")
        print(f"数据集加载成功，包含 {len(dataset['train'])} 个训练样本")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("尝试使用本地备选方案...")
        # 如果网络有问题，我们可以创建一些简单的编程问题作为示例
        return create_simple_coding_examples(output_dir)
    
    # 过滤数据 - 移除需要特殊库的问题
    block_libs = [
        "torch", "scipy", "sklearn", "cv2", "imageio", "matplotlib", 
        "pillow", "seaborn", "pandas", "bs4", "flask", "keras"
    ]
    
    def should_keep_example(example):
        """检查是否应该保留这个示例"""
        solution = example.get('solution', '')
        test = example.get('test', '')
        
        # 过滤包含被阻止库的示例
        for lib in block_libs:
            if lib in solution.lower() or lib in test.lower():
                return False
        return True
    
    # 过滤数据集
    filtered_dataset = dataset['train'].filter(should_keep_example)
    print(f"过滤后剩余 {len(filtered_dataset)} 个样本")
    
    # 转换数据格式
    def transform_example(example, idx):
        """将KodCode格式转换为VERL训练格式"""
        
        # 构建prompt - 强调使用Python解释器来解决问题
        question = example['question'].strip()
        
        # 获取函数声明信息
        test_info = example.get('test_info', [])
        function_declaration = ""
        if test_info and len(test_info) > 0:
            function_declaration = test_info[0].get('function_declaration', '').strip()
        
        # 构建强调使用Python解释器的prompt
        prompt = f"""Please solve the following programming problem step by step. 

You should:
1. First understand the problem carefully
2. Use the Python interpreter to test your understanding and explore solutions
3. Write code step by step, testing each part
4. Provide the final solution

Problem: {question}"""
        
        if function_declaration:
            prompt += f"\n\nNote: The function should follow this declaration: {function_declaration}"
        
        prompt += "\n\nPlease use the Python interpreter to help you solve this problem step by step."
        
        # 构建测试代码
        test_code = "from solution import *\n" + example['test'].strip()
        
        return {
            "data_source": "coding",
            "prompt": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "ability": "coding",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({"pytest": test_code})
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "reference": example['solution'],
                "prompt": prompt,
                "dataset": "KodCode-Light-RL-10K",
                "original_question": question,
                "function_declaration": function_declaration
            }
        }
    
    # 转换数据
    transformed_dataset = filtered_dataset.map(transform_example, with_indices=True)
    
    # 划分训练集和测试集
    if len(transformed_dataset) > test_size:
        splits = transformed_dataset.train_test_split(test_size=test_size, seed=42)
        train_dataset = splits['train']
        test_dataset = splits['test']
    else:
        # 如果数据量太小，使用全部作为训练集，少量作为测试集
        test_size = min(50, len(transformed_dataset) // 10)
        splits = transformed_dataset.train_test_split(test_size=test_size, seed=42)
        train_dataset = splits['train']
        test_dataset = splits['test']
    
    print(f"最终数据集大小：训练集 {len(train_dataset)}, 测试集 {len(test_dataset)}")
    
    # 保存数据
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    
    print(f"数据集已保存到:")
    print(f"  训练集: {train_path}")
    print(f"  测试集: {test_path}")
    
    # 打印一些示例
    print("\n=== 数据样本示例 ===")
    sample = train_dataset[0]
    print("Prompt:", sample['prompt'][0]['content'][:200] + "...")
    print("Reference solution:", sample['extra_info']['reference'][:100] + "...")
    
    return train_path, test_path


def create_simple_coding_examples(output_dir):
    """
    创建一些简单的编程问题作为备选方案
    """
    print("创建简单的编程问题示例...")
    
    examples = [
        {
            "question": "Write a function that returns the sum of two numbers.",
            "solution": "def add_numbers(a, b):\n    return a + b",
            "test": "assert add_numbers(2, 3) == 5\nassert add_numbers(-1, 1) == 0"
        },
        {
            "question": "Write a function that checks if a number is even.",
            "solution": "def is_even(n):\n    return n % 2 == 0",
            "test": "assert is_even(4) == True\nassert is_even(3) == False"
        },
        {
            "question": "Write a function that finds the maximum number in a list.",
            "solution": "def find_max(numbers):\n    return max(numbers)",
            "test": "assert find_max([1, 3, 2]) == 3\nassert find_max([-1, -5, -2]) == -1"
        },
        {
            "question": "Write a function that reverses a string.",
            "solution": "def reverse_string(s):\n    return s[::-1]",
            "test": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('Python') == 'nohtyP'"
        },
        {
            "question": "Write a function that calculates the factorial of a number.",
            "solution": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            "test": "assert factorial(5) == 120\nassert factorial(0) == 1\nassert factorial(3) == 6"
        }
    ]
    
    # 转换为训练格式
    train_data = []
    test_data = []
    
    for i, example in enumerate(examples):
        data_point = {
            "data_source": "coding",
            "prompt": [
                {
                    "role": "user",
                    "content": f"Please solve this programming problem step by step using the Python interpreter:\n\n{example['question']}\n\nUse the Python interpreter to test your solution."
                }
            ],
            "ability": "coding", 
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({"pytest": example['test']})
            },
            "extra_info": {
                "split": "train" if i < 4 else "test",
                "index": i,
                "reference": example['solution'],
                "dataset": "simple_examples"
            }
        }
        
        if i < 4:
            train_data.append(data_point)
        else:
            test_data.append(data_point)
    
    # 保存数据
    from datasets import Dataset
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    
    print(f"简单示例数据集已保存到:")
    print(f"  训练集: {train_path} ({len(train_data)} 样本)")
    print(f"  测试集: {test_path} ({len(test_data)} 样本)")
    
    return train_path, test_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备编程数据集用于Python解释器训练")
    parser.add_argument("--output_dir", default="/data2/lixy/coding", help="输出目录")
    parser.add_argument("--test_size", type=int, default=500, help="测试集大小")
    
    args = parser.parse_args()
    
    try:
        train_path, test_path = prepare_coding_data(args.output_dir, args.test_size)
        print(f"\n✅ 数据准备完成!")
        print(f"训练集: {train_path}")
        print(f"测试集: {test_path}")
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        import traceback
        traceback.print_exc()