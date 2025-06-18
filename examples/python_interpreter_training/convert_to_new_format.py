#!/usr/bin/env python3
"""
将现有的LeetCode数据转换为新的Python解释器格式
"""

import pandas as pd
import json
from pathlib import Path

def convert_to_new_format():
    """
    转换现有数据为新的<python_interpreter>格式
    """
    # 新的系统提示词 - 使用function call格式
    SYSTEM_PROMPT = """You are a helpful programming assistant with access to a Python interpreter.

You have access to the following function:
- python_interpreter: Execute Python code and get results

Use the Python interpreter to:
- Test your understanding of problems
- Explore different approaches  
- Verify your solutions step by step
- Debug and refine your code

Always explain your approach and use the Python interpreter to solve problems incrementally. Call the python_interpreter function whenever you need to run code."""

    def convert_sample(sample):
        """转换单个样本"""
        # 提取原始问题
        if 'src' in sample and sample['src'] is not None:
            problem_text = sample['src']
        elif 'extra_info' in sample and 'original_question' in sample['extra_info']:
            problem_text = sample['extra_info']['original_question']
        else:
            problem_text = "No problem description available"

        # 创建新的prompt
        new_prompt = f"""Please solve the following programming problem step by step.

Problem:
{problem_text}

Instructions:
1. Analyze the problem and understand what's required
2. Call the python_interpreter function to test your understanding and explore solutions
3. Develop your solution incrementally, testing each part with the interpreter
4. Provide the final working solution

Use the python_interpreter function throughout your problem-solving process to verify your approach."""

        # 更新sample
        new_sample = sample.copy()
        new_sample['prompt'] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": new_prompt}
        ]
        
        return new_sample

    # 读取现有数据
    print("读取现有LeetCode数据...")
    train_df = pd.read_parquet('/data/coding_leetcode/train.parquet')
    test_df = pd.read_parquet('/data/coding_leetcode/test.parquet')
    
    print(f"原始训练集: {len(train_df)} 样本")
    print(f"原始测试集: {len(test_df)} 样本")

    # 转换数据格式
    print("转换为新格式...")
    converted_train = []
    for idx, sample in train_df.iterrows():
        converted_train.append(convert_sample(sample))
    
    converted_test = []
    for idx, sample in test_df.iterrows():
        converted_test.append(convert_sample(sample))

    # 保存新数据
    output_dir = Path("/data/coding_leetcode_v2")
    output_dir.mkdir(exist_ok=True)
    
    new_train_df = pd.DataFrame(converted_train)
    new_test_df = pd.DataFrame(converted_test)
    
    new_train_df.to_parquet(output_dir / "train.parquet")
    new_test_df.to_parquet(output_dir / "test.parquet")
    
    print(f"✅ 新格式数据保存到: {output_dir}")
    print(f"新训练集: {len(new_train_df)} 样本")
    print(f"新测试集: {len(new_test_df)} 样本")
    
    # 验证新格式
    print("\n=== 新格式样本验证 ===")
    sample = new_train_df.iloc[0]
    print("System Prompt:")
    print(sample['prompt'][0]['content'][:200] + "...")
    print("\nUser Prompt:")
    print(sample['prompt'][1]['content'][:300] + "...")

if __name__ == "__main__":
    convert_to_new_format()