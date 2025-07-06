"""
This script is used to evaluate the performance of the model on the AIME 2024 dataset.
这份代码不依赖 verl 框架，通过 openai-compatible api 调用模型，并进行评估。
"""

import argparse
import openai
import datasets
from typing import Dict, Any


TOOLS = [
    {
        'function': {
            'description': 'Execute Python code to perform calculations, data analysis, or other computational tasks.', 
            'name': 'code_interpreter', 
            'parameters': {
                'properties': {
                    'code': {
                        'description': 'The Python code to be executed. This can include calculations, data manipulation, or any valid Python code.', 
                        'type': 'string'
                    }, 
                    'language': {
                        'default': 'python', 
                        'description': 'The programming language of the code. Currently only Python is supported.', 
                        'type': 'string'
                    }
                }, 
                'required': ['code'], 
                'type': 'object'
            }
        }, 
        'type': 'function'
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    # parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)

    # TODO: 还需要 load sandbox fusion 相关的参数
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key)

    dataset = datasets.load_dataset("Maxwell-Jia/AIME_2024", split="train")

    def map_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        system_message = "You are a helpful assistant that can solve math problems with interaction Code Interpreter by Python code."
        user_message = f"""Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process.
        
        
        **user question:** 
        {example['problem']}
        
        Remember to place the final answer in the last part using the format: 
        <answer>
        \\boxed{{'The final answer goes here.'}}
        </answer>
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        return {
            "messages": messages,
            "tools": TOOLS
        }
        
    
    


if __name__ == "__main__":
    main()