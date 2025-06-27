
"""
Preprocess AIME24 dataset to multi-turn conversation format for evaluation.
This script converts BytedTsinghua-SIA/AIME-2024 dataset to the same format 
as swordfaith/ReTool-SFT-multi-turn for consistent evaluation.
"""


import os 
import argparse
import datasets 

TOOLS = [
    {
        "function": {
            "description": "Execute Python code to perform calculations, data analysis, or other computational tasks.",
            "name": "code_interpreter",
            "parameters": {
                "properties": {
                    "code": {
                        "description": "The Python code to be executed. This can include calculations, data manipulation, or any valid Python code.",
                        "type": "string"
                    },
                    "language": {
                        "default": "python",
                        "description": "The programming language of the code. Currently only Python is supported.",
                        "type": "string"
                    }
                },
                "required": [
                    "code"
                ],
                "type": "object"
            }
        },
        "type": "function"
    }
]


def create_retool_prompt_messages(question: str):
    """Create messages in ReTool format for AIME24 evaluation."""
    # system_prompt = (
    #     "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. "
    #     "The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. "
    #     "The Python code should be complete scripts, including necessary imports. \n"
    #     "Each code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\n"
    #     "The last part of your response should be in the following format:\n"
    #     "<answer>\n\\boxed{'The final answer goes here.'}\n</answer>\n\n"
    #     "*user question:*\n"
    #     "Answer the following Math Problem and put the answer in the format of \\boxed{answer}\n\n"
    #     f"{question}\n\n\n"
    #     "Remember to place the final answer in the last part using the format: \n"
    #     "<answer>\n\\boxed{'The final answer goes here.'}\n</answer>"
    # )
    system_prompt = "You are a helpful assistant that can solve math problems with interaction Code Interpreter by Python code."
    user_question = ("Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process.\n\n"
                     "**user question:**\n"
                     f"{question}\n\n\n"
                     "Remember to place the final answer in the last part using the format: \n<answer>\n\\boxed{'The final answer goes here.'}\n</answer>"
                     )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/aime24_eval_multiturn")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    # Load AIME24 dataset
    dataset = datasets.load_dataset("BytedTsinghua-SIA/AIME-2024", split="train")
    
    # Convert to multi-turn conversation format
    def process_fn(example):
        messages = create_retool_prompt_messages(example["extra_info"]["raw_problem"])
        return {
            "data_source": "AIME-2024",
            "messages": messages,
            "tools": TOOLS,
            "enable_thinking": False,
            "extra_info": {
                "split": "test",
                "index": example["extra_info"]["index"]
            }
        }

    dataset = dataset.map(process_fn)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet files
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Handle HDFS if specified
    if hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(hdfs_dir)
            copy(src=local_dir, dst=hdfs_dir)
        except ImportError:
            print("HDFS is not available. Skipping HDFS copy.")

    print(f"Evaluation dataset size: {len(dataset)}")
    print(f"Data saved to {local_dir}")


if __name__ == "__main__":
    main()