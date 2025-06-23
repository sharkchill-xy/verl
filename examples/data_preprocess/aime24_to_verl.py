#!/usr/bin/env python3
"""
Convert AIME24 evaluation data to VERL format for validation.
"""

import json
import os

def convert_aime24_to_verl():
    input_file = "/home/lixy/workspace/VerlCoder/ReTool/evaluation/dataset/AIME24.jsonl"
    output_dir = "/data2/lixy/VerlCoder/data/aime24"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "val.jsonl")
    
    # Use the exact same ReTool prompt format as SFT training
    retool_prompt = """Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. 
Each code snippet is wrapped with `<code>
```python
code snippet
```
</code>`.
The last part of your response should be in the following format:
<answer>
\\boxed{{'The final answer goes here.'}}
</answer>

*user question:*
Answer the following Math Problem and put the answer in the format of \\boxed{{answer}}

{query}


Remember to place the final answer in the last part using the format: 
<answer>
\\boxed{{'The final answer goes here.'}}
</answer>"""
    
    converted_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Convert to VERL format with ReTool prompt
            converted = {
                "conversations": [
                    {
                        "role": "user",
                        "content": retool_prompt.format(query=data['input'])
                    }
                ],
                "ground_truth": data["output"]
            }
            converted_data.append(converted)
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted_data)} AIME24 examples to {output_file}")

if __name__ == "__main__":
    convert_aime24_to_verl()