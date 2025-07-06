"""
This script is used to evaluate the performance of the model on the AIME 2024 dataset.
这份代码不依赖 verl 框架，通过 openai-compatible api 调用模型，并进行评估。
"""

import argparse
import openai
import datasets
import json
import re
import requests
import time
from typing import Dict, Any, List
from tqdm import tqdm


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


def execute_code(code: str, sandbox_url: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute Python code using Sandbox Fusion API"""
    try:
        response = requests.post(
            f"{sandbox_url}/run_code",
            json={
                "code": code,
                "language": "python",
                "timeout": timeout
            },
            timeout=timeout + 5
        )
        response.raise_for_status()
        result = response.json()
        
        return {
            "success": True,
            "output": result.get("output", ""),
            "error": result.get("error", "")
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e)
        }


def extract_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Extract tool calls from assistant response"""
    # Look for JSON blocks that might contain tool calls
    # This is a simplified parser - in practice, you might need more robust parsing
    tool_calls = []
    
    # Pattern to match function calls in the response
    pattern = r'```python\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for i, code in enumerate(matches):
        tool_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "arguments": json.dumps({"code": code.strip()})
            }
        })
    
    return tool_calls


def multi_turn_conversation(messages: List[Dict], tools: List[Dict], client, args) -> Dict[str, Any]:
    """Handle multi-turn conversation with tool calls"""
    conversation_history = messages.copy()
    turn_count = 0
    
    while turn_count < args.max_turns:
        try:
            # Make API call to get assistant response
            response = client.chat.completions.create(
                model=args.model_name_or_path,
                messages=conversation_history,
                tools=tools,
                tool_choice="auto",
                max_tokens=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            
            assistant_message = response.choices[0].message
            
            # Add assistant message to conversation history
            assistant_msg = {
                "role": "assistant", 
                "content": assistant_message.content or ""
            }
            
            # Add tool_calls if present
            if assistant_message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in assistant_message.tool_calls
                ]
            
            conversation_history.append(assistant_msg)
            
            # Check if there are tool calls
            if assistant_message.tool_calls:
                # Process each tool call
                for tool_call in assistant_message.tool_calls:
                    if tool_call.function.name == "code_interpreter":
                        # Extract code from arguments
                        args_dict = json.loads(tool_call.function.arguments)
                        code = args_dict.get("code", "")
                        
                        # Execute the code
                        execution_result = execute_code(code, args.sandbox_url, args.sandbox_timeout)
                        
                        # Format output
                        if execution_result["success"]:
                            output_content = execution_result["output"]
                            if execution_result["error"]:
                                output_content += f"\nWarning: {execution_result['error']}"
                        else:
                            output_content = f"Error: {execution_result['error']}"
                        
                        # Add tool response to conversation
                        conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": output_content
                        })
                
                turn_count += 1
            else:
                # No tool calls, conversation is done
                break
                
        except Exception as e:
            print(f"Error in conversation turn {turn_count}: {e}")
            break
    
    return {
        "conversation_history": conversation_history,
        "final_response": conversation_history[-1]["content"] if conversation_history else "",
        "turn_count": turn_count
    }


def extract_answer(text: str) -> str:
    """Extract the final answer from assistant response"""
    # Look for answer in boxed format
    pattern = r'<answer>\s*\\boxed\{([^}]+)\}\s*</answer>'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for boxed format anywhere
    pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return ""


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
    
    # Sandbox Fusion parameters
    parser.add_argument("--sandbox_url", type=str, default="http://210.28.135.36:8080")
    parser.add_argument("--sandbox_timeout", type=int, default=30)
    
    # Multi-turn parameters
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=1)
    
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
            "tools": TOOLS,
            "problem": example['problem'],
            "answer": example['answer']
        }
    
    # Process dataset
    processed_dataset = dataset.map(map_fn)
    
    results = []
    
    print(f"Starting evaluation on {len(processed_dataset)} problems...")
    
    for i, item in enumerate(tqdm(processed_dataset)):
        print(f"\n--- Problem {i+1}/{len(processed_dataset)} ---")
        print(f"Problem: {item['problem'][:100]}...")
        
        for sample in range(args.n_samples):
            try:
                # Run multi-turn conversation
                conversation_result = multi_turn_conversation(
                    item['messages'], 
                    item['tools'], 
                    client, 
                    args
                )
                
                # Extract predicted answer
                predicted_answer = extract_answer(conversation_result['final_response'])
                
                # Save result
                result = {
                    "problem_id": i,
                    "sample_id": sample,
                    "problem": item['problem'],
                    "ground_truth": item['answer'],
                    "predicted_answer": predicted_answer,
                    "final_response": conversation_result['final_response'],
                    "conversation_history": conversation_result['conversation_history'],
                    "turn_count": conversation_result['turn_count'],
                    "correct": predicted_answer.strip() == str(item['answer']).strip()
                }
                
                results.append(result)
                
                print(f"Sample {sample+1}: Predicted={predicted_answer}, Ground Truth={item['answer']}, Correct={result['correct']}")
                
            except Exception as e:
                print(f"Error processing problem {i+1}, sample {sample+1}: {e}")
                result = {
                    "problem_id": i,
                    "sample_id": sample,
                    "problem": item['problem'],
                    "ground_truth": item['answer'],
                    "predicted_answer": "",
                    "final_response": "",
                    "conversation_history": [],
                    "turn_count": 0,
                    "correct": False,
                    "error": str(e)
                }
                results.append(result)
    
    # Calculate metrics
    total_samples = len(results)
    correct_samples = sum(1 for r in results if r['correct'])
    accuracy = correct_samples / total_samples if total_samples > 0 else 0
    
    print(f"\n=== Evaluation Results ===")
    print(f"Total samples: {total_samples}")
    print(f"Correct samples: {correct_samples}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Group by problem for pass@k calculation
    problems = {}
    for result in results:
        pid = result['problem_id']
        if pid not in problems:
            problems[pid] = []
        problems[pid].append(result)
    
    # Calculate pass@1 and pass@k
    pass_at_1 = sum(1 for pid, samples in problems.items() if any(s['correct'] for s in samples)) / len(problems)
    
    print(f"Pass@1: {pass_at_1:.4f} ({pass_at_1*100:.2f}%)")
    
    if args.n_samples > 1:
        # Calculate pass@k for different k values
        for k in [min(args.n_samples, k) for k in [5, 10, 20, 32]]:
            if k <= args.n_samples:
                pass_at_k = sum(1 for pid, samples in problems.items() 
                               if any(s['correct'] for s in samples[:k])) / len(problems)
                print(f"Pass@{k}: {pass_at_k:.4f} ({pass_at_k*100:.2f}%)")
    
    # Save results
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model": args.model_name_or_path,
                "n_samples": args.n_samples,
                "max_turns": args.max_turns,
                "total_problems": len(problems),
                "total_samples": total_samples,
                "accuracy": accuracy,
                "pass_at_1": pass_at_1
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()