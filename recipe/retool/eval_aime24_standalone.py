"""
This script is used to evaluate the performance of the model on the AIME 2024 dataset.
这份代码不依赖 verl 框架，通过 openai-compatible api 调用模型，并进行评估。
"""

import argparse
import asyncio
import aiohttp
import openai
import datasets
import json
import re
import time
import itertools
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm.asyncio import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor


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


async def execute_code_async(session: aiohttp.ClientSession, code: str, sandbox_url: str, timeout: int = 30, semaphore: Optional[asyncio.Semaphore] = None) -> Dict[str, Any]:
    """Execute Python code using Sandbox Fusion API asynchronously"""
    async with semaphore if semaphore else asyncio.nullcontext():
        try:
            async with session.post(
                f"{sandbox_url}/run_code",
                json={
                    "code": code,
                    "language": "python",
                    "timeout": timeout
                },
                timeout=aiohttp.ClientTimeout(total=timeout + 5)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                run_result = result.get("run_result", {})
                return {
                    "success": result.get("status") == "Success",
                    "output": run_result.get("stdout", ""),
                    "error": run_result.get("stderr", "")    
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


async def multi_turn_conversation_async(
    session: aiohttp.ClientSession,
    messages: List[Dict], 
    tools: List[Dict], 
    client, 
    args,
    request_semaphore: Optional[asyncio.Semaphore] = None,
    sandbox_semaphore: Optional[asyncio.Semaphore] = None
) -> Dict[str, Any]:
    """Handle multi-turn conversation with tool calls asynchronously"""
    conversation_history = messages.copy()
    turn_count = 0
    
    while turn_count < args.max_turns:
        try:
            # Make API call to get assistant response
            async with request_semaphore if request_semaphore else asyncio.nullcontext():
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.chat.completions.create(
                        model=args.model_name_or_path,
                        messages=conversation_history,
                        tools=tools,
                        tool_choice="auto",
                        max_tokens=args.max_length,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        timeout=args.request_timeout,
                        extra_body={
                            "top_k": args.top_k,
                        }
                    )
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
                # Process each tool call asynchronously
                tool_tasks = []
                for tool_call in assistant_message.tool_calls:
                    if tool_call.function.name == "code_interpreter":
                        # Extract code from arguments
                        args_dict = json.loads(tool_call.function.arguments)
                        code = args_dict.get("code", "")
                        
                        # Create async task for code execution
                        task = execute_code_async(
                            session, code, args.sandbox_url, args.sandbox_timeout, sandbox_semaphore
                        )
                        tool_tasks.append((tool_call.id, task))
                
                # Execute all tool calls concurrently
                tool_results = await asyncio.gather(*[task for _, task in tool_tasks], return_exceptions=True)
                
                # Process results
                for (tool_call_id, _), execution_result in zip(tool_tasks, tool_results):
                    if isinstance(execution_result, Exception):
                        output_content = f"Error: {execution_result}"
                    else:
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
                        "tool_call_id": tool_call_id,
                        "content": output_content
                    })
                
                turn_count += 1
            else:
                # No tool calls, conversation is done
                break
                
        except Exception as e:
            logging.error(f"Error in conversation turn {turn_count}: {e}")
            break
    
    return {
        "conversation_history": conversation_history,
        "final_response": conversation_history[-1]["content"] if conversation_history else "",
        "turn_count": turn_count
    }


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


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


async def process_single_task(
    session: aiohttp.ClientSession,
    problem_id: int,
    sample_id: int,
    item: Dict[str, Any],
    client,
    args,
    request_semaphore: asyncio.Semaphore,
    sandbox_semaphore: asyncio.Semaphore,
    pbar: tqdm
) -> Dict[str, Any]:
    """Process a single (problem_id, sample_id) task"""
    try:
        # Run multi-turn conversation
        conversation_result = await multi_turn_conversation_async(
            session,
            item['messages'], 
            item['tools'], 
            client, 
            args,
            request_semaphore,
            sandbox_semaphore
        )
        
        # Extract predicted answer
        predicted_answer = extract_answer(conversation_result['final_response'])
        
        # Create result
        result = {
            "problem_id": problem_id,
            "sample_id": sample_id,
            "problem": item['problem'],
            "ground_truth": item['answer'],
            "predicted_answer": predicted_answer,
            "final_response": conversation_result['final_response'],
            "conversation_history": conversation_result['conversation_history'],
            "turn_count": conversation_result['turn_count'],
            "correct": predicted_answer.strip() == str(item['answer']).strip()
        }
        
        # Update progress bar with result info
        pbar.set_postfix({
            'Problem': f"{problem_id+1}",
            'Sample': f"{sample_id+1}",
            'Correct': result['correct'],
            'Turns': result['turn_count']
        })
        pbar.update(1)
        
        return result
        
    except Exception as e:
        logging.error(f"Error processing problem {problem_id+1}, sample {sample_id+1}: {e}")
        result = {
            "problem_id": problem_id,
            "sample_id": sample_id,
            "problem": item['problem'],
            "ground_truth": item['answer'],
            "predicted_answer": "",
            "final_response": "",
            "conversation_history": [],
            "turn_count": 0,
            "correct": False,
            "error": str(e)
        }
        
        pbar.set_postfix({
            'Problem': f"{problem_id+1}",
            'Sample': f"{sample_id+1}",
            'Error': 'Yes'
        })
        pbar.update(1)
        
        return result


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
    
    # Concurrency parameters
    parser.add_argument("--max_concurrent_requests", type=int, default=10)
    parser.add_argument("--max_concurrent_sandbox", type=int, default=5)
    parser.add_argument("--request_timeout", type=int, default=300)
    
    args = parser.parse_args()
    return args


async def main_async():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    
    # Create OpenAI client
    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key)
    
    # Load dataset
    dataset = datasets.load_dataset("Maxwell-Jia/AIME_2024", split="train")
    
    def map_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        system_message = "You are a helpful assistant that can solve math problems with interaction Code Interpreter by Python code."
        user_message = f"""Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process.
        
        
        **user question:** 
        {example['Problem']}
        
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
            "problem": example['Problem'],
            "answer": example['Answer']
        }
    
    # Process dataset
    processed_dataset = dataset.map(map_fn)
    
    # Create semaphores for concurrency control
    request_semaphore = asyncio.Semaphore(args.max_concurrent_requests)
    sandbox_semaphore = asyncio.Semaphore(args.max_concurrent_sandbox)
    
    # Create all tasks
    total_tasks = len(processed_dataset) * args.n_samples
    print(f"Starting evaluation on {len(processed_dataset)} problems with {args.n_samples} samples each (total: {total_tasks} tasks)...")
    
    # Create progress bar
    pbar = tqdm(total=total_tasks, desc="Evaluating")
    
    # Create HTTP session
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Create all tasks
        tasks = []
        for problem_id, item in enumerate(processed_dataset):
            for sample_id in range(args.n_samples):
                task = process_single_task(
                    session,
                    problem_id,
                    sample_id,
                    item,
                    client,
                    args,
                    request_semaphore,
                    sandbox_semaphore,
                    pbar
                )
                tasks.append(task)
        
        # Execute all tasks concurrently and collect results as they complete
        results = []
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            results.append(result)
    
    pbar.close()
    
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
    
    # Sort samples within each problem by sample_id
    for pid in problems:
        problems[pid].sort(key=lambda x: x['sample_id'])
    
    # Calculate pass@k using the correct HuggingFace formula
    # Collect number of correct samples per problem
    num_correct_per_problem = []
    for pid, samples in problems.items():
        num_correct = sum(1 for s in samples if s['correct'])
        num_correct_per_problem.append(num_correct)
    
    # Calculate pass@k for different k values
    pass_at_k_results = {}
    k_values = [1, 5, 10, 20, 32]
    
    for k in k_values:
        if k <= args.n_samples:
            pass_at_k_scores = estimate_pass_at_k(args.n_samples, num_correct_per_problem, k)
            pass_at_k = np.mean(pass_at_k_scores)
            pass_at_k_results[k] = pass_at_k
            print(f"Pass@{k}: {pass_at_k:.4f} ({pass_at_k*100:.2f}%)")
    
    # Extract pass@1 for backward compatibility
    pass_at_1 = pass_at_k_results.get(1, 0)
    
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
                "pass_at_1": pass_at_1,
                "max_concurrent_requests": args.max_concurrent_requests,
                "max_concurrent_sandbox": args.max_concurrent_sandbox
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {args.output_path}")


def main():
    """Synchronous wrapper for the async main function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()