#!/usr/bin/env python3
"""
Quick test script for the parallel evaluation system
"""

import asyncio
import aiohttp
import time
from eval_aime24_standalone import execute_code_async

async def test_sandbox_parallel():
    """Test parallel sandbox execution"""
    print("Testing parallel sandbox execution...")
    
    # Test codes
    test_codes = [
        "print('Hello 1'); import time; time.sleep(0.5); print('Done 1')",
        "print('Hello 2'); import time; time.sleep(1.0); print('Done 2')",
        "print('Hello 3'); import time; time.sleep(0.2); print('Done 3')",
        "print('Hello 4'); import time; time.sleep(0.8); print('Done 4')",
        "print('Hello 5'); import time; time.sleep(0.3); print('Done 5')",
    ]
    
    sandbox_url = "http://210.28.135.36:8080"
    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, code in enumerate(test_codes):
            task = execute_code_async(session, code, sandbox_url, 30, semaphore)
            tasks.append((i, task))
        
        print(f"Created {len(tasks)} tasks, executing in parallel...")
        
        # Execute all tasks and collect results as they complete
        for i, completed_task in enumerate(asyncio.as_completed([task for _, task in tasks])):
            result = await completed_task
            elapsed = time.time() - start_time
            print(f"Task completed at {elapsed:.2f}s: {result['output'].strip() if result['success'] else result['error']}")
    
    total_time = time.time() - start_time
    print(f"\nAll tasks completed in {total_time:.2f}s")
    print("If this was sequential, it would take ~2.8s")
    print("Parallel execution should be much faster!")

if __name__ == "__main__":
    asyncio.run(test_sandbox_parallel())