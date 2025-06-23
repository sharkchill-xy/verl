#!/usr/bin/env python3
"""
Test script for ReTool parser integration.
"""

import sys
import os

# Add the VERL path
sys.path.insert(0, '/home/lixy/workspace/VerlCoder/verl')

from verl.workers.rollout.sglang_rollout.retool_parser import ReToolWrapper

def test_retool_parser():
    """Test the ReTool parser wrapper."""
    
    # Mock original parser
    class MockOriginalParser:
        def has_tool_call(self, content):
            return False
        
        def parse_non_stream(self, content):
            return content, []
    
    # Create wrapper
    wrapper = ReToolWrapper(MockOriginalParser())
    
    # Test ReTool format
    test_content = """Let me calculate this step by step.

<code>
```python
x = 5 + 3
print(f"The result is {x}")
```
</code>

The answer is 8."""

    print("Testing ReTool format detection...")
    has_tool_call = wrapper.has_tool_call(test_content)
    print(f"Has tool call: {has_tool_call}")
    
    if has_tool_call:
        print("\nParsing content...")
        normed_content, tool_calls = wrapper.parse_non_stream(test_content)
        
        print(f"Number of tool calls: {len(tool_calls)}")
        for i, tool_call in enumerate(tool_calls):
            print(f"Tool call {i}:")
            print(f"  Name: {tool_call.name}")
            print(f"  Parameters: {tool_call.parameters}")
            print(f"  Index: {tool_call.tool_index}")
        
        print(f"\nNormalized content:\n{normed_content}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_retool_parser()