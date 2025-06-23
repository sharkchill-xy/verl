# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class ReToolWrapper:
    """
    Wrapper that makes the original SGLang FunctionCallParser compatible with ReTool format.
    This avoids modifying the core VERL code.
    """
    
    def __init__(self, original_parser):
        self.original_parser = original_parser
        self.code_pattern = re.compile(r'<code>\s*```python\s*(.*?)\s*```\s*</code>', re.DOTALL)
    
    def has_tool_call(self, content: str) -> bool:
        """Check if content contains either ReTool code tags or standard function calls."""
        # First check for ReTool format
        if self.code_pattern.search(content):
            return True
        # Then check for standard function calls
        return self.original_parser.has_tool_call(content)
    
    def parse_non_stream(self, content: str) -> Tuple[str, List]:
        """
        Parse both ReTool format and standard function calls.
        """
        # Check if content contains ReTool code blocks
        retool_matches = list(self.code_pattern.finditer(content))
        
        if retool_matches:
            # Convert ReTool format to function calling format
            tool_calls = []
            converted_content = content
            
            # Process matches in reverse order to maintain string indices
            for i, match in enumerate(reversed(retool_matches)):
                code_content = match.group(1).strip()
                
                # Create function call text
                escaped_code = self._escape_code(code_content)
                function_call_text = f'<|tool_call|>\n{{"name": "execute_python", "arguments": {{"code": "{escaped_code}"}}}}\n<|eot_id|>'
                
                # Replace the ReTool code block with function call format
                converted_content = converted_content[:match.start()] + function_call_text + converted_content[match.end():]
                
                # Create a mock tool call object compatible with SGLang
                tool_call = MockToolCall(
                    name="execute_python",
                    parameters={"code": code_content},
                    tool_index=len(retool_matches) - 1 - i
                )
                tool_calls.append(tool_call)
            
            # Reverse to maintain original order
            tool_calls.reverse()
            
            logger.debug(f"Converted {len(tool_calls)} ReTool code blocks to function calls")
            return converted_content, tool_calls
        else:
            # Use original parser for standard function calls
            return self.original_parser.parse_non_stream(content)
    
    def _escape_code(self, code: str) -> str:
        """Escape code for JSON embedding."""
        return code.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    
    def __getattr__(self, name):
        """Delegate other attributes to the original parser."""
        return getattr(self.original_parser, name)


class MockToolCall:
    """Mock tool call object to match SGLang's interface."""
    
    def __init__(self, name: str, parameters: dict, tool_index: int):
        self.name = name
        self.parameters = parameters
        self.tool_index = tool_index