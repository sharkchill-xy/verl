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

"""
ReTool format detector for SGLang function call parser.
"""

import json
import re
from typing import List

try:
    from sglang.srt.function_call_parser import BaseFormatDetector, StreamingParseResult, ToolCallItem, StructureInfo
    from sglang.srt.openai_api.protocol import Tool
except ImportError:
    # Fallback imports for different SGLang versions
    from sglang.srt.function_call.function_call_parser import BaseFormatDetector, StreamingParseResult, ToolCallItem, StructureInfo
    from sglang.srt.openai_api.protocol import Tool

import logging

logger = logging.getLogger(__name__)


class ReToolDetector(BaseFormatDetector):
    """
    Detector for ReTool format.
    Assumes function call format:
      <code>
      ```python
      python_code_here
      ```
      </code>
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "<code>"
        self.eot_token = "</code>"
        self.code_pattern = re.compile(r'<code>\s*```python\s*(.*?)\s*```\s*</code>', re.DOTALL)

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a ReTool format code block."""
        return bool(self.code_pattern.search(text))

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses ReTool code blocks in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: StreamingParseResult with normal text and parsed calls.
        """
        matches = list(self.code_pattern.finditer(text))
        if not matches:
            return StreamingParseResult(normal_text=text, calls=[])

        # Extract normal text (everything before the first code block)
        first_match = matches[0]
        normal_text = text[:first_match.start()].strip()

        # Parse all code blocks into tool calls
        calls = []
        for i, match in enumerate(matches):
            code_content = match.group(1).strip()
            
            # Create tool call item for execute_python function
            tool_call_item = ToolCallItem(
                tool_index=i,
                name="execute_python",
                parameters=json.dumps({"code": code_content})
            )
            calls.append(tool_call_item)
            
            logger.debug(f"Parsed ReTool code block {i}: {code_content[:50]}...")

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def detect_and_parse_stream(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Streaming parsing: Incrementally processes new text chunks.
        
        :param new_text: The new text chunk to process.
        :param tools: List of available tools.
        :return: StreamingParseResult with incremental results.
        """
        # Add new text to buffer
        self._buffer += new_text
        
        # Try to parse complete code blocks from buffer
        complete_matches = list(self.code_pattern.finditer(self._buffer))
        
        if not complete_matches:
            # Check if we have a partial code block starting
            if self.bot_token in self._buffer:
                # Keep buffering, don't return text yet
                return StreamingParseResult(normal_text="", calls=[])
            else:
                # No code blocks, return accumulated text as normal
                result_text = self._buffer
                self._buffer = ""
                return StreamingParseResult(normal_text=result_text, calls=[])
        
        # Find the last complete match
        last_match = complete_matches[-1]
        
        # Extract normal text up to first code block
        normal_text = ""
        calls = []
        
        if complete_matches:
            first_match = complete_matches[0]
            normal_text = self._buffer[:first_match.start()].strip()
            
            # Parse all complete code blocks
            for i, match in enumerate(complete_matches):
                code_content = match.group(1).strip()
                
                tool_call_item = ToolCallItem(
                    tool_index=len(self._calls) + i,  # Use accumulated call count
                    name="execute_python", 
                    parameters=json.dumps({"code": code_content})
                )
                calls.append(tool_call_item)
        
        # Keep any remaining text after last complete match
        self._buffer = self._buffer[last_match.end():]
        
        # Update accumulated calls
        self._calls.extend(calls)
        
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self):
        """
        Returns a function that generates StructureInfo for ReTool format.
        """
        def get_info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=f'<code>\n```python\n# {name}\n',
                end='\n```\n</code>',
                trigger='<code>'
            )
        return get_info

    def reset_state(self):
        """Reset the detector state for new parsing session."""
        super().reset_state()
        self._calls = []