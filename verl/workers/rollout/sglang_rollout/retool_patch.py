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
Monkey patch to enable ReTool parser support in SGLang rollout.
"""

import os
import logging
from .retool_parser import ReToolWrapper

logger = logging.getLogger(__name__)

def patch_sglang_rollout():
    """Apply monkey patch to enable ReTool parser support."""
    
    # Only apply patch if environment variable is set
    if not os.getenv('VERL_USE_RETOOL_PARSER'):
        return
    
    logger.info("Applying ReTool parser patch to SGLang rollout")
    
    # Import the SGLang rollout module
    from . import sglang_rollout
    
    # Store original _setup_tool method
    original_setup_tool = sglang_rollout.SGLangRollout._setup_tool
    
    def patched_setup_tool(self, *args, **kwargs):
        """Patched _setup_tool method that wraps the function call parser."""
        # Call original method
        result = original_setup_tool(self, *args, **kwargs)
        
        # Wrap the function call parser with ReTool wrapper
        if hasattr(self, '_function_call_parser') and self._function_call_parser:
            logger.info("Wrapping function call parser with ReTool support")
            self._function_call_parser = ReToolWrapper(self._function_call_parser)
        
        return result
    
    # Apply the patch
    sglang_rollout.SGLangRollout._setup_tool = patched_setup_tool
    
    logger.info("ReTool parser patch applied successfully")

# Apply patch when module is imported
patch_sglang_rollout()