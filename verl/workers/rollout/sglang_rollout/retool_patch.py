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
ReTool Parser Injection Mechanism for VERL SGLang Rollout

This module implements a non-intrusive monkey patch system to inject ReTool format 
support into VERL's SGLang rollout without modifying core framework code.

## Injection Mechanism Overview

The ReTool parser injection uses a runtime monkey patch approach that:
1. Conditionally activates based on environment variable
2. Wraps the original function call parser with ReTool support
3. Maintains full backward compatibility with existing functionality

## Usage Instructions

### 1. Environment Variable Activation
Set the environment variable to enable ReTool parser support:
```bash
export VERL_USE_RETOOL_PARSER=1
```

### 2. Training Script Integration
Include the environment variable in your training script:
```bash
#!/bin/bash
export VERL_USE_RETOOL_PARSER=1
python3 -m verl.trainer.main_ppo --config-path="config" --config-name="retool_config"
```

### 3. Automatic Injection Process
The injection happens automatically when importing VERL modules:
- Module import triggers patch_sglang_rollout()
- Original SGLangRollout._setup_tool method is wrapped
- Function call parser gets wrapped with ReToolWrapper
- ReTool format support becomes available transparently

## Technical Details

### Supported Formats
- **ReTool Format**: `<code>```python\ncode_here\n```</code>`
- **Standard Format**: Original SGLang function calling format
- **Hybrid Support**: Both formats can be used simultaneously

### Parser Precedence
1. First checks for ReTool code block format
2. Falls back to standard function call parsing
3. Maintains all original parser functionality

### Error Handling
- ReTool parsing errors gracefully fall back to standard parser
- Original parser behavior preserved if patch fails
- No impact on non-ReTool workloads

## Design Benefits

### Non-Intrusive Design
- ✅ Zero modifications to VERL core code
- ✅ Optional activation via environment variable
- ✅ Complete backward compatibility
- ✅ Easy to disable or remove

### Research-Friendly
- ✅ Rapid prototyping and iteration
- ✅ Quick testing of new parser formats
- ✅ Safe experimentation without breaking existing code
- ✅ Hot-swappable functionality

### Maintenance Considerations
- ⚠️ Monitor VERL version compatibility
- ⚠️ Test patch behavior after VERL updates
- ⚠️ Document any version-specific requirements
- ⚠️ Consider upstreaming successful features

## Example Usage in Research

```python
# Model generates ReTool format response
response = \"\"\"Let me calculate this step by step.
<code>
```python
result = 2 + 2
print(f"The answer is: {result}")
```
</code>
The final answer is 4.\"\"\"

# ReToolWrapper automatically detects and processes the code block
# Converts to standard function call format for execution
# Returns result integrated into the conversation flow
```
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