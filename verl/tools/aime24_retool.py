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

# ============================================================================
# UNDER REVIEW: ReTool Core Tool Implementation
# Status: Needs comprehensive review for reward calculation logic
# Priority: HIGH - Core functionality affects RL training effectiveness
# Issues: Reward calculation not using answer correctness, memory management
# ============================================================================

import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from typing import Any, Optional, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AIME24ReTool(BaseTool):
    """ReTool for AIME24 mathematical reasoning.
    
    Supports both code execution and answer validation for mathematical problems.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.timeout = config.get("timeout", 30)
        self.max_code_length = config.get("max_code_length", 10000)
        
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth,
            "final_answer": None,
            "code_executions": [],
            "total_reward": 0.0,
            "step_rewards": [],
            "execution_count": 0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute Python code and return output."""
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)
            
        if len(code) > self.max_code_length:
            return "Code too long (>10000 characters)", -0.05, {}
            
        instance = self._instance_dict[instance_id]
        instance["execution_count"] += 1
        
        # Limit number of executions per instance
        if instance["execution_count"] > 20:
            return "Too many code executions (>20)", -0.1, {}
        
        try:
            result = self._run_code_safely(code)
            instance["code_executions"].append({
                "code": code,
                "output": result,
                "timestamp": time.time()
            })
            
            # Small positive reward for successful execution
            step_reward = 0.01
            instance["step_rewards"].append(step_reward)
            
            return result, step_reward, {"execution_success": True}
            
        except Exception as e:
            error_msg = f"Code execution error: {str(e)}"
            instance["code_executions"].append({
                "code": code,
                "output": error_msg,
                "timestamp": time.time()
            })
            
            # Small negative reward for failed execution
            step_reward = -0.02
            instance["step_rewards"].append(step_reward)
            
            return error_msg, step_reward, {"execution_success": False}


    def _run_code_safely(self, code: str) -> str:
        """Safely execute Python code in a subprocess."""
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Run code with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir()
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                return output if output else "Code executed successfully (no output)"
            else:
                return f"Error: {result.stderr.strip()}"
                
        except subprocess.TimeoutExpired:
            return f"Code execution timed out after {self.timeout} seconds"
        except Exception as e:
            return f"Execution error: {str(e)}"
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass

    def _calculate_final_reward(self, instance_id: str) -> float:
        """Calculate final reward based on answer correctness."""
        instance = self._instance_dict[instance_id]
        submitted_answer = instance["final_answer"]
        ground_truth = instance["ground_truth"]
        
        if not submitted_answer or not ground_truth:
            return 0.0
            
        # Parse numeric answer from submitted answer
        submitted_nums = re.findall(r'\d+', submitted_answer)
        ground_truth_nums = re.findall(r'\d+', ground_truth)
        
        if not submitted_nums or not ground_truth_nums:
            return 0.0
            
        try:
            submitted_val = int(submitted_nums[-1])  # Take last number
            ground_truth_val = int(ground_truth_nums[-1])
            
            if submitted_val == ground_truth_val:
                return 1.0  # Correct answer
            else:
                return 0.0  # Incorrect answer
        except:
            return 0.0

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate total reward for the instance."""
        instance = self._instance_dict[instance_id]
        ground_truth = instance.get("ground_truth", "")
        
        # For now, we'll calculate reward based on execution success
        # The final answer evaluation will be handled by the training framework
        step_rewards = instance.get("step_rewards", [])
        if step_rewards:
            # Return average step reward
            return sum(step_rewards) / len(step_rewards)
        
        return 0.0
    
    def _extract_and_evaluate_answer(self, conversation_content: str, ground_truth: str) -> float:
        """Extract final answer from conversation and evaluate correctness."""
        import re
        
        # Look for boxed answer in the conversation
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_matches = re.findall(boxed_pattern, conversation_content)
        
        if boxed_matches:
            # Take the last boxed answer
            final_answer = boxed_matches[-1].strip()
            
            # Extract numeric values
            answer_nums = re.findall(r'\d+', final_answer)
            truth_nums = re.findall(r'\d+', ground_truth)
            
            if answer_nums and truth_nums:
                try:
                    answer_val = int(answer_nums[-1])
                    truth_val = int(truth_nums[-1])
                    
                    if answer_val == truth_val:
                        return 1.0  # Correct answer
                    else:
                        return 0.0  # Incorrect answer
                except:
                    return 0.0
        
        return 0.0  # No valid answer found

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]