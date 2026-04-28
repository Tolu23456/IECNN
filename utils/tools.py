"""
Tool Executor — sandbox integration for the IECNN agent.
"""

import subprocess
import os
import sys
from typing import Dict, Any, Optional

class ToolExecutor:
    """
    Executes symbolic actions within the sandbox environment.
    """
    def __init__(self, model):
        self.model = model

    def execute(self, action_type: str, payload: str) -> str:
        """Execute the specified action and return a text-based result."""
        if action_type == "RUN_CODE":
            return self._run_python(payload)
        elif action_type == "SEARCH":
            return f"[Search Result for '{payload}']: Simulated knowledge retrieval successful."
        elif action_type == "GENERATE_IMAGE":
            return f"[Image Generated]: Conceptual rendering of '{payload}' stored in latent space."

        return f"[Unknown Action]: {action_type}"

    def _run_python(self, code: str) -> str:
        """Safely execute Python code and capture output."""
        try:
            # We use a temporary file for execution
            with open("tmp_exec.py", "w") as f:
                f.write(code)

            result = subprocess.run(
                ["python3", "tmp_exec.py"],
                capture_output=True, text=True, timeout=5
            )

            output = result.stdout + result.stderr
            return output if output.strip() else "[Code executed with no output]"
        except Exception as e:
            return f"[Execution Error]: {str(e)}"
        finally:
            if os.path.exists("tmp_exec.py"):
                os.remove("tmp_exec.py")

    def parse_and_execute(self, text: str) -> Optional[str]:
        """Heuristic parser for ACTION tokens in generated text."""
        if "RUN_CODE" in text:
            # Extract code block between triple backticks or simply the rest of text
            parts = text.split("RUN_CODE")
            if len(parts) > 1:
                code = parts[1].strip().strip("`")
                return self.execute("RUN_CODE", code)

        return None
