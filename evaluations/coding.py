"""Best-effort isolated execution for code examples produced by the model.

This runner never executes code in the application process.  It uses a fresh
temporary directory, a minimal environment, a timeout, isolated Python mode,
and a conservative AST import deny-list.  Strong OS sandboxing should still be
provided by CI/container infrastructure for untrusted public evaluations.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


DENIED_IMPORTS = {
    "ctypes", "multiprocessing", "os", "pathlib", "pty", "resource",
    "shutil", "signal", "socket", "subprocess", "sys", " tempfile",
    "threading", "urllib", "requests", "httpx", "importlib",
}


@dataclass
class CodeExecutionResult:
    status: str
    stdout: str = ""
    stderr: str = ""
    return_code: Optional[int] = None
    duration_ms: float = 0.0
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _blocked_import(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax_error: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".", 1)[0]]
        else:
            continue
        for name in names:
            if name in DENIED_IMPORTS:
                return f"blocked_import: {name}"
    return None


def run_python_snippet(code: str, expected_stdout: Optional[str] = None, timeout_seconds: float = 3.0) -> CodeExecutionResult:
    """Execute a small Python snippet outside the evaluator process."""
    blocked = _blocked_import(code)
    if blocked:
        return CodeExecutionResult(status="rejected", reason=blocked)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tisw-eval-") as temp_dir:
        script = Path(temp_dir) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        safe_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        command = [sys.executable, "-I", "-S", str(script)]
        try:
            completed = subprocess.run(
                command,
                cwd=temp_dir,
                env=safe_env,
                capture_output=True,
                text=True,
                timeout=max(0.1, timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CodeExecutionResult(
                status="timeout",
                stdout=(exc.stdout or "")[-4000:],
                stderr=(exc.stderr or "")[-4000:],
                duration_ms=(time.perf_counter() - started) * 1000,
                reason=f"exceeded {timeout_seconds:.2f}s",
            )
        except OSError as exc:
            return CodeExecutionResult(
                status="error",
                duration_ms=(time.perf_counter() - started) * 1000,
                reason=f"runner_error: {type(exc).__name__}",
            )

    stdout = completed.stdout[-4000:]
    stderr = completed.stderr[-4000:]
    status = "passed" if completed.returncode == 0 else "failed"
    reason = None
    if expected_stdout is not None and stdout.strip() != expected_stdout.strip():
        status = "failed"
        reason = "stdout_mismatch"
    return CodeExecutionResult(
        status=status,
        stdout=stdout,
        stderr=stderr,
        return_code=completed.returncode,
        duration_ms=(time.perf_counter() - started) * 1000,
        reason=reason,
    )
