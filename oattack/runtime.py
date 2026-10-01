"""Select the active Python, or an explicitly configured evaluation runtime."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def python_command(environment: str | None = None) -> list[str]:
    """Use CONDA_BIN only when requested; ordinary virtual environments work too."""
    environment = environment or os.environ.get("CONDA_ENV", "mattack")
    override = os.environ.get(f"OATTACK_PYTHON_{environment.upper()}")
    if override:
        executable = Path(override).expanduser()
        if not executable.is_file():
            raise FileNotFoundError(f"Python executable does not exist: {executable}")
        return [str(executable)]
    conda = os.environ.get("CONDA_BIN")
    if conda:
        return [str(Path(conda).expanduser()), "run", "--no-capture-output", "-n", environment, "python"]
    return [sys.executable]
