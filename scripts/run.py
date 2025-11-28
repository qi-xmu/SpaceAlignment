import subprocess
from pathlib import Path


class Executable:
    exec_name: str
    _run_cmd: list[str]

    def __init__(self, exec_name: str | Path):
        pass
