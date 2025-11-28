import subprocess
from pathlib import Path

from base.args_parser import DatasetArgsParser


class Executable:
    exec_name: str
    _run_cmd: list[str]

    def __init__(self, exec_name: str | Path, *extra):
        if isinstance(exec_name, Path):
            assert exec_name.exists()
            self.exec_name = exec_name.as_posix()
        else:
            self.exec_name = exec_name

        self._run_cmd = []
        self._run_cmd.append(self.exec_name)
        self._run_cmd.extend(extra)

    def add_params(self, *, op: str | None = None, val: str | None = None):
        if op:
            self._run_cmd.append(op)
        if val:
            self._run_cmd.append(val)

    def run(self):
        print("运行：", self._run_cmd)
        try:
            res = subprocess.run(self._run_cmd, check=True)
            print(f"> {self.exec_name} 执行完成")
            return res
        except subprocess.CalledProcessError as e:
            print(f"! Error occurred while running {self.exec_name}: {e}")
            return e


class PyExecutable(Executable):
    def __init__(self, script_name: str, *extra):
        super().__init__("uv", "run", script_name, *extra)


# if __name__ == "__main__":
#     dap = DatasetArgsParser().parse()

#     data_check = PyExecutable("DataCheck.py", "-u", args.)

#     pass
