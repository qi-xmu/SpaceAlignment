"""
NOTE:
    参数解析

Author: qi
"""

import argparse


class DatasetArgsParser:
    dataset: str | None = None
    group: str | None = None
    unit: str | None = None
    output: str | None = None

    # NOTE  Not default parameters
    regen: bool = False
    visual: bool = False

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Dataset Arguments Parser")
        self.parser.add_argument("-d", "--dataset", type=str, help="Path to dataset")
        self.parser.add_argument("-g", "--group", type=str, help="Path to group path")
        self.parser.add_argument("-u", "--unit", type=str, help="Path to unit path")
        self.parser.add_argument("-o", "--output", type=str, help="Path to output")
        self.parser.add_argument(
            "-r", "--regen", help="Regenerate Dataset", action="store_true"
        )
        self.parser.add_argument(
            "-v", "--visual", action="store_true", help="Visualize"
        )

    def parse(self):
        self.parser.parse_args()
        self.args = self.parser.parse_args()

        print("命令行参数:")
        for k, v in self.args.__dict__.items():
            print(f"- {k}: {v}")
            setattr(self, k, v)

        return self
