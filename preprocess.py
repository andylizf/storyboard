import os
from pathlib import Path
import numpy as np
import threading
import time
from network import *


def empty_dir(path):
    return len([p for p in path.glob("**/*")]) == 0


class DataProject:
    def __init__(self, path: Path):
        self.path = path
        self.video = next(path.glob("*.mp4"))
        self.txt = next(path.glob("*.txt"))

    def calc(self):
        self.frames = self.path / "frames"

        if not self.frames.is_dir():
            self.frames.mkdir()
        if empty_dir(self.frames):
            os.system(
                f'ffmpeg -i {self.video} -s {WIDTH}*{HEIGHT} -start_number 0 {self.frames / "%6d.jpg"} > {(self.path / "frames.log").resolve()}'
            )

    def __str__(self):
        return f"{{path: {self.path}, video: {self.video}, txt: {self.txt}}}"


datas = [DataProject(x) for x in sorted(Path("data/").iterdir()) if x.is_dir()]
