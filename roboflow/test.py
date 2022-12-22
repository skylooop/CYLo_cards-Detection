from roboflow import Roboflow
import pyrallis
from dataclasses import dataclass, field
import os

@dataclass
class Project_CFG:
    dataset_name: str = "mtg_cards"
    save_path: str = field(default="./datasets")
    api_key: str = "d5bTIrPXus6KlhQsCC8V"
    workspace_name: str = "mtg-dwrx0"
    project: str = "mtg-i1iij"
    model: str = "yolov5"

    def __post_init__(self):
        self.location = os.path.join(self.save_path, self.dataset_name)


def download(args) -> None:

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace_name).project(args.project)
    dataset = project.version(1).download(args.model, location=args.location)


if __name__ == "__main__":
    args = pyrallis.parse(config_class=Project_CFG)
    download(args)
