import os
import json
from typing import Any, Optional, List, Dict


class Logger(object):
    def __init__(self, path: str, resume: bool, wandb: Optional[Any] = None) -> None:
        """Logging metrics.

        Args:
            path (str): Path to save the logs
            resume (bool): If the training is resumed, previous logs are loaded from thhe path and continued to be appended.
            wandb (Optional[Any], optional): Wandb instance, if used. Defaults to None.
        """
        self.metrics_path = os.path.join(path, "metrics.jsonl")
        self.interventions_path = os.path.join(path, "interventions.csv")
        if not resume:

            try:
                os.unlink(self.metrics_path)
                os.unlink(self.interventions_path)
            except FileNotFoundError:
                pass
        self.wandb = wandb

    def log_interventions(self, iteration: int, nodes: List[int], values: List[float]) -> None:
        with open(self.interventions_path, "a") as f:
            for (node, value) in zip(nodes, values):
                f.write(f'{iteration},{node},{value}\n')
            f.close()

    def log_metrics(self, scalars: Dict[str, Any]) -> None:
        """Log metrics.

        Args:
            scalars (Dict[str, Any]): All the scalars to be logged.
        """
        for key, value in scalars.items():
            if self.wandb is not None:
                self.wandb.log({key: value})
            print(f"{key}: {value}")

        with open(self.metrics_path, "a") as f:
            f.write(json.dumps({**scalars}) + "\n")
            f.close()
