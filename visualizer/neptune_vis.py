from abc import abstractmethod
from typing import Any, Union
import numpy as np
import torch
import neptune.new as neptune
from utils.config import Config
from mmengine.visualization import BaseVisBackend
from mmengine.visualization.vis_backend import force_init_env
run = neptune.init(
    project="vase/PyTorchtest",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZjY2Mzk4OS0zNDdhLTQwYzEtYTQ3Mi00NmI4ZjEyMGJhOTMifQ==",
)  # your credentials

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].log(0.9 ** epoch)

run["eval/f1_score"] = 0.66

run.stop()


class BaseVisBackend(BaseVisBackend):
    """Base class for visualization backend.

    All backends must inherit ``BaseVisBackend`` and implement
    the required functions.

    Args:
        save_dir (str, optional): The root directory to save
            the files produced by the backend.
    """

    def __init__(self, save_dir: str):
        self._save_dir = save_dir
        self._env_initialized = False

    @force_init_env
    def experiment(self) -> Any:
        """Return the experiment object associated with this visualization
        backend.

        The experiment attribute can get the visualization backend, such as
        wandb, tensorboard. If you want to write other data, such as writing a
        table, you can directly get the visualization backend through
        experiment.
        """
        return self._neptune


    def _init_env(self) -> Any:
        """Setup env for VisBackend."""
        run = neptune.init(
            project="vase/PyTorchtest",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZjY2Mzk4OS0zNDdhLTQwYzEtYTQ3Mi00NmI4ZjEyMGJhOTMifQ==",
        )  # your credentials
        self._neptune = run
        params = {"learning_rate": 0.001, "optimizer": "Adam"}
        run["parameters"] = params
        pass




    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config.

        Args:
            config (Config): The Config object
        """
        pass

    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict],
                  **kwargs) -> None:
        """Record the model graph.

        Args:
            model (torch.nn.Module): Model to draw.
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        pass

    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
        """
        pass

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar.

        Args:
            name (str): The scalar identifier.
            value (int, float): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        pass

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars' data.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): The scalar's data will be
                saved to the `file_path` file at the same time
                if the `file_path` parameter is specified.
                Default to None.
        """
        pass

    def close(self) -> None:
        """close an opened object."""
        pass