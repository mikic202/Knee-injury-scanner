from abc import ABC, abstractmethod
import wandb


class DataLogger(ABC):

    @staticmethod
    @abstractmethod
    def log(data):
        pass


class ConsoleLogger(DataLogger):
    @staticmethod
    def log(data):
        print(data)


class WandbLogger(DataLogger):
    RUN = None

    def __init__(self, project_name, run_name=None, config=None):
        if WandbLogger.RUN is None:
            WandbLogger.RUN = wandb.init(
                project=project_name, name=run_name, config=config
            )

    @staticmethod
    def log(data):
        wandb.log(data)

    @staticmethod
    def finish():
        if WandbLogger.RUN is not None:
            WandbLogger.RUN.finish()
            WandbLogger.RUN = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()
