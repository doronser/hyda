import os

from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger

from brain_mri.data import MRIDataModule
from hyda.utils import namespace_to_dict


class SaveConfigWandB(SaveConfigCallback):
    def __init__( self, *args,  **kwargs) -> None:
        super().__init__(save_to_log_dir=False, *args, **kwargs)

    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.logger is not None:
            logger : WandbLogger = trainer.logger
            data = namespace_to_dict(self.config.data) if logger.name == 'mri_uda' else self.config.data
            logger.experiment.config.update(dict(data=data, model=namespace_to_dict(self.config.model)))

class LoggerSaveConfigCallback(SaveConfigCallback):
    def __init__( self, *args,  **kwargs) -> None:
        super().__init__(save_to_log_dir=False, *args, **kwargs)
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
            trainer.logger.log_hyperparams({"config": config})

class CLI(LightningCLI):
    def add_default_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument('--seed_everything', type=int, help='Seed for reproducibility')

        # # Define data.task_weights as a class group
        # parser.add_class_arguments(LightningDataModule, "data")
        # parser.add_class_arguments(LightningModule, "model")
        #
        # parser.link_arguments("data.task_weights", "model.task_weights", apply_on="instantiate")

        parser.add_argument("experiment_name", type=str, help="Name of the experiment")
        parser.add_argument('--trainer.logger.init_args.name', type=str, help='W&B experiment name')
        parser.link_arguments(source="experiment_name", target="trainer.logger.init_args.name", apply_on="parse")
        parser.add_lightning_class_args(ModelCheckpoint, "chkpt_callback")
        parser.link_arguments(source="experiment_name", target="chkpt_callback.filename", apply_on="parse",
                              compute_fn=lambda x: x + "_{epoch:03d}")
        parser.link_arguments(source="experiment_name", target="chkpt_callback.dirpath", apply_on="parse",
                              compute_fn=lambda x: f"{os.environ['HOME']}/ckpts/{x}")

def cli_main():
    cli = CLI(model_class=LightningModule, subclass_mode_model=True, save_config_callback=SaveConfigWandB)


if __name__ == "__main__":
    cli_main()