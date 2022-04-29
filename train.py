import torch
import argparse
from pytorch_lightning import Trainer
import multiprocessing
from src.config import MusicVAEConfig, EncodingName
from src.utils import InstrumentTarget
from src.datasets import MidiDataModule
from src.models import LtMusicVAE


def main(args: argparse.Namespace):
    config = MusicVAEConfig(args.dataset_dir, args.encoding_name, args)
    data = MidiDataModule(config)
    model = LtMusicVAE(data.tokenizer, args)

    trainer = Trainer(gpus=1,
                      max_epochs=args.epochs,
                      progress_bar_refresh_rate=20)

    if trainer.logger:
        trainer.logger.log_hyperparams(args)
    trainer.fit(model, data)
    trainer.test(dataloaders=data)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Basic settings and hyper-parameters for training the model")
    argparser.add_argument("-e", "--encoding_name", type=EncodingName,
                           default="remi",
                           help=f"encoding methods: {str(EncodingName).replace('typing.Literal', '')}")
    argparser.add_argument("-d", "--dataset_dir", type=str,
                           default="/this/is/a/test/data/",
                           help="data directory for training")
    argparser.add_argument("-t", "--target_instrument",
                           type=InstrumentTarget,
                           default="melody",
                           help=f"select target inst: {str(InstrumentTarget).replace('typing.Literal', '')}")
    argparser.add_argument("--lr",  type=float,
                           default=1e-4, help="learning rate")
    argparser.add_argument("--epochs", type=int,
                           default=20, help="n of epochs")
    args = argparser.parse_args()
    main(args)
