import argparse
import multiprocessing
import os
from typing import get_args

import torch
from pytorch_lightning import Trainer

from src.config import EncodingName, MusicVAEConfig
from src.datasets import MidiDataModule
from src.models import LtMusicVAE
from src.utils import InstrumentTarget


def _check_encoding_methods(e: str) -> str:
    ve = get_args(EncodingName)
    assert e in ve, f"`encoding_name` should be one of {ve}"
    return e


def _check_inst_target(e: str) -> str:
    ve = get_args(InstrumentTarget)
    assert e in ve, f"`target_instrument` should be one of {ve}"
    return e


def main(args: argparse.Namespace):
    config = MusicVAEConfig(args)
    tokenizer = config.tokenizer_type()
    data = MidiDataModule(tokenizer, config)
    model = LtMusicVAE(tokenizer, config)

    trainer = Trainer(gpus=1,
                      max_epochs=config.hparams.epochs,
                      default_root_dir=(f"model_checkpoints/{config.hparams.target_instrument}" +
                                        f"__{config.tokenizer_type}" +
                                        f"__{config.hparams.max_midi_files}midis"),
                      progress_bar_refresh_rate=20)

    if trainer.logger:
        trainer.logger.log_hyperparams(args)
    trainer.fit(model, data)
    trainer.test(dataloaders=data)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Basic settings and hyper-parameters for training the model")
    argparser.add_argument("-e", "--encoding_method", type=_check_encoding_methods,
                           default="remi",
                           help=f"encoding methods: {get_args(EncodingName)}")
    argparser.add_argument("-d", "--dataset_dir", type=str,
                           default=os.path.expanduser(
                               "~/datasets/meta_midi_dataset/MMD_MIDI/"),
                           help="data directory for training")
    argparser.add_argument("-t", "--target_instrument",
                           type=_check_inst_target,
                           default="melody",
                           help=f"select target inst: {get_args(InstrumentTarget)}")
    argparser.add_argument("--max_midi_files",  type=int,
                           default=10000, help="n of midi files")
    argparser.add_argument("--batchsize",  type=int,
                           default=32, help="batch size")
    argparser.add_argument("--learning_rate",  type=float,
                           default=1e-4, help="learning rate")
    argparser.add_argument("--epochs", type=int,
                           default=20, help="n of epochs")
    argparser.add_argument("--dropout", type=float,
                           default=0.2, help="dropout rate")
    args = argparser.parse_args()
    main(args)
