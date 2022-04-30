import os
from argparse import Namespace
from collections import namedtuple
from typing import Dict, Literal, Type, Union

from miditok import REMI, CPWord, MIDILike, MIDITokenizer, MuMIDI, OctupleMono

encodings: Dict[str, Union[
    Type[REMI], Type[CPWord],
    Type[MIDILike], Type[OctupleMono],
    Type[MuMIDI]]] = {
    # original midi-like and encodings that has Bar events
    "remi": REMI,
    "cpword": CPWord,
    "midi-like": MIDILike,
    "octuple": OctupleMono,
    "mumidi": MuMIDI
}

EncodingName = Literal["remi", "cpword", "midi-like", "octuple-mono", "mumidi"]


class MusicVAEConfig:
    def __init__(self, hparams: Namespace):
        assert os.path.isdir(
            hparams.dataset_dir), f"{hparams.dataset_dir} isn't exist"
        self.dataset_dir = hparams.dataset_dir
        self.tokenizer_type = encodings[hparams.encoding_method]
        self.batchsize = hparams.batchsize
        self.hparams = hparams


if __name__ == "__main__":
    pass
