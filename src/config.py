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
    def __init__(self,
                 dataset_dir: str,
                 encoding_method: EncodingName,
                 hparams: Namespace,
                 ):
        assert os.path.exists(dataset_dir),\
            f"{dataset_dir} isn't exist"
        self.dataset_dir = dataset_dir
        self.tokenizer_class = encodings[encoding_method]
        self.hparams = hparams


if __name__ == "__main__":
    pass
