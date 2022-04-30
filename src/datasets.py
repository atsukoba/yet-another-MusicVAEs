import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor
from glob import glob
from random import shuffle
from typing import Dict, List, Optional, Tuple, Type
from numpy import rec
from sklearn.semi_supervised import SelfTrainingClassifier

import torch
from miditok import MIDITokenizer
from miditoolkit import MidiFile
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from src.config import MusicVAEConfig
from src.utils import (InstrumentTarget, extract_target_part,
                       get_multitrack_n_bars)


def _valid_midi_file(midi: MidiFile) -> bool:
    # TODO: set the definition of valid midi object
    return True


def _tokenize_midi_file(tokenizer: MIDITokenizer,
                        path: str,
                        n_bars: int,
                        target_track: InstrumentTarget) -> Optional[List[List[int]]]:
    try:
        midi = extract_target_part(MidiFile(path), target_track)
        if midi is None:
            return
        tokens: List[List[int]] = []
        midi_samples: List[MidiFile] = []
        for m in midi:
            if _valid_midi_file(m):
                midi_samples += get_multitrack_n_bars(tokenizer,
                                                      m,
                                                      n_bars=n_bars,
                                                      n_bars_stride=n_bars//2,
                                                      n_notes_threshold=4)
        for sample in midi_samples:
            tokens.append(tokenizer.midi_to_tokens(sample))  # type: ignore
        return tokens
    except Exception as e:
        # print(e)
        return


class MidiDataset(Dataset):
    def __init__(self,
                 midi_pathes: List[str],
                 tokenizer: MIDITokenizer,
                 bar_length: int = 16,
                 target_track: InstrumentTarget = "melody"):
        self.midi_pathes = midi_pathes
        self.tokenizer = tokenizer
        self.bar_length = bar_length
        self.target_track = target_track

        assert len(tokenizer.vocab.tokens_of_type("Bar")) > 0,\
            "input tokenizer doesn't have Bar tokens"

        self.all_tokens: List[List[int]] = []
        num_cores = mp.cpu_count()
        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            with tqdm(total=len(self.midi_pathes)) as progress:
                futures: List[Future] = []
                for path in self.midi_pathes:
                    future = pool.submit(_tokenize_midi_file,
                                         tokenizer,
                                         path,
                                         self.bar_length,
                                         target_track)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)
                for future in futures:
                    result: Optional[List[List[int]]] = future.result()
                    if result is not None:
                        self.all_tokens += result

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor(self.all_tokens[idx])}

    def __len__(self):
        return len(self.all_tokens)


class MidiDataModule(LightningDataModule):
    def __init__(self,
                 tokenizer: MIDITokenizer,
                 conf: MusicVAEConfig,
                 batch_size: int = 32,
                 shuffle_data_pathes: bool = False,
                 max_n_midi_files: Optional[int] = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.conf = conf
        self.midi_pathes: List[str] = []
        self.batch_size = batch_size
        self.shuffle_data_pathes = shuffle_data_pathes
        self.max_n_midi_files = max_n_midi_files or conf.hparams.max_midi_files

    def prepare_data(self) -> None:
        # load data
        self.midi_pathes = glob(os.path.join(
            self.conf.dataset_dir, "**", "*.mid"), recursive=True)
        if self.shuffle_data_pathes:
            shuffle(self.midi_pathes)
        if self.max_n_midi_files:
            self.midi_pathes = self.midi_pathes[:self.max_n_midi_files]
        assert len(self.midi_pathes) != 0, \
            f"No MIDI file found in {self.conf.dataset_dir}"

    def setup(self, stage: Optional[str] = None):
        self.dataset = MidiDataset(self.midi_pathes, self.tokenizer)
        n_train = int(len(self.dataset) * 0.7)
        n_val = int(len(self.dataset) * 0.2)
        n_test = len(self.dataset) - (n_train + n_val)
        self.dataset_train, self.dataset_val, self.dataset_test = random_split(
            self.dataset, [n_train, n_val, n_test])

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        return


if __name__ == "__main__":
    pass
