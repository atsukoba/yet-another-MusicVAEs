import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor
from glob import glob
from logging import debug
from random import shuffle
from typing import Dict, List, Optional

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
                        target_track: InstrumentTarget,
                        max_seq_length: Optional[int]) -> Optional[List[List[int]]]:
    try:
        midi = extract_target_part(MidiFile(path), target_track)
        if midi is None:
            debug(f"No {target_track} track found from {path}")
            return
        tokens: List[List[int]] = []
        midi_samples: List[MidiFile] = []
        for m in midi:
            if _valid_midi_file(m):
                midi_samples += get_multitrack_n_bars(tokenizer,
                                                      m,
                                                      n_bars=n_bars,
                                                      n_bars_stride=n_bars//2,
                                                      n_instruments_threshold=0,
                                                      n_notes_threshold=4)
        for sample in midi_samples:
            token = tokenizer.midi_to_tokens(sample)
            if type(token[0]) == list:
                token = token[0]  # type: ignore
            if max_seq_length is None or len(token) < max_seq_length:
                # ignore too long samples
                tokens.append(_simple_padding(
                    token, max_seq_length))  # type: ignore
        return tokens
    except Exception as e:
        # print(e)
        return


def _simple_padding(seq: List[int], length: int, pad_index=0) -> List[int]:
    return seq + [pad_index] * (length - len(seq))


class MidiDataset(Dataset):
    def __init__(self,
                 midi_pathes: List[str],
                 tokenizer: MIDITokenizer,
                 bar_length: int = 16,
                 target_track: InstrumentTarget = "melody",
                 max_seq_length: Optional[int] = None):
        self.midi_pathes = midi_pathes
        self.tokenizer = tokenizer
        self.bar_length = bar_length
        self.target_track = target_track

        assert len(tokenizer.vocab.tokens_of_type("Bar")) > 0,\
            "input tokenizer doesn't have Bar tokens"

        self.all_tokens: List[List[int]] = []
        all_lengthes: List[int] = []  # for padding
        # notes: avoid using torch.tensor processing in multi processing
        num_cores = mp.cpu_count()
        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            with tqdm(total=len(self.midi_pathes), desc="Loading MIDI files") as progress:
                futures: List[Future] = []
                for path in self.midi_pathes:
                    future = pool.submit(_tokenize_midi_file,
                                         tokenizer,
                                         path,
                                         self.bar_length,
                                         target_track,
                                         max_seq_length)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)
                for future in futures:
                    result: Optional[List[List[int]]] = future.result()
                    if result is not None:
                        self.all_tokens += result
                        all_lengthes.append(len(result))
        self.max_seq_len = max(all_lengthes)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor(
            _simple_padding(self.all_tokens[idx], self.max_seq_len))}

    def __len__(self):
        return len(self.all_tokens)


class MidiDataModule(LightningDataModule):
    def __init__(self,
                 tokenizer: MIDITokenizer,
                 conf: MusicVAEConfig,
                 max_seq_length: Optional[int] = 1024,
                 shuffle_data_pathes: bool = False,
                 max_n_midi_files: Optional[int] = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.conf = conf
        self.midi_pathes: List[str] = []
        self.batch_size = conf.batchsize
        self.shuffle_data_pathes = shuffle_data_pathes
        self.max_n_midi_files = max_n_midi_files or conf.hparams.max_midi_files
        self.max_seq_length = max_seq_length

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
        self.dataset = MidiDataset(self.midi_pathes,
                                   self.tokenizer,
                                   bar_length=self.conf.hparams.n_bars,
                                   max_seq_length=self.max_seq_length)
        n_train = int(len(self.dataset) * 0.7)
        n_val = int(len(self.dataset) * 0.2)
        n_test = len(self.dataset) - (n_train + n_val)
        self.dataset_train, self.dataset_val, self.dataset_test = random_split(
            self.dataset, [n_train, n_val, n_test])

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.batch_size,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          pin_memory=True)

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        return


if __name__ == "__main__":
    pass
