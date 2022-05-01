from itertools import groupby
from typing import List, Optional, Type

from miditok import (REMI, CPWord, Event, MIDILike, MIDITokenizer, MuMIDI,
                     OctupleMono)


def _split_to_bars_remi(tokens: List[int],
                        tokenizer: REMI) -> Optional[List[List[int]]]:
    bar_sep = tokenizer.vocab.event_to_token["Bar_None"]
    if type(bar_sep) != int:
        return
    chunks = [list(map(int, chunk)) for k, chunk in
              groupby(tokens, lambda x: x == bar_sep) if not k]
    return list(filter(lambda c: len(c) > 1, chunks))


def _split_to_bars_cpword(tokens: List[int],
                          tokenizer: CPWord) -> Optional[List[List[int]]]:
    bar_sep = tokenizer.vocab.event_to_token["Bar_None"]
    if type(bar_sep) != int:
        return
    chunks = [list(map(int, chunk)) for k, chunk in
              groupby(tokens, lambda x: x == bar_sep) if not k]
    return list(filter(lambda c: len(c) > 1, chunks))


def _split_to_bars_midilike(tokens: List[int],
                            tokenizer: MIDILike) -> Optional[List[List[int]]]:

    pseudo_ticks_per_beat = 960
    bar_ticks = pseudo_ticks_per_beat * 4
    current_bars = 1
    rest_indices = tokenizer.vocab.tokens_of_type("Rest")
    time_shift_indices = tokenizer.vocab.tokens_of_type("Time-Shift")
    chunks: List[List[int]] = []
    chunk: List[int] = []
    for t in tokens:
        chunk.append(t)
        if t in rest_indices or t in time_shift_indices:
            e: Event = tokenizer.vocab.token_to_event[t]
            val = e.value
            if tokenizer._token_duration_to_ticks(
                    val, pseudo_ticks_per_beat) >= bar_ticks * current_bars:
                chunks.append(chunk.copy())
                chunk = []
                current_bars += 1
    return chunks


def _split_to_bars_mumidi(tokens: List[int],
                          tokenizer: MuMIDI) -> Optional[List[List[int]]]:
    bar_tokens = tokenizer.vocab.tokens_of_type("Bar")
    chunks: List[List[int]] = []
    chunk: List[int] = []
    current_bar: int = 1
    for t in tokens:
        if t in bar_tokens:
            if tokenizer.vocab.token_to_event[t].value > current_bar:
                chunks.append(chunk)
                chunk = []
        chunk.append(t)

    return list(filter(lambda c: len(c) > 1, chunks))


def _split_to_bars_octuple(tokens: List[int],
                           tokenizer: OctupleMono) -> Optional[List[List[int]]]:
    raise NotImplementedError


def split_to_bars(tokens: List[int],
                  tokenizer: MIDITokenizer) -> Optional[List[List[int]]]:
    if type(tokenizer) is REMI:
        return _split_to_bars_remi(tokens, tokenizer)
    if type(tokenizer) is CPWord:
        return _split_to_bars_cpword(tokens, tokenizer)
    if type(tokenizer) is MIDILike:
        return _split_to_bars_midilike(tokens, tokenizer)
    if type(tokenizer) is MuMIDI:
        return _split_to_bars_mumidi(tokens, tokenizer)
    if type(tokenizer) is OctupleMono:
        return _split_to_bars_octuple(tokens, tokenizer)
