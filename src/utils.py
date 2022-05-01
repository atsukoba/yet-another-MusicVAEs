import cProfile
from typing import List, Literal, Optional

import _pickle as cPickle
from miditok import MIDITokenizer
from miditok.constants import INSTRUMENT_CLASSES_RANGES, MIDI_INSTRUMENTS
from miditoolkit import Instrument, MidiFile
from miditoolkit.midi.utils import example_midi_file
from note_seq import plot_sequence
from note_seq.protobuf.music_pb2 import NoteSequence

InstrumentTarget = Literal["melody", "bass"]


def deepcopy(obj):
    return cPickle.loads(cPickle.dumps(obj, -1))


def split_instruments(song: MidiFile) -> List[MidiFile]:
    tracks: List[MidiFile] = []
    for inst in song.instruments:
        m = MidiFile()
        m.max_tick = song.max_tick
        m.ticks_per_beat = song.ticks_per_beat
        m.tempo_changes = song.tempo_changes
        m.time_signature_changes = song.time_signature_changes
        m.instruments = [inst]
        tracks.append(m)
    return tracks


def extract_drums(song: MidiFile) -> Optional[List[MidiFile]]:
    extracted_midi: List[MidiFile] = []
    for inst in song.instruments:
        if inst.is_drum:
            m = MidiFile()
            m.ticks_per_beat = song.ticks_per_beat
            m.tempo_changes = song.tempo_changes
            m.time_signature_changes = song.time_signature_changes
            m.instruments = [inst]
            extracted_midi.append(m)
    return extracted_midi


def extract_target_part(song: MidiFile,
                        target: InstrumentTarget) -> Optional[List[MidiFile]]:
    """ NOTES: if extracting drums track, call `extract_drums` instead """
    if target == "melody":
        target_names = ["Piano", "Synth Lead", "Organ"]
    elif target == "bass":
        target_names = ["Bass"]
    else:
        return
    target_programs = []
    for target_name in target_names:
        target_range = INSTRUMENT_CLASSES_RANGES[target_name]
        target_programs += list(range(target_range[0], target_range[1]+1))
    extracted_midi: List[MidiFile] = []

    for inst in song.instruments:
        inst: Instrument = inst
        # filter instrument by program number or name description
        if not inst.is_drum and (inst.program in target_programs or target in inst.name):
            m = MidiFile()
            m.max_tick = song.max_tick
            m.ticks_per_beat = song.ticks_per_beat
            m.tempo_changes = song.tempo_changes
            m.time_signature_changes = song.time_signature_changes
            m.instruments = [inst]
            extracted_midi.append(m)
    return extracted_midi


def get_multitrack_n_bars(tokenizer: MIDITokenizer,
                          midi: MidiFile,
                          n_bars: int = 4,
                          n_bars_stride: Optional[int] = None,
                          n_notes_threshold: Optional[int] = None,
                          n_instruments_threshold: int = 2) -> List[MidiFile]:
    """
    returns list of Midifile samples that separated into
    single track and chunk of bars with given length
    """
    ticks_per_bar = midi.ticks_per_beat * 4
    if n_bars_stride is None:
        n_bars_stride = n_bars

    bar_start_ticks = [start_tick for i in range(midi.max_tick // (ticks_per_bar * n_bars_stride)) if (
        (start_tick := (ticks_per_bar * i * n_bars_stride)) + ticks_per_bar * n_bars) < midi.max_tick]

    if n_notes_threshold is None:
        n_notes_threshold = 1 * n_bars

    tokenizer.preprocess_midi(midi)  # remove note-less track and quantize

    all_instruments_list: List[List[Instrument]] = []
    for idx, start_tick in enumerate(bar_start_ticks):
        end_tick = n_bars * ticks_per_bar + start_tick - 1
        instruments: List[Instrument] = []
        for idx in range(len(midi.instruments)):
            inst = deepcopy(midi.instruments[idx])
            notes = list(filter(
                lambda n: n.start >= start_tick and
                n.start < end_tick, inst.notes))
            for n in notes:  # align start time
                n.start -= start_tick
                n.end -= start_tick
            if len(notes) > n_notes_threshold:
                inst.notes = notes
                instruments.append(inst)
        if len(instruments) > n_instruments_threshold:
            all_instruments_list.append(instruments)

    midi_samples: List[MidiFile] = []
    for insts in all_instruments_list:
        m = MidiFile(ticks_per_beat=midi.ticks_per_beat)
        m.max_tick = ticks_per_bar * n_bars
        m.instruments = insts
        # inherit meta info
        m.tempo_changes = midi.tempo_changes
        m.key_signature_changes = midi.key_signature_changes
        m.time_signature_changes = midi.time_signature_changes
        midi_samples.append(m)
    return midi_samples


def miditoolkit_to_notesequence(midi: MidiFile) -> NoteSequence:
    ticks_per_sec = midi.ticks_per_beat * midi.tempo_changes[0].tempo / 60
    def ticks_to_sec(ticks): return ticks * (1 / ticks_per_sec)
    ns = NoteSequence()
    ns.tempos.add(qpm=midi.tempo_changes[0].tempo)  # type: ignore
    for inst in midi.instruments:
        for note in inst.notes:
            ns.notes.add(pitch=note.pitch,   # type: ignore
                         start_time=ticks_to_sec(note.start),
                         end_time=ticks_to_sec(note.end),
                         velocity=note.velocity,
                         is_drum=inst.is_drum,
                         # https://ja.wikipedia.org/wiki/General_MIDI
                         instrument=inst.program if inst.is_drum else 0)

    ns.total_time = ticks_to_sec(midi.max_tick)  # type: ignore
    return ns


def check_conversion_with_plots(path: str):
    path_midi = example_midi_file()
    midi = MidiFile(path_midi)
    ns = miditoolkit_to_notesequence(midi)
    plot_sequence(ns)
    plot_sequence(miditoolkit_to_notesequence(midi))
