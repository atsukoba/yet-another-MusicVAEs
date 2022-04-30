# Yet Another MusicVAEs: pytorch-lightning implementation with several data representations

MusicVAE Paper: [Roberts, A., Engel, J., Raffel, C., Hawthorne, C., & Eck, D. (2018, July). A hierarchical latent vector model for learning long-term structure in music. In International conference on machine learning (pp. 4364-4373). PMLR.](https://arxiv.org/abs/1803.05428)

### What's New?

- implemented on Pytorch Lightning
- Several MIDI data representations provided on [MidiTok](https://github.com/Natooz/MidiTok)
    - MIDI-Like Representations
    - REMI
    - MuMIDI
    - Octuple MIDI (single track)
    - Compound Word

<img src="https://magenta.tensorflow.org/assets/music_vae/architecture.png" width=500 style="display:block;margin:0 auto;">


## environments

tested on

- Python=3.10.4
- nvidia-driver=460 (NVIDIA GeForce RTX 3060)
- CUDA=11.4

## setup

1. get your datasets such as [LMD (Lakh MIDI Dataset)](https://colinraffel.com/projects/lmd/#get) or [MMD (Meta MIDI Dataset)](https://colinraffel.com/projects/lmd/#get)
1. install dependencies

```shell
pip install -r requrements.txt
```

## Training

```shell
usage: train.py [-h] [-e ENCODING_METHOD] [-d DATASET_DIR] [-t TARGET_INSTRUMENT] [--n_bars N_BARS] [--max_midi_files MAX_MIDI_FILES] [--batchsize BATCHSIZE] [--learning_rate LEARNING_RATE]
                [--epochs EPOCHS] [--dropout DROPOUT] [--latent_space_size LATENT_SPACE_SIZE] [--encoder_hidden_size ENCODER_HIDDEN_SIZE] [--decoder_hidden_size DECODER_HIDDEN_SIZE]
                [--decoder_feed_forward_size DECODER_FEED_FORWARD_SIZE]

Basic settings and hyper-parameters for training the model

optional arguments:
  -h, --help            show this help message and exit
  -e ENCODING_METHOD, --encoding_method ENCODING_METHOD
                        encoding methods: ('remi', 'cpword', 'midi-like', 'octuple-mono', 'mumidi')
  -d DATASET_DIR, --dataset_dir DATASET_DIR
                        data directory for training
  -t TARGET_INSTRUMENT, --target_instrument TARGET_INSTRUMENT
                        select target inst: ('melody', 'bass')
  --n_bars N_BARS       n of bars
  --max_midi_files MAX_MIDI_FILES
                        n of midi files
  --batchsize BATCHSIZE
                        batch size
  --learning_rate LEARNING_RATE
                        learning rate
  --epochs EPOCHS       n of epochs
  --dropout DROPOUT     dropout rate
  --latent_space_size LATENT_SPACE_SIZE
                        decoder hidden dimention
  --encoder_hidden_size ENCODER_HIDDEN_SIZE
                        encoder hidden dimention
  --decoder_hidden_size DECODER_HIDDEN_SIZE
                        decoder hidden dimention
  --decoder_feed_forward_size DECODER_FEED_FORWARD_SIZE
```

for example

```shell
python train.py -e remi -d /datasets/meta-midi-dataset/MMD_MIDI --n_bars 16  --max_midi_files 5000 --batchsize 32 --latent_space_size 100 --encoder_hidden_size 256 --decoder_hidden_size 256 --decoder_feed_forward_size 128
```

## Evaluation

comming soon

## Pre-trained Models

comming soon
