# Yet Another MusicVAE: pytorch-lightning implementation with several data representations on MidiTok

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
python train.py --data /PATH/TO/YOUR/DATASETs/ --tokenizer
```

## Evaluation
