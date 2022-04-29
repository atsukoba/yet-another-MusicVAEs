from argparse import Namespace
from turtle import forward
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from miditok import MIDITokenizer
from pytorch_lightning import LightningModule
from torch import Tensor as T
from torch import nn
from src.config import MusicVAEConfig


class LSTMEncoder(nn.Module):
    def __init__(self, tokenizer: MIDITokenizer,
                 latent_size: int = 512,
                 hidden_size: int = 2048,
                 embedding_size: int = 128):
        super().__init__()
        self.encoder = nn.Embedding(num_embeddings=len(tokenizer.vocab),
                                    embedding_dim=embedding_size,
                                    padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first=True)
        self.liner_mu = nn.Linear(hidden_size, latent_size, bias=True)
        self.liner_sigma = nn.Linear(hidden_size, latent_size, bias=True)

    def forward(self, x: T) -> Tuple[T, T]:
        x = self.encoder(x)
        _, (c_n, h_n) = self.lstm(x)
        mu = self.liner_mu(h_n[:, -1, :])
        sigma = torch.log(self.liner_sigma(h_n[:, -1, :]) + 1.0)
        return (mu, sigma)


class HierarchicalLSTMDecoder(nn.Module):
    def __init__(self, tokenizer: MIDITokenizer,
                 latent_size: int = 512,
                 hidden_size: int = 1024,
                 output_size: int = 256,
                 conductor_max_len: int = 16,  # n of bars
                 decoder_max_len: int = 256):  # n of tokens for each bar
        super().__init__()
        self.fc = F.tanh(nn.Linear(latent_size, output_size))
        # First-level LSTM decoder
        self.conductor = nn.LSTM(input_size=output_size,
                                 hidden_size=hidden_size,
                                 num_layers=2,
                                 batch_first=True)
        # Sub-sequence LSTM decoder
        self.decoder = nn.LSTM(input_size=output_size, output_size=output_size)
        self.output = nn.Linear(output_size, len(tokenizer.vocab))

    def forward(self, z: T) -> T:
        # TODO atsuya: implement hierarchical decoding
        output: T = torch.tensor([])  # [batch, seq]
        for sample in range(z.size(0)):  # sample: [1, seq, latent_size]
            chunk: T = torch.tensor([])
            decoder_init = torch.zeros()
            _, (conductor_out, _) = self.conductor(
                self.fc(z))  # conductor_out: [1, output_size]
            self.decoder.hidden_state = conductor_out
            _, (dec_c, dec_d) = self.decoder()
            logit = self.output(dec_c)
            F.softmax(logit)
        return z

    def sample(self, logit: T) -> T:
        # TODO atsuya: implement sampling methods (temperature, top_k, top_p)
        return logit


class LtMusicVAE(LightningModule):
    def __init__(self, tokenizer: MIDITokenizer, hparams: Namespace):
        super().__init__()
        self.learning_rate = hparams.learning_rate
        self.encoder = LSTMEncoder(tokenizer)
        self.decoder = HierarchicalLSTMDecoder(tokenizer, )

    def forward(self, x: T) -> T:
        mu, sigma = self.encoder(x)
        z = self.reparametrize(mu, sigma)
        x = self.decoder(z)
        return x

    def reconstruction_loss(self, x: T) -> T:
        # TODO atsuya: implement
        return x

    def reparametrize(self, mu: T, sigma: T) -> T:
        eps = torch.randn(mu.shape)
        return mu + eps * torch.exp(0.5 * sigma)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
