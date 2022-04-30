from argparse import Namespace
from turtle import forward
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from miditok import MIDITokenizer
from pytorch_lightning import LightningModule
from torch import Tensor as T
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.nn.functional import binary_cross_entropy

from src.config import MusicVAEConfig


class LSTMEncoder(nn.Module):
    def __init__(self, tokenizer: MIDITokenizer,
                 latent_size: int = 512,
                 hidden_size: int = 2048,
                 embedding_size: int = 128):
        super(LSTMEncoder, self).__init__()
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
        super(HierarchicalLSTMDecoder, self).__init__()
        self.fc = nn.Linear(latent_size, output_size)
        # First-level LSTM decoder
        self.conductor = nn.LSTM(input_size=output_size,
                                 hidden_size=hidden_size,
                                 num_layers=2,
                                 batch_first=True)
        # Sub-sequence LSTM decoder
        self.decoder = nn.LSTM(input_size=output_size,
                               hidden_size=hidden_size,
                               batch_first=True)
        self.output = nn.Linear(output_size, len(tokenizer.vocab))

        self.output_size = output_size
        self.conductor_max_len = conductor_max_len
        self.decoder_max_len = decoder_max_len

    def forward(self, z: T) -> T:
        # TODO atsuya: implement hierarchical decoding
        output: T = torch.tensor([[]])  # [batch, segment_seq]
        # conductor_out: [1, output_size]
        _, (conductor_out, _) = self.conductor(F.tanh(self.fc(z)))
        for segment_idx in range(self.conductor_max_len):
            self.decoder.hidden_state = conductor_out
            _input = torch.zeros(
                z.size(0), self.decoder_max_len, self.output_size)
            decoder_out, (_, _) = self.decoder(_input)  # [batch, output_dim]
            results: T = torch.tensor([[]])
            for token_idx in range(decoder_out.size(1)):
                result = self.sample(decoder_out[:, token_idx, :])  # [batch]]
                output = torch.cat([output, result], 1)  # [batch, seq_len]
        return output

    def sample(self, logit: T, _type="argmax") -> T:
        # TODO atsuya: implement sampling methods (temperature, top_k, top_p)
        results = torch.argmax(self.output(logit), dim=1)
        return results


class LtMusicVAE(LightningModule):
    def __init__(self, tokenizer: MIDITokenizer, config: MusicVAEConfig):
        super(LtMusicVAE, self).__init__()
        self.learning_rate = config.hparams.learning_rate
        self.encoder = LSTMEncoder(tokenizer)
        self.decoder = HierarchicalLSTMDecoder(tokenizer, )

    def forward(self, x: T) -> Tuple[T, T, T, T]:
        mu, sigma = self.encoder(x)
        z = self.reparametrize(mu, sigma)
        x = self.decoder(z)
        return (x, z, mu, sigma)

    def elbo(self, pred, target, mu, sigma, free_bits=256) -> Tuple[T, T]:
        """
        Evidence Lower Bound
        Return KL Divergence and KL Regularization using free bits
        """
        # Reconstruction error
        # Pytorch cross_entropy combines LogSoftmax and NLLLoss
        likelihood = -binary_cross_entropy(pred, target, reduction='sum')
        # Regularization error
        sigma_prior = torch.tensor([1], dtype=torch.float)
        mu_prior = torch.tensor([0], dtype=torch.float)
        p = Normal(mu_prior, sigma_prior)
        q = Normal(mu, sigma)
        kl_div = kl_divergence(q, p)
        elbo = torch.mean(likelihood) - torch.max(
            torch.mean(kl_div) - free_bits, torch.tensor([0], dtype=torch.float))
        return -elbo, kl_div.mean()

    def reparametrize(self, mu: T, sigma: T) -> T:
        eps = torch.randn(mu.shape)
        return mu + eps * torch.exp(0.5 * sigma)

    def training_step(self, batch: T, batch_idx: int) -> T:
        x, z, mu, sigma = self(batch)
        elbo_loss, kl_loss = self.elbo(x, batch, mu, sigma)
        self.log("train_loss", elbo_loss, prog_bar=True)
        return elbo_loss

    def validation_step(self, batch: T, batch_idx: int) -> T:
        x, z, mu, sigma = self(batch)
        elbo_loss, kl_loss = self.elbo(x, batch, mu, sigma)
        self.log("val_loss", elbo_loss, prog_bar=True)
        return elbo_loss

    def test_step(self, batch: T, batch_idx: int) -> T:
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
