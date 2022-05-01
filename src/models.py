from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from miditok import MIDITokenizer
from pytorch_lightning import LightningModule
from torch import Tensor as T
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.nn.functional import nll_loss

from src.config import MusicVAEConfig


class LSTMEncoder(LightningModule):
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
        self.liner_mu = nn.Linear(hidden_size * 2, latent_size, bias=True)
        self.liner_sigma = nn.Linear(hidden_size * 2, latent_size, bias=True)

    def forward(self, x: T) -> Tuple[T, T]:
        x = self.encoder(x)
        _, (_, h) = self.lstm(x)
        concat_hidden = torch.cat([h[0, :, :], h[1, :, :]], dim=1)
        mu = self.liner_mu(concat_hidden)
        sigma = torch.log(self.liner_sigma(concat_hidden) + 1.0)
        return (mu, sigma)


class HierarchicalLSTMDecoder(LightningModule):
    def __init__(self, tokenizer: MIDITokenizer,
                 latent_size: int = 512,
                 hidden_size: int = 1024,
                 output_size: int = 512,
                 conductor_max_len: int = 16,  # n of bars
                 decoder_max_len: int = 64):  # n of tokens for each bar
        super(HierarchicalLSTMDecoder, self).__init__()
        # z to conductor
        self.fc = nn.Linear(latent_size, hidden_size)
        # conductor (2-layer) out to decoder (1-layer) state
        self.fc_2 = nn.Linear(hidden_size, output_size // 2)
        # First-level LSTM decoder
        self.conductor = nn.LSTM(input_size=output_size,
                                 hidden_size=hidden_size,
                                 num_layers=2,
                                 batch_first=True)
        # Sub-sequence LSTM decoder
        self.decoder = nn.LSTM(input_size=output_size,
                               hidden_size=hidden_size,
                               num_layers=2,
                               batch_first=True)
        self.output = nn.Linear(hidden_size, len(tokenizer.vocab))

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.conductor_max_len = conductor_max_len
        self.decoder_max_len = decoder_max_len

    def forward(self, z: T) -> T:
        # TODO atsuya: implement hierarchical decoding
        batch_size = z.size(0)

        # conductor_out: [1, output_size]
        _cond_input = torch.zeros(
            batch_size, self.conductor_max_len, self.output_size).to(self.device)
        _cond_init = (torch.rand(2, batch_size, self.hidden_size).to(self.device),
                      torch.stack([torch.tanh(self.fc(z)),
                                   torch.tanh(self.fc(z))]).to(self.device))  # 2 layers LSTM
        conductor_out, (_, _) = self.conductor(_cond_input, _cond_init)

        output: List[T] = []  # [batch, segment_seq]
        last_decoder_c: T = torch.rand(
            2, batch_size, self.hidden_size).to(self.device)
        for segment_idx in range(self.conductor_max_len):
            results: T = torch.tensor([[]])
            for token_idx in range(self.decoder_max_len):
                # concat last hidden and concuctor out
                _input = torch.cat(
                    [torch.tanh(self.fc_2(conductor_out[:, segment_idx, :])),
                     torch.tanh(self.fc_2(last_decoder_c.mean(dim=0)))], dim=1
                ).view(batch_size, 1, self.output_size)
                decoder_out, (decoder_c, _) = self.decoder(_input)
                last_decoder_c = decoder_c
                pred_out = self.output(decoder_out[:, 0, :])
                result = F.softmax(pred_out, 1).view(batch_size,
                                                     1,
                                                     pred_out.size(1))
                output.append(result)
        return torch.cat(output, dim=1).to(self.device)

    def sample(self, logit: T, _type="argmax") -> T:
        # TODO atsuya: implement sampling methods (temperature, top_k, top_p)
        results = torch.argmax(self.output(logit), dim=1)
        return results


class LtMusicVAE(LightningModule):
    def __init__(self, tokenizer: MIDITokenizer, config: MusicVAEConfig):
        super(LtMusicVAE, self).__init__()
        self.learning_rate = config.hparams.learning_rate
        self.encoder = LSTMEncoder(tokenizer,
                                   latent_size=config.hparams.latent_space_size,
                                   hidden_size=config.hparams.encoder_hidden_size)
        self.decoder = HierarchicalLSTMDecoder(tokenizer,
                                               output_size=config.hparams.decoder_feed_forward_size,
                                               hidden_size=config.hparams.decoder_hidden_size,
                                               latent_size=config.hparams.latent_space_size,
                                               conductor_max_len=config.hparams.n_bars)

    def forward(self, x: T) -> Tuple[T, T, T, T]:
        mu, sigma = self.encoder(x)
        z = self.reparametrize(mu, sigma)
        x = self.decoder(z)
        return (x, z, mu, sigma)

    def elbo(self, pred: T, target: T, mu: T, sigma: T, free_bits=256) -> Tuple[T, T]:
        """
        Evidence Lower Bound
        Return KL Divergence and KL Regularization using free bits
        """
        # Reconstruction error
        # Pytorch cross_entropy combines LogSoftmax and NLLLoss
        # [batch, seq_len, n_vocab] -> [batch, n_vocab, seq_len]
        likelihood = nll_loss(pred.transpose(1, 2), target, ignore_index=0)
        # Regularization error
        # TODO: handle negative sigma
        try:
            q_z = Normal(mu, torch.exp(sigma))
        except ValueError as e:
            print("mu: ", mu)
            print("sigma: ", sigma)
            raise e
        sigma_prior = torch.tensor([1], dtype=torch.float).to(self.device)
        mu_prior = torch.tensor([0], dtype=torch.float).to(self.device)
        p_z = Normal(mu_prior, sigma_prior)
        kl_div = kl_divergence(q_z, p_z)
        elbo = torch.mean(likelihood) - torch.max(
            torch.mean(kl_div) - free_bits, torch.tensor([0], dtype=torch.float).to(self.device))
        return -elbo, kl_div.mean()

    def reparametrize(self, mu: T, sigma: T) -> T:
        eps = torch.randn(mu.shape).to(self.device)
        return mu + eps * torch.exp(0.5 * sigma)

    def training_step(self, batch: Dict[str, T], batch_idx: int) -> T:
        x, z, mu, sigma = self(batch["input_ids"])
        elbo_loss, kl_loss = self.elbo(x, batch["input_ids"], mu, sigma)
        self.log("train_loss", elbo_loss, prog_bar=True)
        return elbo_loss

    def validation_step(self, batch: Dict[str, T], batch_idx: int) -> T:
        x, z, mu, sigma = self(batch["input_ids"])
        elbo_loss, kl_loss = self.elbo(x, batch["input_ids"], mu, sigma)
        self.log("val_loss", elbo_loss, prog_bar=True)
        return elbo_loss

    def test_step(self, batch: Dict[str, T], batch_idx: int) -> T:
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
