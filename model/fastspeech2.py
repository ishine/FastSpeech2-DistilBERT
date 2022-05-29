import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths, pad_2D


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        # maps DistilBERT hidden sizes to
        # encoder hidden sizes
        self.dbert_linear_hidden = nn.Linear(
            model_config["transformer"]["dbert_hidden"],
            model_config["transformer"]["decoder_hidden"],
        )
        # TODO(danj): this maps DistilBERT sequences
        # to encoder sequences
        self.dbert_linear_seq = nn.Linear(
            model_config["max_seq_len"],
            model_config["max_seq_len"],
        )
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config,
                                                model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"],
                        "speakers.json"),
                    "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        dbert_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (get_mask_from_lengths(mel_lens, max_mel_len)
                     if mel_lens is not None else None)

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        # TODO(danj): improve this
        # dbert hidden size => encoder hidden size
        dbert = self.dbert_linear_hidden(dbert_targets)
        # dbert sequence => phoneme embedded sequence
        dbert = self.dbert_linear_seq(dbert.permute(0, 2, 1))
        # adding dbert embedding to output, truncating extra
        # this shouldn't be a problem because almost all weights
        # associate left to right, since dbert sequences are
        # almost always shorter than phoneme sequences
        output += dbert.permute(0, 2, 1)[:, :output.shape[1], :]
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
