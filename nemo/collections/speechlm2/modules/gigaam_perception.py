# SPDX-License-Identifier: Apache-2.0
"""GigaAM's official batched forward adapted to the DuplexSTT perception interface."""

import torch
from torch import nn


class GigaAMPerception(nn.Module):
    """Offline perception: GigaAM is bidirectional, not a streaming encoder."""

    def __init__(self, checkpoint: str, output_dim: int, frame_length: float = 0.08):
        super().__init__()
        import gigaam

        self.model = gigaam.load_model(
            checkpoint, device=torch.device("cuda", torch.cuda.current_device()), fp16_encoder=True, use_flash=True
        )
        del self.model.head
        actual_frame = self.model.preprocessor.hop_length * self.model.cfg.encoder.subsampling_factor / 16000
        self.subsampling_factor = round(frame_length / actual_frame)
        if self.subsampling_factor < 1 or abs(actual_frame * self.subsampling_factor - frame_length) > 1e-6:
            raise ValueError(f"Cannot map GigaAM frame duration {actual_frame} to dataset {frame_length}")
        self.pool = nn.AvgPool1d(self.subsampling_factor, ceil_mode=True)
        self.proj = nn.Linear(self.model.cfg.encoder.d_model, output_dim)

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def preprocessor(self):
        return self.model.preprocessor

    def forward(self, input_signal, input_signal_length, return_encoder_emb=False):
        # Spectrograms stay FP32 on GPU; GigaAM.forward uses its native encoder autocast.
        self.preprocessor.float()
        encoded, lengths = self.model(input_signal.float(), input_signal_length)
        encoded = self.pool(encoded)
        lengths = (lengths + self.subsampling_factor - 1) // self.subsampling_factor
        projected = self.proj(encoded.transpose(1, 2).to(self.proj.weight.dtype))
        valid = torch.arange(projected.size(1), device=projected.device)[None] < lengths[:, None]
        projected = projected.masked_fill(~valid[..., None], 0)
        result = (projected, lengths)
        return (*result, encoded) if return_encoder_emb else result
