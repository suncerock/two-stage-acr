import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2ConformerConfig
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerEncoder

class Conformer(nn.Module):
    def __init__(
        self,
        chunk_n_frames=100,

        freq_dim=216,

        hidden_size=192,
        num_hidden_layers=4,
        num_attention_heads=6,
        intermediate_size=768,

        *args,
        **kwargs
    ):
        super().__init__()

        self.chunk_n_frames = chunk_n_frames
        config = Wav2Vec2ConformerConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_source_positions=chunk_n_frames,
            *args,
            **kwargs
        )

        self.input_proj = nn.Linear(freq_dim, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(config.feat_proj_dropout)

        # Initialize the encoder only
        self.encoder = Wav2Vec2ConformerEncoder(config)

        self.output_dropout = nn.Dropout(config.final_dropout)
        self.output_dim = config.hidden_size

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, 1, freq_dim, sequence_length)
        hidden_states = x.squeeze(1).permute(0, 2, 1)  # (batch_size, sequence_length, freq_dim)
        hidden_states = self.input_dropout(hidden_states)
        if hidden_states.shape[1] == self.chunk_n_frames:
            output = self.forward_segment(hidden_states)
        else:
            output = self.forward_moving_window(hidden_states)
        return output

    def forward_segment(self, hidden_states: torch.Tensor):
        hidden_states = self.input_proj(hidden_states)  # (batch_size, sequence_length, hidden_size)
        hidden_states = self.input_ln(hidden_states)

        # Use it with pre-computed hidden states
        # hidden_states shape: (batch_size, sequence_length, hidden_size)
        encoder_outputs = self.encoder(hidden_states)
        output = self.output_dropout(encoder_outputs.last_hidden_state)  # (batch_size, sequence_length, hidden_size)
        return output  # (batch_size, sequence_length, hidden_size)

    def forward_moving_window(self, hidden_states: torch.Tensor):
        if hidden_states.shape[1] < self.chunk_n_frames:
            return self.forward_segment(hidden_states)

        num_instances = (hidden_states.shape[1] - self.chunk_n_frames) // (self.chunk_n_frames // 2) + 1
        if hidden_states.shape[1] % (self.chunk_n_frames // 2) != 0:
            num_instances += 1

        all_outputs = [self.forward_segment(hidden_states[:, self.chunk_n_frames // 2 * i: self.chunk_n_frames // 2 * i + self.chunk_n_frames]) for i in range(num_instances)]
        
        output = torch.concat([all_outputs[0][:, :(self.chunk_n_frames // 4 * 3)] ]+ [output[:, (self.chunk_n_frames // 4) : (self.chunk_n_frames // 4 * 3)] for output in all_outputs[1:-1]] + [all_outputs[-1][:, (self.chunk_n_frames // 4):]], dim=1)
       
        return output
