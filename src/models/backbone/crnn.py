import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, input_channels=1, freq_dim=216, num_channels=32, num_rnn_layers=2, dropout=0.2):
        super().__init__()

        # First convolution: single 5x5 filter
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=(5, 5),
            padding="same"
        )

        # Second convolution: 36 full-height filters
        self.conv2 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_channels,
            kernel_size=(freq_dim, 1),
            groups=input_channels
        )

        self.ln = nn.LayerNorm(num_channels)

        # Bidirectional GRU
        self.gru1 = nn.GRU(
            input_size=num_channels,
            hidden_size=num_channels * 4,
            bidirectional=True,
            batch_first=True
        )

        self.gru2 = nn.GRU(
            input_size=num_channels * 8,
            hidden_size=num_channels * 4,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
            num_layers=num_rnn_layers - 1
        )
       
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_channels * 8     

    def forward(self, x: torch.Tensor):
        # Apply first convolution and activation
        x = F.relu(self.conv1(x))

        # Apply second convolution and activation
        x = F.relu(self.conv2(x))
        x = x.squeeze(2)  # Remove frequency dimension
        x = x.permute(0, 2, 1)  # N, C, T -> N, T, C

        x = self.ln(x)  # Apply LayerNorm if specified
        x = self.dropout(x)

        # Process with Bi-GRU
        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, _ = self.gru2(x)
        x = self.dropout(x)
        return x
