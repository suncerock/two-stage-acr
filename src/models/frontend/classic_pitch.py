import torch
import torch.nn as nn

class LayerNormCF(nn.Module):
    """
    LayerNorm applied across Channel and Frequency dimensions (C, F),
    independently for each time step.
    
    Input:  (B, C, F, T)
    Output: (B, C, F, T)
    """
    def __init__(self, num_channels, num_freq_bins, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, num_freq_bins, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, num_freq_bins, 1))
        self.eps = eps

    def forward(self, x):
        var, mean = torch.var_mean( x, dim=(1, 2), keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight + self.bias
    
class LayerNormC(nn.Module):
    """
    LayerNorm applied across Channel dimension (C),
    independently for each time step and frequency bin.
    
    Input:  (B, C, F, T)
    Output: (B, C, F, T)
    """
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        var, mean = torch.var_mean(x, dim=1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight + self.bias

class StochasticDepth(nn.Module):
    """Drop paths (stochastic depth) per sample (when applied in the main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # Sample binary mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x / keep_prob * binary_mask

class ConvNeXtBlockMPE(nn.Module):
    def __init__(self, in_channels, in_bins, out_channels, kernel_size=11, drop_path=0.0, focus_dim=None, secondary_dim=5, norm="CF"):
        super().__init__()
        if focus_dim is None:
            padding = kernel_size // 2
        elif focus_dim == "frequency":
            padding = (kernel_size // 2, secondary_dim // 2)
            kernel_size = (kernel_size, secondary_dim)
        elif focus_dim == "time":
                padding = (secondary_dim // 2, kernel_size // 2)
                kernel_size = (secondary_dim, kernel_size)
        else:
            raise ValueError(f"Unknown focus_dim: {focus_dim}")

        # 1. Depthwise Conv
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                padding=padding, groups=in_channels)
        
        # 2. LayerNormCF
        self.norm = LayerNormCF(in_channels, in_bins) if norm == "CF" else LayerNormC(in_channels)
        
        # 3. Pointwise Convs using 1x1 convolutions
        self.pwconv1 = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(2 * in_channels, out_channels, kernel_size=1)
        

        self.drop_path = nn.Identity() if drop_path == 0 else StochasticDepth(drop_path)
    
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        return self.drop_path(x) + residual

class HCQTEncoder(nn.Module):
    """
    Input shape: (batch_size, harmonics, freq, time)
    Args:
        freq_bins_in:     Number of input frequency bins (e.g. 216 for 6 octaves with 12 bins each)
        in_channels:     Number of input channels (harmonics in HCQT)
        out_channels:    Number of output channels (1 for pitch, 12 for pitch class)
        widths:          List of channel widths for the convolutional layers
        depths:          List of number of layers for the two stages of the model
    """
    def __init__(self, in_bins=216, in_channels=6, out_channels=1, widths=[20, 40], depths=[6, 10], kernel_sizes=[9, 15], drop_path_rate=0.1, focus_dim=None, kernel_secondary_dim=5, norm="CF", use_head=True, **kwargs):
        super(HCQTEncoder, self).__init__()

        self.depths = depths
        self.use_head = use_head
        total_depth = sum(depths)
        dp_rates = [drop_path_rate * i / (total_depth - 1) for i in range(total_depth)]
    
        self.layernorm = LayerNormCF(in_channels, in_bins)
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=11, padding=11 // 2, stride=(1, 1)), 
            nn.GELU(),
        )

        # 2. Stage 1
        self.stage1 = nn.ModuleList([
            ConvNeXtBlockMPE(in_channels=widths[0], 
                             in_bins=in_bins, 
                             out_channels=widths[0], 
                             kernel_size=kernel_sizes[0], 
                             drop_path=dp_rates[i],
                             #focus_dim=focus_dim,
                             secondary_dim=kernel_secondary_dim,
                             norm=norm
                             )
            for i in range(depths[0])
        ])

        # 3. Downsampling Frequency and increasing channels
        self.downsample = nn.Sequential(
            nn.Conv2d(widths[0], widths[1], kernel_size=(3, 1), stride=(3, 1)),
            nn.GELU(),
        )

        # 4. Stage 2
        self.stage2 = nn.ModuleList([
            ConvNeXtBlockMPE(in_channels=widths[1], 
                             in_bins=in_bins // 3, 
                             out_channels=widths[1], 
                             kernel_size=kernel_sizes[1], 
                             drop_path=dp_rates[i + depths[0]],
                            #focus_dim=focus_dim,
                             secondary_dim=kernel_secondary_dim,
                             norm=norm
                             )
            for i in range(depths[1])
        ])

        # 5. Classification Head
        self.head = nn.Sequential(
            nn.Conv2d(widths[1], widths[1], 1),
            nn.GELU(),
            nn.Conv2d(widths[1], out_channels, 1),
        )

    def forward(self, x):
        # x: (B, Harmonics, Freq, Time)
        x = self.layernorm(x)
        x = self.stem(x)
        for layer in self.stage1:
            x = layer(x)
        x = self.downsample(x)
        for layer in self.stage2:
            x = layer(x)
            
        return self.head(x) if self.use_head else x
    
if __name__ == "__main__":
    # Example usage
    model = HCQTEncoder(in_bins=216, in_channels=6, out_channels=1, widths=[20, 40], depths=[6, 10], kernel_sizes=[9, 15], drop_path_rate=0.1, focus_dim=None, kernel_secondary_dim=5, norm="CF", use_head=True)
    input_tensor = torch.randn(2, 6, 216, 437)  # (batch_size, harmonics, freq_bins, time_frames)
    output = model(input_tensor)
    print(output.shape)  # Expected shape: (2, 1, 72, 437) after downsampling frequency by a factor of 3