import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3,5), padding=(1,2)),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity(),
            nn.BatchNorm2d(out_ch)
        )
        self.pool = nn.MaxPool2d((1,2))
        self.act = nn.GELU()
        self.se = SEBlock(out_ch, reduction=8)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.block(x)
        x = self.act(x + residual)  # residual connection
        x = self.se(x)  # SE block after activation
        x = self.pool(x)
        return x


class RangeAttentionPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels//4, 1),
            nn.GELU(),
            nn.Conv1d(channels//4, channels, 1)
        )

    def forward(self,x):
        B,C,T,R = x.shape

        x = x.permute(0,2,1,3)

        x = x.reshape(B*T,C,R)

        w = self.attn(x)

        w = torch.softmax(w, dim=-1)

        pooled = (x * w).sum(-1)

        pooled = pooled.reshape(B,T,C)

        pooled = pooled.permute(0,2,1)

        return pooled



class TemporalAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim,1)

    def forward(self,x,padding_mask=None):

        w = self.attn(x)
        if padding_mask is not None:
            w = w.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))

        w = torch.softmax(w,dim=1)

        x = (x*w).sum(1)

        return x


class RadarModel(nn.Module):
    def __init__(self, num_classes=126, in_channels=12, T=64):
        super().__init__()
        # scaled CNN backbone
        self.backbone = nn.Sequential(
            ResidualConvBlock(in_channels,48),
            ResidualConvBlock(48,96),
            ResidualConvBlock(96,192),
            ResidualConvBlock(192,384)
        )

        self.range_pool = RangeAttentionPool(384)
        self.pos = nn.Parameter(torch.randn(1,T,384) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=384,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.2,
            batch_first=True
        )

        self.temporal = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )

        self.temporal_pool = TemporalAttentionPool(384)
        self.cls = nn.Linear(384,num_classes)


    def forward(self,x,padding_mask=None):
        feat = self.backbone(x)
        
        feat = self.range_pool(feat)

        feat = feat.transpose(1,2)

        feat = feat + self.pos[:,:feat.size(1)]

        feat = self.temporal(feat, src_key_padding_mask=padding_mask)

        feat = self.temporal_pool(feat, padding_mask=padding_mask)

        out = self.cls(feat)

        return out