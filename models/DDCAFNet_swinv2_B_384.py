import torch
import numpy as np
from functools import partial
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import math
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from .AssNet_encoder import *
from logger_ import get_root_logger
from mmcv.cnn import build_norm_layer
from .swinv2 import swinv2_base_window8_384
import torch.nn.functional as F




import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import trunc_normal_
except Exception:
    from torch.nn.init import trunc_normal_


# -------------------------------------------------
# Utility
# -------------------------------------------------
def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1, groups=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=bias),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


# -------------------------------------------------
# Novelty 1:
# Dual-Domain Confidence-Gated Cross Attention
# -------------------------------------------------
class SpectralCrossAttention(nn.Module):
    """
    Frequency-domain cross-attention branch
    Similar in spirit to your CrossAttention
    """
    def __init__(
        self,
        query_dim,
        context_dim=None,
        out_dim=None,
        map_feature=(56, 56),
        num_heads=8,
        qkv_bias=False,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()

        context_dim = context_dim if context_dim is not None else query_dim
        out_dim = out_dim if out_dim is not None else query_dim

        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"

        self.dim = query_dim
        self.context_dim = context_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.H, self.W = map_feature
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, query_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(query_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(context_dim, context_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(context_dim)
        else:
            self.sr = None
            self.norm = None

        self.dwconv_v = nn.Conv2d(query_dim, query_dim, 3, 1, 1, groups=query_dim, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= max(1, m.groups)
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, context):
        """
        x:       [B, Cq, H, W]
        context: [B, Cc, Hc, Wc]
        """
        B, _, Hq, Wq = x.shape
        Bc, Cc, Hc, Wc = context.shape
        assert B == Bc

        x_tokens = x.permute(0, 2, 3, 1)  # [B,H,W,C]
        q = self.q(x_tokens).permute(0, 3, 1, 2)  # [B,C,H,W]

        if self.sr is not None:
            context_red = self.sr(context)  # [B,Cc,Hr,Wr]
            Hr, Wr = context_red.shape[-2:]
            context_red = context_red.flatten(2).transpose(1, 2)  # [B,Hr*Wr,Cc]
            context_red = self.norm(context_red)
            kv = self.kv(context_red).reshape(B, Hr * Wr, 2, self.num_heads, self.head_dim)
            kv = kv.permute(2, 0, 3, 4, 1)  # [2,B,heads,head_dim,N]
            k, v = kv[0], kv[1]             # [B,heads,head_dim,N]
            k = k.reshape(B, self.dim, Hr, Wr)
            v = v.reshape(B, self.dim, Hr, Wr)
            if (Hr, Wr) != (Hq, Wq):
                k = F.interpolate(k, size=(Hq, Wq), mode='bilinear', align_corners=False)
                v = F.interpolate(v, size=(Hq, Wq), mode='bilinear', align_corners=False)
        else:
            context_tokens = context.permute(0, 2, 3, 1)  # [B,Hc,Wc,Cc]
            kv = self.kv(context_tokens).reshape(B, Hc, Wc, 2, self.dim).permute(3, 0, 4, 1, 2)
            k, v = kv[0], kv[1]  # [B,C,Hc,Wc]
            if (Hc, Wc) != (Hq, Wq):
                k = F.interpolate(k, size=(Hq, Wq), mode='bilinear', align_corners=False)
                v = F.interpolate(v, size=(Hq, Wq), mode='bilinear', align_corners=False)

        v_local = self.dwconv_v(v)

        # Frequency correlation

        q_fft = torch.fft.rfft2(q.float().cpu(), s=(Hq, Wq), dim=(-2, -1), norm='ortho').to(q.device)
        k_fft = torch.fft.rfft2(k.float().cpu(), s=(Hq, Wq), dim=(-2, -1), norm='ortho').to(k.device)
        q_fft = q_fft.reshape(B, self.num_heads, self.head_dim, Hq, Wq // 2 + 1)
        k_fft = k_fft.reshape(B, self.num_heads, self.head_dim, Hq, Wq // 2 + 1)

        attn_fft = q_fft * k_fft
        attn_map = torch.fft.irfft2(attn_fft, s=(Hq, Wq), dim=(-2, -1), norm='ortho')
        attn_map = attn_map.reshape(B, self.dim, Hq, Wq)

        out = attn_map * v
        out = out + v_local

        out = out.permute(0, 2, 3, 1)
        out = self.proj(out).permute(0, 3, 1, 2)
        out = self.proj_drop(out)
        return out


class TokenSpectralResidualCrossAttention(nn.Module):
    """
    Token-domain cross attention + spectral residual value enhancement
    Similar in spirit to your CrossAttention1
    """
    def __init__(
        self,
        query_dim,
        context_dim=None,
        out_dim=None,
        map_feature=(56, 56),
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()

        context_dim = context_dim if context_dim is not None else query_dim
        out_dim = out_dim if out_dim is not None else query_dim

        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"

        self.dim = query_dim
        self.context_dim = context_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.H, self.W = map_feature
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, query_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(query_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(context_dim, context_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(context_dim)
            self.de_conv = nn.ConvTranspose2d(
                query_dim, query_dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
                groups=query_dim
            )
        else:
            self.sr = None
            self.norm = None
            self.de_conv = nn.Identity()

        # rfft2 width is W//2+1
        self.complex_weight = nn.Parameter(
            torch.randn(self.dim, self.H, self.W // 2 + 1, 2, dtype=torch.float32) * 0.02
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= max(1, m.groups)
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, context):
        """
        x:       [B, Cq, H, W]
        context: [B, Cc, Hc, Wc]
        """
        B, _, Hq, Wq = x.shape
        Bc, Cc, Hc, Wc = context.shape
        assert B == Bc

        x_tokens = x.reshape(B, self.dim, -1).permute(0, 2, 1)         # [B,N,C]
        context_tokens = context.reshape(B, self.context_dim, -1).permute(0, 2, 1)

        N = x_tokens.shape[1]
        q = self.q(x_tokens).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            context_red = self.sr(context).reshape(B, Cc, -1).permute(0, 2, 1)
            context_red = self.norm(context_red)
            kv = self.kv(context_red).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(context_tokens).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]  # [B,heads,Nc,head_dim]

        # token attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)

        # spectral residual enhancement of v
        v_img = v.permute(0, 1, 3, 2).reshape(B, self.dim, -1)

        if self.sr_ratio > 1:
            Hr = Hc // self.sr_ratio
            Wr = Wc // self.sr_ratio
        else:
            Hr, Wr = Hc, Wc

        v_img = v_img.reshape(B, self.dim, Hr, Wr)
        v_img = self.de_conv(v_img)

        if v_img.shape[-2:] != (Hq, Wq):
            v_img = F.interpolate(v_img, size=(Hq, Wq), mode='bilinear', align_corners=False)

        v_fft = torch.fft.rfft2(v_img.float(), s=(Hq, Wq), dim=(-2, -1), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)  # [C,H,W//2+1]
        v_fft = v_fft * weight.unsqueeze(0)
        v_res = torch.fft.irfft2(v_fft, s=(Hq, Wq), dim=(-2, -1), norm='ortho')
        v_res = v_res.reshape(B, self.dim, -1).permute(0, 2, 1)  # [B,N,C]

        x_out = x_attn + v_res
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        x_out = x_out.transpose(1, 2).reshape(B, self.out_dim, Hq, Wq)

        return x_out


class DualDomainCrossAttention(nn.Module):
    """
    Novelty 1:
    Adaptive fusion of two different cross-attention paradigms
    """
    def __init__(
        self,
        query_dim,
        context_dim=None,
        out_dim=None,
        map_feature=(56, 56),
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()

        out_dim = out_dim if out_dim is not None else query_dim

        self.spectral_branch = SpectralCrossAttention(
            query_dim=query_dim,
            context_dim=context_dim,
            out_dim=out_dim,
            map_feature=map_feature,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            sr_ratio=sr_ratio,
        )

        self.token_branch = TokenSpectralResidualCrossAttention(
            query_dim=query_dim,
            context_dim=context_dim,
            out_dim=out_dim,
            map_feature=map_feature,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            sr_ratio=sr_ratio,
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        hidden = max(out_dim // 4, 16)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim * 2, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.out_refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
        )

    def forward(self, x, context):
        x_fft = self.spectral_branch(x, context)
        x_tok = self.token_branch(x, context)

        fusion_feat = torch.cat([x_fft, x_tok], dim=1)

        g_sp = self.spatial_gate(fusion_feat)   # [B,1,H,W]
        g_ch = self.channel_gate(fusion_feat)   # [B,C,1,1]
        gate = g_sp * g_ch

        out = gate * x_fft + (1.0 - gate) * x_tok
        out = out + self.out_refine(out)
        return out



# -------------------------------------------------
# Novelty 2:
# Edge-guided adaptive frequency decoupling
# -------------------------------------------------
class EdgeGuidedFFParser(nn.Module):
    def __init__(self, in_channels, out_channels=None, map_feature=(56, 56)):
        super().__init__()
        self.H = map_feature[0]
        self.W = map_feature[1]
        out_channels = out_channels if out_channels is not None else in_channels

        self.conv_low = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_high = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_fuse = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.max = nn.MaxPool2d(2)
        self.deconv = nn.ConvTranspose2d(
            in_channels + in_channels * 2,
            in_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.edge_gate = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels // 4, 8), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(max(in_channels // 4, 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // 4, 8), 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        lpf, hpf = self._build_frequency_masks(self.H, self.W)
        self.register_buffer("lpf", lpf, persistent=False)
        self.register_buffer("hpf", hpf, persistent=False)

    def _build_frequency_masks(self, H, W):
        lpf = torch.zeros((H, W), dtype=torch.float32)
        R = (H + W) // 8
        for x in range(W):
            for y in range(H):
                if ((x - (W - 1) / 2) ** 2 + (y - (H - 1) / 2) ** 2) < (R ** 2):
                    lpf[y, x] = 1.0
        hpf = 1.0 - lpf
        return lpf[None, None, :, :], hpf[None, None, :, :]

    def forward(self, x, x_skip):
        """
        x:      [B,C,H,W]
        x_skip: [B,2C,H/2,W/2] or compatible
        """
        B, C, H, W = x.shape
        x_float = x.float()

        # FFT decomposition
        f = torch.fft.fftn(x_float, dim=(2, 3))
        f = torch.roll(f, shifts=(H // 2, W // 2), dims=(2, 3))

        if (H, W) != (self.H, self.W):
            lpf, hpf = self._build_frequency_masks(H, W)
            lpf = lpf.to(x.device)
            hpf = hpf.to(x.device)
        else:
            lpf = self.lpf.to(x.device)
            hpf = self.hpf.to(x.device)

        f_l = f * lpf
        f_h = f * hpf

        x_l = torch.abs(torch.fft.ifftn(f_l, dim=(2, 3)))
        x_h = torch.abs(torch.fft.ifftn(f_h, dim=(2, 3)))
        x_l = torch.fft.ifftn(f_l, dim=(2, 3)).real
        x_h = torch.fft.ifftn(f_h, dim=(2, 3)).real
        temp_l = self.conv_low(x_l)
        temp_h = self.conv_high(x_h)

        h = self.max(x_float)
        h = self.deconv(torch.cat((h, x_skip), dim=1))

        # edge-guided adaptive routing
        edge_prior = self.edge_gate(h)  # higher near boundaries
        freq_feat = (1.0 - edge_prior) * temp_l + edge_prior * temp_h

        out = self.conv_fuse(h + freq_feat)
        return out



# -------------------------------------------------
# Main Net
# -------------------------------------------------
class Net(nn.Module):
    def __init__(self, ckpt=None, img_size=(256, 256), encoder_ch=64, **kwargs):
        super(Net, self).__init__(**kwargs)

        assert isinstance(img_size, tuple) and len(img_size) == 2
        H, W = img_size[0] // 4, img_size[1] // 4
        self.img_size = img_size
        self.ckpt = ckpt

        
        print('img_size',img_size)
        self.backbone = swinv2_base_window8_384(img_size=img_size)  
        self.encoder = AssNet(3,encoder_ch)
        
        self.filter_bk = [128,256,512,1024]
        self.filter_enc = [encoder_ch, encoder_ch * 2, encoder_ch * 4, encoder_ch * 8]
        self.head =[4, 8, 16, 32]  
        

 
        self.ca1 = DualDomainCrossAttention(
            self.filter_bk[0], self.filter_enc[0], self.filter_enc[0],
            (H, W), num_heads=self.head[0], sr_ratio=4
        )
        self.ca2 = DualDomainCrossAttention(
            self.filter_bk[1], self.filter_enc[1], self.filter_enc[1],
            (H // 2, W // 2), num_heads=self.head[1], sr_ratio=2
        )
        self.ca3 = DualDomainCrossAttention(
            self.filter_bk[2], self.filter_enc[2], self.filter_enc[2],
            (H // 4, W // 4), num_heads=self.head[2], sr_ratio=2
        )
        self.ca4 = DualDomainCrossAttention(
            self.filter_bk[3], self.filter_enc[3], self.filter_enc[3],
            (H // 8, W // 8), num_heads=self.head[3], sr_ratio=1
        )

        self.down1 = nn.Conv2d(self.filter_enc[0], self.filter_enc[1], kernel_size=4, stride=2, padding=1, bias=False)
        self.down2 = nn.Conv2d(self.filter_enc[1], self.filter_enc[2], kernel_size=4, stride=2, padding=1, bias=False)
        self.down3 = nn.Conv2d(self.filter_enc[2], self.filter_enc[3], kernel_size=4, stride=2, padding=1, bias=False)
        self.down4 = nn.Conv2d(self.filter_enc[3], self.filter_enc[3], kernel_size=1, stride=1, padding=0, bias=False)

        self.final = nn.Sequential(
            nn.BatchNorm2d(self.filter_enc[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter_enc[3], out_channels=1024, kernel_size=1, padding=0)
        )

        # Novelty 2 integrated here
        self.edge1 = EdgeGuidedFFParser(in_channels=self.filter_enc[0], map_feature=(H, W))
        self.edge2 = EdgeGuidedFFParser(in_channels=self.filter_bk[0], map_feature=(H, W))

        self.edge_out = nn.Sequential(
            nn.Conv2d((self.filter_enc[0] + self.filter_bk[0]), self.filter_enc[3], kernel_size=1, padding=0),
            nn.BatchNorm2d(self.filter_enc[3]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.filter_enc[3], self.filter_enc[3] // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.filter_enc[3] // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.filter_enc[3] // 2, 1, kernel_size=4, stride=2, padding=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= max(1, m.groups)
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self):
        # Keep your original loading logic
        if self.ckpt is not None:
            ckpt = torch.load(self.ckpt, map_location='cpu')
            self.backbone.load_state_dict(ckpt['model'], strict=False)

    def forward(self, x, context):
        """
        x:       main input image
        context: auxiliary/modal/context image
        """
        B, C, H, W = x.shape

        # Backbone features
        out_7r, out_14r, out_28r, out_56r = self.backbone(x)[::-1]
        
        out_7r1, out_14r1, out_28r1, out_56r1, out_56r1_ = self.encoder(context)[::-1]

        ca1 = self.ca1(out_56r, out_56r1)
        ca1_down = self.down1(ca1)

        ca2 = self.ca2(out_28r, out_28r1)
        ca2_down = self.down2(ca1_down + ca2)

        ca3 = self.ca3(out_14r, out_14r1)
        ca3_down = self.down3(ca2_down + ca3)

        ca4 = self.ca4(out_7r, out_7r1)
        ca4 = self.down4(ca3_down + ca4)

        # Main prediction
        cc = self.final(ca4)
        cc1 = F.pixel_shuffle(cc, 32)

        # Edge branch
        edge1 = self.edge1(out_56r1_, out_28r1)
        edge2 = self.edge2(out_56r, out_28r)
        edge_out = self.edge_out(torch.cat((edge1, edge2), dim=1))

        return cc1, edge_out 