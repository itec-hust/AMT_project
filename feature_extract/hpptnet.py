import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import nnAudio.features
from functools import partial
import librosa
import numpy as np

# 受限于音频长度
class CQTSpectrogram(nn.Module):
    def __init__(self, sr, hop_length, fmin, n_bins, bins_per_octave, n_fft=None, log_scale=True):
        super(CQTSpectrogram, self).__init__()
        self.cqt = nnAudio.features.cqt.CQT(sr=sr, hop_length=hop_length, fmin=fmin,
                            n_bins=n_bins, bins_per_octave=bins_per_octave, center=False)
        self.log_scale = log_scale

    def forward(self, audio):
        cqt = self.cqt(audio)
        if self.log_scale:
          cqt = torch.log(torch.clamp(cqt, min=1e-7))
        return cqt

# # some problem cannot be used not the same as V1 or V3
class CQTSpectrogramV2(nn.Module):
    def __init__(self, sr, hop_length, fmin, n_bins, bins_per_octave, n_fft=None, log_scale=True):
        super().__init__()
        # CQTSpectrogramV2
        fft_basis, n_fft, lengths = self._cqt_filter_fft(
          sr, fmin, n_bins, bins_per_octave, hop_length
        )
        fft_basis = np.abs(fft_basis)
        fft_basis = torch.tensor(fft_basis).float()
        self.register_buffer('fft_basis', fft_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_scale = log_scale

    def _cqt_filter_fft(self, sr, fmin, n_bins, bins_per_octave, hop_length,
              filter_scale=1, norm=1, window="hann", gamma=0.0):
      freqs = librosa.cqt_frequencies(fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
      def __bpo_to_alpha(bins_per_octave):
        r = 2 ** (1 / bins_per_octave)
        return (r ** 2 - 1) / (r ** 2 + 1)
      alpha = __bpo_to_alpha(bins_per_octave)
      lengths, _ = librosa.filters.wavelet_lengths(
        freqs=freqs, sr=sr, window=window, filter_scale=filter_scale, alpha=alpha
      )

      basis, lengths = librosa.filters.wavelet(freqs=freqs, sr=sr, window=window, filter_scale=filter_scale,
                                                alpha=alpha)
      n_fft = basis.shape[1]
      if hop_length is not None and n_fft < 2.0 ** (1 + np.ceil(np.log2(hop_length))):
          n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))
      basis *= lengths[:, np.newaxis] / float(n_fft)

      # FFT and retain only the non-negative frequencies
      fft = librosa.get_fftlib()
      fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, : (n_fft // 2) + 1]

      return fft_basis, n_fft, lengths

    def pad_audio(self, audio):
      n_segment = max(1, math.ceil((audio.shape[-1] - self.n_fft) / self.hop_length) + 1)
      pad_length = (n_segment - 1) * self.hop_length + self.n_fft - audio.shape[-1]
      audio = F.pad(audio, (0, pad_length))
      return audio

    def complex_matmul(self, x, y):
      real_x, imag_x = torch.split(x, 1, dim=-1)
      real_y, imag_y = torch.split(y, 1, dim=-1)

      real_out = torch.einsum('bqkd, bktd -> bqt', [real_x, real_y]) - \
                  torch.einsum('bqkd, bktd -> bqt', [imag_x, imag_y])
      imag_out = torch.einsum('bqkd, bktd -> bqt', [real_x, imag_y]) + \
                  torch.einsum('bqkd, bktd -> bqt', [imag_x, real_y])

      return real_out + 1j*imag_out

    # pseudo_cqt
    def forward(self, audio):
        audio = self.pad_audio(audio)
        stft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, center=False)
        stft = (stft.pow(2).sum(dim=-1) + 1e-7).sqrt()
        cqt = torch.einsum('bft, qf -> bqt', [stft, self.fft_basis])
        cqt /= torch.sqrt(torch.tensor(self.n_fft).float()) # scale = True
        if self.log_scale:
          cqt = torch.log(torch.clamp(cqt, min=1e-7))
        return cqt


class CQTSpectrogramV3(nn.Module):
    def __init__(self, sr, hop_length, fmin, n_fft, n_bins, bins_per_octave,
                  log_scale=True, pad_audio=False):
        super(CQTSpectrogramV3, self).__init__()
        # CQTSpectrogramV3
        fft_basis = self.get_weights(sr, n_fft, n_bins, fmin, bins_per_octave)
        fft_basis = torch.tensor(fft_basis)
        self.register_buffer('fft_basis', fft_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_scale = log_scale
        self.pad_audio = pad_audio
        self.register_buffer('scale', torch.tensor(float(1.0 / self.n_fft)))

    def get_weights(self, sr, n_fft, n_bins, fmin, bins_per_octave, norm="slaney"):

      # Initialize the weights
      weights = np.zeros((n_bins, int(1 + n_fft // 2)), dtype=np.float32)

      # Center freqs of each FFT bin
      fftfreqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

      # 'Center freqs' of mel bands - uniformly spaced between limits
      # mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
      freqs = librosa.cqt_frequencies(fmin=fmin/2**(1/bins_per_octave), n_bins=n_bins+2,
                                        bins_per_octave=bins_per_octave)

      # 353
      fdiff = np.diff(freqs) # 频率差距 freqs[i+1] - freq[i] --> out[i]
      # [354, 1025]
      ramps = np.subtract.outer(freqs, fftfreqs) # 二维矩阵out[i, j] = freqs[i] - fftfreqs[j]
      for i in range(n_bins):
          # lower and upper slopes for all bins
          lower = -ramps[i] / fdiff[i]
          upper = ramps[i + 2] / fdiff[i + 1]
          # .. then intersect them with each other and zero
          weights[i] = np.maximum(0, np.minimum(lower, upper))

      if norm == "slaney":
          # Slaney-style mel is scaled to be approx constant energy per channel
          enorm = 2.0 / (freqs[2 : n_bins + 2] - freqs[:n_bins])
          weights *= enorm[:, np.newaxis]
      else:
          weights = librosa.util.normalize(weights, norm=norm, axis=-1)

      # Only check weights if f_mel[0] is positive
      # fft频点数量不够，滤波器系数为全零 按照n_fft=2048 n_bins=352 只有254个滤波器组系数非全零
      # if not np.all((freqs[:-2] == 0) | (weights.max(axis=1) > 0)):
      #     # This means we have an empty channel somewhere
      #     import warnings
      #     warnings.warn(
      #         "Empty filters detected in mel frequency basis. "
      #         "Some channels will produce empty responses. "
      #         "Try increasing your sampling rate (and fmax) or "
      #         "reducing n_mels."
      #     )

      return weights

    def pad_audio(self, audio):
      n_segment = max(1, math.ceil((int(audio.shape[-1]) - self.n_fft) / self.hop_length) + 1)
      pad_length = (n_segment - 1) * self.hop_length + self.n_fft - audio.shape[-1]
      if pad_length >0:
        audio = F.pad(audio, (0, pad_length))
      return audio

    def forward(self, audio):
      if self.pad_audio:
        audio = self.pad_audio(audio)
      stft = torch.stft(audio, n_fft=self.n_fft, win_length=self.n_fft,
                        hop_length=self.hop_length, center=False)
      stft = (stft.pow(2).sum(dim=-1) + 1e-7).sqrt()
      cqt = torch.einsum('bft, qf -> bqt', [stft, self.fft_basis])
      cqt = cqt * self.scale # scale = True
      if self.log_scale:
        cqt = torch.log(torch.clamp(cqt, min=1e-7))
      return cqt

class Mlp(nn.Module):
  def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
    super(Mlp, self).__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    # up projection 向上投影
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = act_layer()
    # down projection 向下投影
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop = nn.Dropout(drop)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# attention block
class Attention(nn.Module):
    def __init__(self, dim, num_heads, freq_attn_mask=None, qkv_bias=False,
                  qk_scale=None, attn_drop=0., proj_drop=0., apply_sparse_attn=False):
      super(Attention, self).__init__()
      self.num_heads = num_heads
      head_dim = dim // num_heads
      self.scale = qk_scale or head_dim ** -0.5
      self.apply_sparse_attn = apply_sparse_attn

      if freq_attn_mask is not None:
        self.register_buffer('freq_attn_mask', freq_attn_mask.int())
        if self.apply_sparse_attn:
          x_indices, y_indices = torch.nonzero(~self.freq_attn_mask, as_tuple=True)
          self.attn_contex = [[] for _ in range(self.freq_attn_mask.shape[0])]
          for x, y in zip(x_indices, y_indices):
            self.attn_contex[int(x)].append(int(y))

          self.attn_contex_num = [0 for _ in range(len(self.attn_contex[0]))]
          for idx in range(len(self.attn_contex)):
            self.attn_contex_num[len(self.attn_contex[idx])-1] += 1

      self.qk_v = nn.Linear(dim, dim * 2, bias=qkv_bias)
      self.attn_drop = nn.Dropout(attn_drop)
      self.proj = nn.Linear(dim, dim)
      self.proj_drop = nn.Dropout(proj_drop)

    def _apply_mask_attn(self, q, k, v):
      # *根据数据类型不同可以进行不同的运算，@是只做乘法运算
      attn = (q @ k.transpose(-2, -1)) * self.scale

      if hasattr(self, 'freq_attn_mask'):
        # 将不关注的频点attn分数设置-inf
        mask_value = max_neg_value(attn)
        attn.masked_fill_(self.freq_attn_mask.bool(), mask_value)

      attn = attn.softmax(dim=-1)
      attn = self.attn_drop(attn)

      B, num_heads, N, head_dim = q.shape
      x = (attn @ v).transpose(1, 2).reshape(B, N, num_heads*head_dim).contiguous()
      return x

    # try to use sparse_attn using block matmul but failed. require more GPU memory
    def _apply_sparse_attn(self, q, k, v):
      # q k v [5, 1, 352, 128]
      B, num_heads, N, head_dim = q.shape
      # k [5, 1, 128, 352]
      start = 0
      values = torch.zeros(B, num_heads, N, head_dim, device=q.device)
      for block in range(len(self.attn_contex_num)):
        block_length = self.attn_contex_num[len(self.attn_contex_num)-block-1]
        end = start + block_length
        # 5, 1, 200, 9
        attn = torch.zeros(B, num_heads, block_length, len(self.attn_contex[start]),
                            device=q.device)
        # 5, 1, 200, 128, 9
        vs = torch.zeros(B, num_heads, block_length, head_dim, len(self.attn_contex[start]),
                          device=q.device)
        for idx, harmonic in enumerate(self.attn_contex[start]):
          q_block = q[:, :, harmonic: harmonic + block_length] # [5, 1, 200, 128]
          k_block = k[:, :, harmonic: harmonic + block_length] # [5, 1, 200, 128]
          attn[..., idx] = torch.einsum('bhlc, bhlc -> bhl', [q_block, k_block])
          v_block = v[:, :, harmonic: harmonic + block_length] # [5, 1, 200, 128]
          vs[..., idx] = v_block

        attn = attn * self.scale
        # 5, 1, 200, 128
        values[:, :, start: end] = torch.einsum('bhlk, bhlck -> bhlc', [attn, vs])
        start += block_length

      return values

    def forward(self, x):
      B, N, C = x.shape # N = 352
      qkv = self.qk_v(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
      q, k, v = qkv[0], qkv[0], qkv[1] # 5, 1, 352, 12

      if self.apply_sparse_attn:
        x = self._apply_sparse_attn(q, k, v)
      else:
        x = self._apply_mask_attn(q, k, v)

      x = self.proj(x)
      x = self.proj_drop(x)
      return x

# Transformer block nn.GELU效果ReLu但是平滑函数
class Transformer(nn.Module):
  def __init__(self, dim, num_heads, freq_attn_mask=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
               act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
    super(Transformer, self).__init__()
    self.norm1 = norm_layer(dim)
    self.attn = Attention(
      dim, num_heads, freq_attn_mask, qkv_bias, qk_scale, attn_drop, drop)
    self.norm2 = norm_layer(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

  def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x

# 每一层在时间维度使用带空洞的TCN和频率维度的Transformer
class ConvTrans(nn.Module):
  def __init__(self, in_channel, dilation, num_heads, freq_attn_mask=None, out_channel=None, stride=1):
    super(ConvTrans, self).__init__()
    kernel_size = 3
    padding = dilation * (kernel_size // 2)
    if out_channel is None:
      out_channel = in_channel
    # 频率维度不下采样
    if stride == 1:
      self.conv = nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size, 1, padding, dilation, bias=False),
        nn.ReLU(inplace=True)
      )
    # 频率维度下采样
    else:
      self.conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, (stride, 1), kernel_size//2, bias=False),
        nn.ReLU(inplace=True)
      )
    self.stride = stride
    self.trans = Transformer(out_channel, num_heads, freq_attn_mask) # 最后的考虑88个音符之间的关系

  def forward(self, x):

    if self.stride == 1:
      # x [1, 16, 352, 61]
      b, c, f, t = x.shape
      # [352, 16, 61]
      x = x.permute(0, 2, 1, 3).reshape(-1, c, t)
      # [352, 128, 61]
      x = self.conv(x)
      bf, c, t = x.shape
      # [61, 352, 128]
      x = x.reshape(-1, f, c, t).permute(0, 3, 1 ,2).reshape(-1, f, c)
    else:
      # [1, 128, 88, 61]
      x = self.conv(x)
      #
      b, c, f, t = x.shape
      # [61, 88, 128]
      x = x.permute(0, 3, 2, 1).reshape(-1, f, c)
    # [1, 128, 352, 61]
    out = self.trans(x).reshape(-1, t, f, c).permute(0, 3, 2, 1)
    return out

class HPPTNet(nn.Module):
    def __init__(self, cqt_config, conv_channels, convtrans_channel, output_channel, dilations,
                  num_heads, num_classes=88):
        super(HPPTNet, self).__init__()
        self.cqt = CQTSpectrogram(**cqt_config)
        conv_base = []
        for idx in range(len(conv_channels)-1):
          conv_base.append(nn.Conv2d(conv_channels[idx], conv_channels[idx+1], 3, padding=1, bias=False))
          conv_base.append(nn.BatchNorm2d(conv_channels[idx+1]))
          conv_base.append(nn.ReLU(inplace=True))
        self.conv_base = nn.Sequential(*conv_base)

        # # Time-Freq attention
        freq_attn_mask = self._make_freq_attn_mask(cqt_config)
        self.onset = self._make_head(dilations, num_heads, freq_attn_mask, conv_channels[-1],
                                      convtrans_channel, output_channel)
        self.offset = self._make_head(dilations, num_heads, freq_attn_mask, conv_channels[-1],
                                        convtrans_channel, output_channel)
        self.frame = self._make_head(dilations, num_heads, freq_attn_mask, conv_channels[-1],
                                        convtrans_channel, output_channel)

        # # AVGPooling layer
        self.onset_pool = nn.AdaptiveAvgPool2d(1)
        self.offset_pool = nn.AdaptiveAvgPool2d(1)
        self.frame_pool = nn.AdaptiveAvgPool2d(1)

        # # cls head
        self.onset_head = nn.Linear(output_channel, num_classes)
        self.offset_head = nn.Linear(output_channel, num_classes)
        self.frame_head = nn.Linear(output_channel, num_classes)

    def _make_freq_attn_mask(self, cqt_config, max_ratio=9):
      # 计算mask矩阵，哪些地方不关注
      mask = torch.ones(cqt_config['n_bins'], cqt_config['n_bins']).bool()
      for n_bin in range(cqt_config['n_bins']):
        for ratio in range(1, max_ratio + 1):
          bin_distance = int(0.5 + cqt_config['bins_per_octave'] * math.log2(ratio))
          if n_bin + bin_distance < cqt_config['n_bins']:
            mask[n_bin][n_bin + bin_distance] = False
      return mask

    def _make_head(self, dilations, num_heads, freq_attn_mask, conv_channel,
                    convtrans_channel, out_channel):
      conv_trans = []
      for idx, dilation in enumerate(dilations):
        if idx == 0:
          conv_trans.append(ConvTrans(conv_channel, dilation, num_heads,
                              freq_attn_mask, out_channel=convtrans_channel))
        if idx == len(dilations) - 1:
          conv_trans.append(ConvTrans(convtrans_channel, dilation, num_heads, stride=4))
        else:
          conv_trans.append(ConvTrans(convtrans_channel, dilation, num_heads, freq_attn_mask))
      # downsampling
      down = [
        nn.Conv2d(convtrans_channel, out_channel, 3, (4, 1), 1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
      ]
      conv_trans.extend(down)
      conv_trans = nn.Sequential(*conv_trans)

      return conv_trans

    # 先不考虑velocity问题
    def forward(self, audio):
      # audio [1, 96000]
      # [1, 1, 352, 61]
      x = self.cqt(audio)[:, None]
      # [1, 16, 352, 61]
      x = self.conv_base(x)

      onset = self.onset_pool(self.onset(x)).flatten(1)
      offset = self.offset_pool(self.offset(x)).flatten(1)
      frame = self.frame_pool(self.frame(x)).flatten(1)

      return onset, offset, frame