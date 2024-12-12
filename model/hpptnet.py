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

# 每一层在时间维度使用带空洞的TCN和频率维度的Transformer
class ConvLayer(nn.Module):
  def __init__(self, in_channel, out_channel=None, stride=1):
    super(ConvLayer, self).__init__()
    kernel_size = 3
    if out_channel is None:
      out_channel = in_channel
    embed_channel = out_channel // 4
    self.conv = nn.Sequential(
      nn.Conv2d(in_channel, embed_channel, 1, 1),
      nn.BatchNorm2d(embed_channel),
      nn.ReLU(inplace=True),
      nn.Conv2d(embed_channel, embed_channel, kernel_size, (stride, 1), 1),
      nn.BatchNorm2d(embed_channel),
      nn.ReLU(inplace=True),
      nn.Conv2d(embed_channel, out_channel, 1, 1),
      nn.BatchNorm2d(out_channel),
      nn.ReLU(inplace=True),
    )

  def forward(self, x):
    out = self.conv(x)
    return out

class HPPTNet(nn.Module):  
    def __init__(self, cqt_config, conv_channels, convtrans_channel, output_channel, dilations=None, n_layer=None,
                  num_heads=None, num_classes=88, ignore_offset_head=False, ignore_velocity_head=True,
                  shared_head=True, freq_sparse=True, freq_attn=True):
        super(HPPTNet, self).__init__()
        self.cqt = CQTSpectrogram(**cqt_config)
        conv_base = []
        for idx in range(len(conv_channels)-1):
          conv_base.append(nn.Conv2d(conv_channels[idx], conv_channels[idx+1], 3, padding=1, bias=False))
          conv_base.append(nn.BatchNorm2d(conv_channels[idx+1]))
          conv_base.append(nn.ReLU(inplace=True))
        self.conv_base = nn.Sequential(*conv_base)

        # # Time-Freq attention
        if freq_attn:
          freq_attn_mask = self._make_freq_attn_mask(cqt_config) if freq_sparse else None
          self.onset = self._make_head(dilations, num_heads, freq_attn_mask, conv_channels[-1], 
                                        convtrans_channel, output_channel)
          self.frame = self._make_head(dilations, num_heads, freq_attn_mask, conv_channels[-1], 
                                          convtrans_channel, output_channel)
        else:
          self.onset = self._make_head_v2(conv_channels[-1], convtrans_channel, output_channel, n_layer)
          self.frame = self._make_head_v2(conv_channels[-1], convtrans_channel, output_channel, n_layer)
        
        # # AVGPooling layer && cls head
        self.onset_pool = nn.AdaptiveAvgPool2d(1)
        self.frame_pool = nn.AdaptiveAvgPool2d(1)

        if shared_head:
          self.head = nn.Linear(output_channel, num_classes)
        else:
          self.onset_head = nn.Linear(output_channel, num_classes)
          self.frame_head = nn.Linear(output_channel, num_classes)

        if not ignore_offset_head:
          if freq_attn:
            self.offset = self._make_head(dilations, num_heads, freq_attn_mask, conv_channels[-1], 
                                          convtrans_channel, output_channel)
          else:
            self.offset = self._make_head_v2(conv_channels[-1], convtrans_channel, output_channel, n_layer)
          self.offset_pool = nn.AdaptiveAvgPool2d(1)
          if not shared_head:
            self.offset_head = nn.Linear(output_channel, num_classes)

        if not ignore_velocity_head:
          if freq_attn:
            self.velocity = self._make_head(dilations, num_heads, freq_attn_mask, conv_channels[-1], 
                                        convtrans_channel, output_channel)
          else:
            self.velocity = self._make_head_v2(conv_channels[-1], convtrans_channel, output_channel, n_layer)
          self.velocity_pool = nn.AdaptiveAvgPool2d(1)
          # velocity的意义不是概率，不共享检测头
          self.velocity_head = nn.Linear(output_channel, num_classes)       

        self.ignore_offset_head = ignore_offset_head
        self.ignore_velocity_head = ignore_velocity_head
        self.shared_head = shared_head

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
  
    def _make_head_v2(self, in_channel, embed_channel, out_channel, n_layer):
      layers = []
      for i in range(n_layer):
        if i == 0:
          layers.append(ConvLayer(in_channel, embed_channel))
        if i == n_layer - 1:
          layers.append(ConvLayer(embed_channel, stride=4))
        else:
          layers.append(ConvLayer(embed_channel))
        
      down = [
        nn.Conv2d(embed_channel, out_channel, 3, (4, 1), 1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
      ]
      layers.extend(down)
      layers = nn.Sequential(*layers)
      return layers

    # 先不考虑velocity问题
    def forward(self, audio):
      # audio [1, 96000]
      # [1, 1, 352, 61]
      x = self.cqt(audio)[:, None]
      # [1, 16, 352, 61]
      x = self.conv_base(x)

      onset = self.onset_pool(self.onset(x)).flatten(1)
      frame = self.frame_pool(self.frame(x)).flatten(1)

      if self.shared_head:
        onset_logits = self.head(onset)
        frame_logits = self.head(frame)
      else:
        onset_logits = self.onset_head(onset)
        frame_logits = self.frame_head(frame)

      if self.ignore_offset_head:
        return onset_logits, frame_logits

      offset = self.offset_pool(self.offset(x)).flatten(1)
      if self.shared_head:
        offset_logits = self.head(offset)
      else:
        offset_logits = self.offset_head(offset)

      if self.ignore_velocity_head:
        return onset_logits, offset_logits, frame_logits
      
      velocity = self.velocity_pool(self.velocity(x)).flatten(1)
      velocity_logits = self.velocity_head(velocity)

      return onset_logits, offset_logits, frame_logits, velocity_logits

if __name__ == '__main__':
  from ptflops import get_model_complexity_info
  cqt_config = {
    'sr': 16000,
    'hop_length': 320,
    'n_fft': 2048,
    'fmin': 27.5,
    'n_bins': 352,
    'bins_per_octave': 48
  }
  conv_channels = [1, 16, 16]
  convtrans_channel = 128
  output_channel = 256
  dilations = [1, 1, 2, 2]
  num_heads = 1
  device = 'cuda'
  # model = HPPTNet(cqt_config, conv_channels, convtrans_channel, 
  #                   output_channel, dilations, num_heads=num_heads, shared_head=True, ignore_offset_head=True)
  model = HPPTNet(cqt_config, conv_channels, convtrans_channel, 
                    output_channel, n_layer=4, ignore_offset_head=False, shared_head=True, freq_attn=False)
  model = model.to(device)
  def prepare_input(shapes):
    x = torch.randn(1, *shapes)
    x /= (x.max() + 1e-7)
    x = x.to(device)
    return dict(audio = x)
 
  flops, params = get_model_complexity_info(model, (int(cqt_config['sr']*6), ), input_constructor=prepare_input,
                                            as_strings=True, print_per_layer_stat=True)
  print('Flops: ' + flops)
  print('Params: ' + params)
