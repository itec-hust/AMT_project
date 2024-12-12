import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

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

# attention block
class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
      super(Attention, self).__init__()
      self.num_heads = num_heads
      head_dim = dim // num_heads
      self.scale = qk_scale or head_dim ** -0.5

      self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
      self.attn_drop = nn.Dropout(attn_drop)
      self.proj = nn.Linear(dim, dim)
      self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
      B, N, C = x.shape
      qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
      q, k, v = qkv[0], qkv[1], qkv[2]

      # *根据数据类型不同可以进行不同的运算，@是只做乘法运算
      attn = (q @ k.transpose(-2, -1)) * self.scale
      attn = attn.softmax(dim=-1)
      attn = self.attn_drop(attn)

      x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
      x = self.proj(x)
      x = self.proj_drop(x)
      return x

# Transformer block nn.GELU效果ReLu但是平滑函数
class Block(nn.Module):
  def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
               drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
    super(Block, self).__init__()
    self.norm1 = norm_layer(dim)
    self.attn = Attention(
      dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    self.norm2 = norm_layer(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

  def forward(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x


# ResNet bottleneck 其中第一个层将输入的通道数变为输出要求的1/4，第二个层进行特征提取，第三个层将通道*4变换到输出要求
class ConvBlock(nn.Module):
  def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
               norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
    super(ConvBlock, self).__init__()

    expansion = 4
    med_planes = outplanes // expansion

    self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn1 = norm_layer(med_planes)
    self.act1 = act_layer(inplace=True)

    self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
    self.bn2 = norm_layer(med_planes)
    self.act2 = act_layer(inplace=True)

    self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn3 = norm_layer(outplanes)
    self.act3 = act_layer(inplace=True)

    if res_conv:
      self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
      self.residual_bn = norm_layer(outplanes)
    
    self.res_conv = res_conv
    self.drop_block = drop_block
    self.drop_path = drop_path

  def zero_init_last_bn(self):
    nn.init.zeros_(self.bn3.weight)
  
  def forward(self, x, x_t=None, return_x_2=True):
    residual = x

    x = self.conv1(x)
    x = self.bn1(x)
    if self.drop_block is not None:
      x = self.drop_block(x)
    x = self.act1(x)

    # 第二层进行融合
    x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
    x = self.bn2(x)
    if self.drop_block is not None:
      x = self.drop_block(x)
    x2 = self.act2(x)

    x = self.conv3(x2)
    x = self.bn3(x)
    if self.drop_block is not None:
      x = self.drop_block(x)
    
    if self.drop_path is not None:
      x = self.drop_path(x)

    if self.res_conv:
      residual = self.residual_conv(residual)
      residual = self.residual_bn(residual)

    x += residual
    x = self.act3(x)

    if return_x_2:
      return x, x2
    
    return x

class FCUDown(nn.Module):
  def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
               norm_layer=partial(nn.LayerNorm, eps=1e-6)):
      super(FCUDown, self).__init__()
      self.dw_stride = dw_stride

      self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
      self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

      self.ln = norm_layer(outplanes)
      self.act = act_layer()

  # x_t的作用主要是考虑将cls_token拼接给来自CNN的特征
  def forward(self, x, x_t):
    x = self.conv_project(x)
    x = self.sample_pooling(x).flatten(2).transpose(1, 2)
    x = self.ln(x)
    x = self.act(x)
    x = torch.cat([x_t[:, 0][:, None], x], dim=1)

    return x


class FCUUp(nn.Module):
  def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
               norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
    super(FCUUp, self).__init__()
    self.up_stride = up_stride
    self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
    self.bn = norm_layer(outplanes)
    self.act = act_layer()
  
  # 跟原本论文描述有所区别，论文中先插值在做通道变换和BN等
  def forward(self, x, H, W):
    B, _, C, = x.shape
    # 序列维度的第一个是cls_token 融合的时候需要去掉保持维度一致
    x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W).contiguous()
    x_r = self.act(self.bn(self.conv_project(x_r)))

    return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


# 类似于ConvBlock 只不过ConvBlock中考虑了Transformer特征的融合以及中间层输出
# Med_ConvBlock只是给卷积部分提取特征用的
class Med_ConvBlock(nn.Module):
  def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
               drop_block=None, drop_path=None):
    super(Med_ConvBlock, self).__init__()

    expansion = 4
    med_planes = inplanes // expansion

    self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn1 = norm_layer(med_planes)
    self.act1 = act_layer(inplace=True)

    self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
    self.bn2 = norm_layer(med_planes)
    self.act2 = act_layer(inplace=True)

    self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn3 = norm_layer(inplanes)
    self.act3 = act_layer(inplace=True)

    self.drop_block = drop_block
    self.drop_path = drop_path

  def zero_init_last_bn(self):
    nn.init.zeros_(self.bn3.weight)

  def forward(self, x):
    residual = x

    x = self.conv1(x)
    x = self.bn1(x)
    if self.drop_block is not None:
      x = self.drop_block(x)
    x = self.act1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    if self.drop_block is not None:
      x = self.drop_block(x)
    x = self.act2(x)

    x = self.conv3(x)
    x = self.bn3(x)
    if self.drop_block is not None:
      x = self.drop_block(x)

    if self.drop_path is not None:
      x = self.drop_path(x)
    
    x += residual
    x = self.act3(x)

    return x


# 将ConvBlock和xxx结合起来，模型中使用的主要模块，能够将CNN的local特征和Transformer中的全局特征进行融合
# 也就是论文中的FCU模块 Feature Coupling Unit
class ConvTransBlock(nn.Module):
  def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
               qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
               last_fusion=False, num_med_block=0, groups=1):
    super(ConvTransBlock, self).__init__()
    expansion = 4
    self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)

    if last_fusion:
      self.fusion_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
    else:
      self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

    if num_med_block > 0:
      self.med_block = []
      for i in range(num_med_block):
        self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
      self.med_block = nn.Sequential(*self.med_block)

    self.squeeze_block = FCUDown(inplanes=outplanes//expansion, outplanes=embed_dim, dw_stride=dw_stride)
    self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes//expansion, up_stride=dw_stride)

    self.trans_block = Block(
      dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate
    )

    self.dw_stride = dw_stride
    self.embed_dim = embed_dim
    self.num_med_block = num_med_block
    self.last_fusion = last_fusion

  def forward(self, x, x_t):
    x, x2 = self.cnn_block(x)

    _, _, H, W = x2.shape
    x_st = self.squeeze_block(x2, x_t) # x_t的目的是传递cls_token
    x_t = self.trans_block(x_st + x_t)

    if self.num_med_block > 0:
      x = self.med_block(x)
    
    # 传入shape告诉需要reshape的尺寸
    x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
    x = self.fusion_block(x, x_t_r, return_x_2=False)

    return x, x_t


# Conformer中的Transformer没有使用位置信息，位置相关的信息靠CNN来补充
class Conformer(nn.Module):
  def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0, 
               embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0.,
               attn_drop_rate=0., drop_path_rate=0.):
    super(Conformer, self).__init__()
    # Transformer
    self.num_classes = num_classes
    self.num_features = self.embed_dim = embed_dim
    assert depth % 3 == 0
    
    # 随机初始化，分类问题中选择哪个序列来进行输出都不太合适，使用全局平均又使得每个序列权重相同且固定，不太好，
    # 于是考虑新增一个序列特征用来聚合所有序列特征
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
    
    # Classifier head
    self.trans_norm = nn.LayerNorm(embed_dim)
    self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    self.pooling = nn.AdaptiveAvgPool2d(1)
    self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

    # Stem stage: get the feature maps by conv block ( copied from resnet.py)
    self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.act1 = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.aggregation = nn.Conv3d(64, 64, (5, 1, 1), stride=(1, 1, 1), bias=False) #

    # 1 stage 64-->256
    stage_1_channel = int(base_channel * channel_ratio)
    trans_dw_stride = patch_size // 4
    self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
    self.trans_patch_conv = nn.Conv2d(256, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
    self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0]
                       )

    # 2~4 stage 256-->256 (卷积部分的瓶颈块会自己先down4在up4) 3 Block
    init_stage = 2
    fin_stage = depth // 3 + 1
    # self.add_module 向网络中注册模块，当结构既不是串行也不是并行时，适合在循环中使用
    for i in range(init_stage, fin_stage):
      self.add_module('conv_trans_' + str(i),
                      ConvTransBlock(
                        stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block
                      ))
    
    stage_2_channel = int(base_channel * channel_ratio * 2)
    # 5~8 stage 4 Block 256-->512-->512
    init_stage = fin_stage
    fin_stage = fin_stage + depth // 3
    for i in range(init_stage, fin_stage):
      # 每个开始块先进行stride=2并且进行通道变换，后续块通道不变，第一个块进行需要使用带卷积的残差连接（因为通道变化了）
      s = 2 if i == init_stage else 1
      in_channel = stage_1_channel if i == init_stage else stage_2_channel
      res_conv = True if i == init_stage else False
      self.add_module('conv_trans_' + str(i),
        ConvTransBlock(
          in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
          num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
          drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
          num_med_block=num_med_block
        )
      )

    stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
    # 9~12 stage 4 Block 512-->1024-->1024
    init_stage = fin_stage
    fin_stage = fin_stage + depth // 3
    for i in range(init_stage, fin_stage):
      s = 2 if i == init_stage else 1
      in_channel = stage_2_channel if i == init_stage else stage_3_channel
      res_conv = True if i == init_stage else False
      last_fusion = True if i == depth else False
      self.add_module('conv_trans_' + str(i),
        ConvTransBlock(
          in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
          num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
          drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
          num_med_block=num_med_block, last_fusion=last_fusion
        )
      )
    self.fin_stage = fin_stage

    trunc_normal_(self.cls_token, std=.02)

    self.apply(self._init_weights)
  
  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      trunc_normal_(m.weight, std=.02)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight, 1.)
      nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.GroupNorm):
      nn.init.constant_(m.weight, 1.)
    elif isinstance(m, nn.GroupNorm):
      nn.init.constant_(m.weight, 1.)
      nn.init.constant_(m.bias, 0.)

  @torch.jit.ignore
  def no_weight_decay(self):
    return {'cls_token'}

  def forward(self, x):
    B, K, C, H, W = x.shape
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = x.reshape(-1, C, H, W) # [5, 1, 144, 800]

    # aggregation
    x_base = self.maxpool(self.act1(self.bn1(self.conv1(x)))) # [5, 64, 36, 200]
    B_K, C, H, W = x_base.shape
    x_base = x_base.reshape(-1, K, C, H, W).permute(0, 2, 1, 3, 4).contiguous() # [1, 64, 5, 36, 200]
    x_base = self.aggregation(x_base).squeeze(dim=2) # [1, 64, 36, 200]

    # 1 stage
    x_base = self.conv_1(x_base, return_x_2=False) # [1, 256, 36, 200]
    x = x_base
    x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2) # [1, 450, 384] 450个patch
    x_t = torch.cat([cls_tokens, x_t], dim=1) # [1, 451, 384]
    x_t = self.trans_1(x_t) # [1, 451, 384]

    # 2~fianl
    for i in range(2, self.fin_stage): # self.fin_stage=13
      x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)

    # conv classification
    x_p = self.pooling(x).flatten(1)
    conv_cls = self.conv_cls_head(x_p)

    # trans classification
    x_t = self.trans_norm(x_t)
    trans_cls = self.trans_cls_head(x_t[:, 0]) # 使用cls_token维度来分类

    # return logits
    return (conv_cls, trans_cls)
