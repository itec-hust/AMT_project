import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class VideoAudioConcat(nn.Module):
  def __init__(self, video_dim=768, audio_dim=256, drop_prb=0.):
    super(VideoAudioConcat, self).__init__()
    self.cls = nn.Sequential(
      nn.Dropout(drop_prb),
      nn.Linear(video_dim + audio_dim, 512),
      nn.ReLU(inplace = True),
      nn.Dropout(drop_prb),
      nn.Linear(512, 88)
    )

  def forward(self, video_feature, audio_feature):
    x = torch.cat([video_feature, audio_feature], dim=-1)
    logits = self.cls(x)
    return logits


'''
reference https://github.com/Justin1904/Low-rank-Multimodal-Fusion/blob/master/model.py
'''
class VideoAudioLMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, output_dim, rank):
        '''
        Args:
            input_dims - a length-2 tuple, contains (video_dim, audio_dim)
            hidden_dims - another length-2 tuple, hidden dims of the sub-networks
            dropouts - a length-3 tuple, contains (video_dropout, audio_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(VideoAudioLMF, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.video_in = input_dims[0]
        self.audio_in = input_dims[1]
        self.output_dim = output_dim
        self.rank = rank

        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_in + 1, self.output_dim))
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_in + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, video_h, audio_h):

        batch_size = audio_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        # 1, 769
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        # 1, 257
        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)

        # 8, 1, 88
        fusion_video = torch.matmul(_video_h, self.video_factor) # video_factor 8, 257, 88
        # 8, 1, 88
        fusion_audio = torch.matmul(_audio_h, self.audio_factor) # audio_factor 8, 257, 88
        fusion_zy = fusion_video * fusion_audio # 8, 1, 88

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        # 1, 88
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias # fusion_weights 1, 8
        output = output.view(-1, self.output_dim)
 
        return output

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
      super(CrossAttention, self).__init__()
      self.num_heads = num_heads
      head_dim = dim // num_heads
      self.scale = qk_scale or head_dim ** -0.5

      self.qkv_x = nn.Linear(dim, dim * 2, bias=qkv_bias)
      self.qkv_y = nn.Linear(dim, dim * 2, bias=qkv_bias)
      self.attn_drop = nn.Dropout(attn_drop)
      self.proj_x = nn.Linear(dim, dim)
      self.proj_drop_x = nn.Dropout(proj_drop)
      self.proj_y = nn.Linear(dim, dim)
      self.proj_drop_y = nn.Dropout(proj_drop)

    def _apply_attn(self, q, k, v):
      # *根据数据类型不同可以进行不同的运算，@是只做乘法运算
      attn = (q @ k.transpose(-2, -1)) * self.scale
      attn = attn.softmax(dim=-1)
      attn = self.attn_drop(attn)

      B, num_heads, N, head_dim = q.shape
      x = (attn @ v).transpose(1, 2).reshape(B, N, num_heads*head_dim).contiguous()
      return x

    def forward(self, x, y):
      # x, y的特征维度已经事先调整到一致
      B, Nx, C = x.shape 
      B, Ny, C = y.shape
      qkv_x = self.qkv_x(x).reshape(B, Nx, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
      qx, kx, vx = qkv_x[0], qkv_x[0], qkv_x[1] # 5, 1, 352, 12
      qkv_y = self.qkv_y(x).reshape(B, Ny, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
      qy, ky, vy = qkv_y[0], qkv_y[0], qkv_y[1] # 5, 1, 352, 12
      x = self._apply_attn(qx, ky, vy)
      y = self._apply_attn(qy, kx, vx)

      x = self.proj_x(x)
      x = self.proj_drop_x(x)

      y = self.proj_y(y)
      y = self.proj_drop_y(y)

      return x, y



class VideoAudioAttn(nn.Module):
  def __init__(self, video_dim, audio_dim, hidden_dim, num_heads, attn_drop=0.5, proj_drop=0.5, final_drop=0.0):
    super(VideoAudioAttn, self).__init__()

    self.linear_v = nn.Sequential(
      nn.Linear(video_dim, hidden_dim),
      nn.Dropout(proj_drop)
    )
    self.linear_a = nn.Sequential(
      nn.Linear(audio_dim, hidden_dim),
      nn.Dropout(proj_drop)
    )
    self.lstmv = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.lstma = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.attn = CrossAttention(2*hidden_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
    self.cls = nn.Linear(4 * hidden_dim, 88)
    if final_drop > 0.:
        self.drop = nn.Dropout(final_drop)
    self.final_drop = final_drop

  def forward(self, video_feature, audio_feature):
    v = self.linear_v(video_feature)
    a = self.linear_a(audio_feature)
    v, _ = self.lstmv(v)
    a, _ = self.lstma(a)
    v, a = self.attn(v, a)
    x = torch.cat([v, a], dim=-1)
    logits = self.cls(x)
    if self.final_drop > 0.:
        logits = self.drop(logits)
    return logits


