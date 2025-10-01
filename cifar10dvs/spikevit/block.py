
import torch
from torch import nn

from .spiking import SpikeConv, SpikeDense, SpikeBottleneck


class ToFormerBridge(nn.Module):
  def __init__(self, in_channels, embed_dim):
    super().__init__()

    self.token_reduce = SpikeDense(in_channels=embed_dim,
                                   out_channels=in_channels)
    

    self.token_expand = SpikeDense(in_channels=in_channels,
                                   out_channels=embed_dim)
    
  def forward(self, x, tokens):
    # Reduce Token Dimension
    z = self.token_reduce(tokens).transpose(2,3) # T B M C

    # Reshape Input
    x = x.flatten(3,4) # T B C L
    xT = x.transpose(2,3) # T B L C

    # Matrix Multiplication
    z_x = torch.matmul(z, x) # T B M L
    z_x_xT = torch.matmul(z_x, xT) # T B M C

    # Expand Token Dimension
    z = self.token_expand(z_x_xT.transpose(2,3)) # T B D M

    # Residual Connection
    return z + tokens

class ToMobileBridge(nn.Module):
  def __init__(self, in_channels, embed_dim):
    super().__init__()

    self.k_proj = SpikeDense(in_channels=embed_dim,
                             out_channels=in_channels)
    
    self.v_proj = SpikeDense(in_channels=embed_dim,
                             out_channels=in_channels)
  
  def forward(self, x, tokens):

    # Reshape Input
    x_ = x.flatten(3,4).transpose(2,3) # T B L C

    # Project Tokens
    k = self.k_proj(tokens) # T B C M 
    v = self.v_proj(tokens).transpose(2,3) # T B M C

    # Matrix Multiplication
    x_k = torch.matmul(x_,k)     # T B L M 
    x_k_v = torch.matmul(x_k, v) # T B L C

    # Reshape Output
    x_ = x_k_v.transpose(2,3).reshape(x.shape)

    # Residual Connection
    return x + x_

class SpikeMobile(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, stride, padding):
    super().__init__()

    self.conv = SpikeConv(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding) 
    
  def forward(self, x):
    return self.conv(x)

    
# class SpikeMobile(nn.Module):
#   def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, stride, padding):
#     super().__init__()

#     self.downsample = False
#     if stride > 1:
#       self.downsample = True     
#       self.conv = SpikeConv(in_channels=in_channels,
#                             out_channels=in_channels,
#                             kernel_size=kernel_size,
#                             stride=stride,
#                             padding=padding) 
    
#     self.bneck = SpikeBottleneck(in_channels=in_channels,
#                                   hidden_channels=hidden_channels,
#                                   out_channels=out_channels,
#                                   kernel_size=kernel_size,
#                                   padding=padding)

#   def forward(self, x):
#     if self.downsample: x = self.conv(x)
#     return self.bneck(x)

class SpikeFormer(nn.Module):
  def __init__(self, embed_dim, mlp_ratio, heads):
    super().__init__()

    self.attn = SpikeAttention(embed_dim=embed_dim, heads=heads)

    self.mlp = SpikeMLP(in_channels=embed_dim,
                        hidden_channels=embed_dim*mlp_ratio)
  
  def forward(self, tokens):

    # Self Attention
    z = self.attn(tokens)
    tokens = z + tokens

    # Feed Forward Network
    z = self.mlp(tokens)
    tokens = z + tokens

    return tokens


class SpikeAttention(nn.Module):
  def __init__(self, embed_dim, heads):
    super().__init__()
    self.heads = heads

    self.qkv_proj = SpikeDense(in_channels=embed_dim,
                               out_channels=3*embed_dim)
    
    self.output = SpikeDense(in_channels=embed_dim,
                             out_channels=embed_dim)
    
    self.scale = nn.Parameter(torch.Tensor([0.25,]))
    
  def forward(self, tokens):
    
    # Initial Shape
    T, B, D, M = tokens.shape

    # Project Tokens
    qkv = self.qkv_proj(tokens) # T B 3D M

    # Multi-Headed Attention
    qkv = qkv.reshape(T, B, self.heads, 3*D//self.heads, M) # T B h 3d M
    q,k,v = qkv.transpose(3,4).chunk(3, axis=-1) # T B h M d

    # Matrix Multiplication
    q_k = torch.matmul(q, k.transpose(3,4)) # T B h M M
    # q_k_v = torch.matmul(q_k, v) # T B h M d
    q_k_v = torch.matmul(q_k, v) * self.scale # T B h M d

    # Concatanate Multi Heads
    tokens = q_k_v.transpose(3,4).reshape(tokens.shape) # T B D M

    # Linear Layer
    return self.output(tokens)


class SpikeMLP(nn.Module):
  def __init__(self, in_channels, hidden_channels):
    super().__init__()

    self.net = nn.Sequential(
      SpikeDense(in_channels=in_channels, out_channels=hidden_channels),
      SpikeDense(in_channels=hidden_channels, out_channels=in_channels),
    )

  def forward(self, tokens):
    return self.net(tokens)