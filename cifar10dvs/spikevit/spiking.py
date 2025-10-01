import torch
from torch import nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode as LIF


class SpikeDense(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
    self.bn = nn.BatchNorm1d(out_channels)
    self.lif = LIF(tau=2.0, detach_reset=True, backend='cupy')

  def forward(self, x):
    T, B, _, M, = x.shape
    x = self.conv(x.flatten(0,1))
    x = self.bn(x).reshape(T, B, -1, M)
    return self.lif(x.contiguous())
  

class SpikeConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    super().__init__()

    self.conv = nn.Conv2d(in_channels, out_channels, 
                          kernel_size=kernel_size, stride=stride, 
                          padding=padding, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.lif = LIF(tau=2.0, detach_reset=True, backend='cupy')

  def forward(self, x):
    T, B, _, _, _ = x.shape
    x = self.conv(x.flatten(0,1))
    _, C, H, W = x.shape
    x = self.bn(x).reshape(T, B, C, H, W)
    return self.lif(x.contiguous())


class SpikeBottleneck(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, padding=1):
    super().__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(hidden_channels),
      nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
      nn.BatchNorm2d(hidden_channels),
      nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
    )
    self.bn = nn.BatchNorm2d(out_channels)
    self.lif = LIF(tau=2.0, detach_reset=True, backend='cupy')

  def forward(self, x):
    T, B, _, _, _ = x.shape
    x = self.conv(x.flatten(0,1))
    _, C, H, W = x.shape
    x = self.bn(x).reshape(T, B, C, H, W)
    return self.lif(x.contiguous())