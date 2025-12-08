import torch
import torch.nn.functional as F
import torch.nn as nn


def nonlinearity():
  return nn.SiLU()


def normalize(num_groups=32, num_channels=32):
  return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)



def upsampleBlock(in_channel=None,out_channel=None,with_conv=False):
  if not with_conv:
    return nn.ModuleList([nn.Upsample(scale_factor=2, mode='nearest')])
  else:
    return nn.ModuleList([nn.Upsample(scale_factor=2, mode='nearest')],nn.Conv2d(in_channel,out_channel,3,padding=1))


def downsampleBlock(in_channel=None,out_channel=None,with_conv=False):
  if not with_conv:
    return nn.ModuleList([nn.AvgPool2d(2)])
  else:
    return nn.ModuleList([nn.Conv2d(in_channel,out_channel,4,stride=2,padding=1)])

class resnetBlock(nn.Module):
  def __init__(self, in_ch,out_ch, conv_shortcut=False, dropout=0.8,**kwarg):
    super().__init__(in_ch, out_ch, conv_shortcut=False, dropout=0.8,**kwarg)
    
    self.conv1 = nn.ModuleList([normalize(num_channels=out_ch),nonlinearity(),nn.Conv2d(in_ch,out_ch,3,padding=1)])
    self.tmb_res = nn.ModuleList([nonlinearity(num_channels=out_ch),nn.Linear(out_ch,out_ch)])
    self.conv2 = nn.ModuleList([nonlinearity(num_channels=out_ch),nonlinearity(),nn.Dropout(dropout),nn.Conv2d(out_ch,out_ch,3,padding=1)])

    if in_ch != out_ch:
      if conv_shortcut:
        self.adapt = nn.Conv2d(in_ch, out_ch, 3,1,1)
      else:
        self.adapt = nn.Conv2d(in_ch, out_ch, 1,1,0)

  def forward(self, x, temb):
    h = self.conv1(x)
    t = self.tmb_res(temb)
    h = h + t
    h = self.conv2(h)
    if self.adapt is not None:
      x = self.adapt(x)
    return x + h


class attn_block(nn.Module):
  def __init__(self, num_channels):
    super().__init__(num_channels)
    self.norm = normalize(num_channels=num_channels)
    self.q = nn.Conv2d(num_channels,num_channels,1)
    self.k = nn.Conv2d(num_channels,num_channels,1)
    self.v = nn.Conv2d(num_channels,num_channels,1)
    self.proj_out = nn.Conv2d(num_channels,num_channels,1)

  def forward(self, x, temb):
    B, H, W, C = x.shape
  
    h = self.norm(x)
    q = self.q(h)
    k = self.k(h)
    v = self.v(h)

    w = torch.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, [B, H, W, H * W])
    w = torch.nn.functional.softmax(w, -1)
    w = torch.reshape(w, [B, H, W, H, W])

    h = torch.einsum('bhwHW,bHWc->bhwc', w, v)
    h = self.proj_out(h)

    assert h.shape == x.shape
    # print(tf.get_default_graph().get_name_scope(), x.shape)
    return x + h


class model(nn.Module):
  def __init__(self,t_emb_dim,ch,out_ch,ch_mult=(1, 2, 2, 2), num_res_blocks=2, attn_resolutions=(16,),
          dropout=0.8, resamp_with_conv=True):
    super().__init__()
    #Time embedding layers
    self.tmb_layers = nn.ModuleList([nn.Linear(t_emb_dim, t_emb_dim * 4),nonlinearity(),nn.Linear(t_emb_dim * 4, t_emb_dim * 4)])

    #Downsampling layers
    self.in_conv = nn.Conv2d(3, ch, 3, padding=1)
    downs = []
    num_muls = len(ch_mult)
    for i in range(num_muls):
      for _ in range(num_res_blocks):
        downs.append(resnetBlock(in_ch=ch, out_ch=ch * ch_mult[i], dropout=dropout))
          ch = ch * ch_mult[i]
      if (2 ** i) in attn_resolutions:
        downs.append(attn_block(num_channels=ch))
      if i != num_muls - 1:
        downs.append(downsampleBlock(in_channel=ch,out_channel=ch,with_conv=resamp_with_conv))

    self.down_layers = nn.ModuleList(downs)

    #Middle layers
    self.middle_layers = nn.ModuleList([
      resnetBlock(in_ch=ch, out_ch=ch, dropout=dropout),
      attn_block(num_channels=ch),
      resnetBlock(in_ch=ch, out_ch=ch, dropout=dropout)
    ])

    #Upsampling layers
    ups = []
    for i in reversed(range(num_muls)):
      for _ in range(num_res_blocks + 1):
        ups.append(resnetBlock(in_ch=ch * 2, out_ch=ch // ch_mult[i], dropout=dropout))
        ch = ch // ch_mult[i]
      if (2 ** i) in attn_resolutions:
        ups.append(attn_block(num_channels=ch))
      if i != 0:
        ups.append(upsampleBlock(in_channel=ch,out_channel=ch,with_conv=resamp_with_conv))

  def forward(self, x, *, t, num_classes, reuse=tf.AUTO_REUSE):
    B, S, _, _ = x.shape
    # assert x.dtype == tf.float32 and x.shape[2] == S
    # assert t.dtype in [tf.int32, tf.int64]

    # Timestep embedding
    temb = utils.get_timestep_embedding(t, self.t_emb_dim)
    temb = self.tmb_layers(temb)

    # Downsampling
    hs = [nn.conv2d(x, name='conv_in', num_units=ch)]
    for block in self.down_layers:
        h  = block(hs[-1], temb=temb)
        hs.append(h)
      
    # Middle
    for block in self.middle_layers:
        h = block(h, temb=temb)

    # Upsampling
    for block in self.up_layers:
      h = block(tf.concat([h, hs.pop()], axis=-1), temb=temb)

    # End
    h = nonlinearity(normalize(h, temb=temb, name='norm_out'))
    h = nn.conv2d(h, name='conv_out', num_units=out_ch, init_scale=0.)
    assert h.shape == x.shape[:3] + [out_ch]
    return h

class model(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
