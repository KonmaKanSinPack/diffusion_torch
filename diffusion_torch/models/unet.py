import torch
import torch.nn.functional as F
import torch.nn as nn


def nonlinearity():
  return nn.SiLU()


# def normalize(num_groups=32, num_channels=32):
#   return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
def normalize(x, num_groups=32):
  num_channels = x.shape[1]
  num_groups = min(num_groups, num_channels)
  while num_channels % num_groups != 0:
    num_groups -= 1
  return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels).to(x.device)(x)


def upsampleBlock(in_channel=None, out_channel=None, with_conv=False):
  if not with_conv:
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'))
  else:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channel, out_channel, 3, padding=1)
    )

def downsampleBlock(in_channel=None, out_channel=None, with_conv=False):
  if not with_conv:
    return nn.Sequential(nn.AvgPool2d(2))
  else:
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, 4, stride=2, padding=1))
  
class resnetBlock(nn.Module):
  def __init__(self, in_ch, out_ch, conv_shortcut=False, dropout=0.8, temb_ch=512, **kwarg):
    super().__init__()
    
    self.in_ch = in_ch
    self.out_ch = out_ch
    
    self.act1 = nn.SiLU()
    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    
    self.act_temb = nn.SiLU()
    self.dense_temb = nn.Linear(temb_ch, out_ch)
    
    self.act2 = nn.SiLU()
    self.dropout = nn.Dropout(dropout)
    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    self.adapt = None
    if in_ch != out_ch:
      if conv_shortcut:
        self.adapt = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
      else:
        self.adapt = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

  def forward(self, x, temb):
    # normalize
    h = normalize(x)  
    h = self.act1(h)
    h = self.conv1(h)
    
    # time embed
    t = self.act_temb(temb)
    t = self.dense_temb(t)[:, :, None, None] 
    h = h + t
    
    h = normalize(h)
    h = self.act2(h)
    h = self.dropout(h)
    h = self.conv2(h)
    
    if self.adapt is not None:
      x = self.adapt(x)
    return x + h


class attn_block(nn.Module):
  def __init__(self, num_channels):
    super().__init__()
    self.q = nn.Conv2d(num_channels,num_channels,1)
    self.k = nn.Conv2d(num_channels,num_channels,1)
    self.v = nn.Conv2d(num_channels,num_channels,1)
    self.proj_out = nn.Conv2d(num_channels,num_channels,1)

  def forward(self, x, temb):
    B, C, H, W = x.shape
  
    h = normalize(x)

    q = self.q(h)
    k = self.k(h)
    v = self.v(h)

    # BCHW -> BHWC
    q = q.permute(0, 2, 3, 1)  # BCHW -> BHWC
    k = k.permute(0, 2, 3, 1)  # BCHW -> BHWC
    v = v.permute(0, 2, 3, 1)  # BCHW -> BHWC

    w = torch.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, [B, H, W, H * W])
    w = torch.nn.functional.softmax(w, -1)
    w = torch.reshape(w, [B, H, W, H, W])

    h = torch.einsum('bhwHW,bHWc->bhwc', w, v)
    h = h.permute(0, 3, 1, 2)  # BHWC -> BCHW
    h = self.proj_out(h)

    assert h.shape == x.shape
    # print(tf.get_default_graph().get_name_scope(), x.shape)
    return x + h


class UNet(nn.Module):
  def __init__(
      self,
      t_emb_dim,
      ch,
      out_ch,
      ch_mult=(1, 2, 2, 2),
      num_res_blocks=2,
      attn_resolutions=(16,),
      dropout=0.1,
      resamp_with_conv=True,
      in_ch=3,
  ):
    super().__init__()

    # Time embedding layers
    self.t_emb_dim = t_emb_dim
    self.tmb_layers = nn.Sequential(
        nn.Linear(t_emb_dim, t_emb_dim * 4),
        nonlinearity(),
        nn.Linear(t_emb_dim * 4, t_emb_dim * 4)
    )
    self.with_conv = resamp_with_conv

    # Downsampling layers
    self.in_conv = nn.Conv2d(in_ch, ch, 3, padding=1)
    downs = []
    num_muls = len(ch_mult)
    current_ch = ch
    hs_channels = [current_ch]  
    
    for i in range(num_muls):
        target_ch = ch * ch_mult[i]  
        
        for block_idx in range(num_res_blocks):
            downs.append(resnetBlock(in_ch=current_ch, out_ch=target_ch, dropout=dropout, temb_ch=t_emb_dim * 4))
            current_ch = target_ch
            hs_channels.append(current_ch)
        
        if (2 ** i) in attn_resolutions:
            downs.append(attn_block(num_channels=current_ch))
            hs_channels.append(current_ch)
        
        if i != num_muls - 1:
            downs.append(downsampleBlock(in_channel=current_ch, out_channel=current_ch, with_conv=resamp_with_conv))
            hs_channels.append(current_ch)

    self.down_layers = nn.ModuleList(downs)

    # Middle layers
    self.middle_layers = nn.ModuleList([
        resnetBlock(in_ch=current_ch, out_ch=current_ch, dropout=dropout, temb_ch=t_emb_dim * 4),
        attn_block(num_channels=current_ch),
        resnetBlock(in_ch=current_ch, out_ch=current_ch, dropout=dropout, temb_ch=t_emb_dim * 4)
    ])

    # Upsampling layers
    ups = []
    skip_ch_stack = list(hs_channels)
    
    for i in reversed(range(num_muls)):
        target_ch = ch * ch_mult[i] 
        
        for _ in range(num_res_blocks + 1):
            skip_ch = skip_ch_stack.pop()
            ups.append(resnetBlock(in_ch=current_ch + skip_ch, out_ch=target_ch, dropout=dropout, temb_ch=t_emb_dim * 4))
            current_ch = target_ch
        
        if (2 ** i) in attn_resolutions:
            ups.append(attn_block(num_channels=current_ch))
        
        if i != 0:
            ups.append(upsampleBlock(in_channel=current_ch, out_channel=current_ch, with_conv=resamp_with_conv))

    self.up_layers = nn.ModuleList(ups)

    # End
    self.out_conv = nn.Conv2d(current_ch, out_ch, 3, padding=1)
    self.out_ch = out_ch


  def forward(self, x, t):
    B, C, H, W = x.shape

    # Timestep embedding
    temb = self.get_timestep_embedding(t, self.t_emb_dim)
    temb = self.tmb_layers(temb)

    # Downsampling
    hs = [self.in_conv(x)]
    for block in self.down_layers:
        if isinstance(block, (resnetBlock, attn_block)):
            h = block(hs[-1], temb=temb)
        else:
            h = block(hs[-1])
        hs.append(h)
      
    # Middle
    h = hs[-1]
    for block in self.middle_layers:
        h = block(h, temb=temb)

    # Upsampling
    for block in self.up_layers:
        if isinstance(block, resnetBlock):
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, temb=temb)
        elif isinstance(block, attn_block):
            h = block(h, temb=temb)
        else:
            h = block(h)
    
    # End
    h = normalize(h)
    h = nonlinearity()(h)
    h = self.out_conv(h)

    assert h.shape == (B, self.out_ch, H, W), f"h.shape: {h.shape}, expected: {(B, self.out_ch, H, W)}"
    return h
  
  def get_timestep_embedding(self, timesteps, embedding_dim):
    # sin positional encoding
    assert len(timesteps.shape) == 1
    
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    
    return emb

# class model(nn.Module):
#   def __init__(self, *args, **kwargs):
#     super().__init__(*args, **kwargs)
