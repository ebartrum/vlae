import torch
from torch import nn
from .wan_video_dit import WanModel, DiTBlock, modulate

class TracksAwareWanModel(WanModel):
    @classmethod
    def from_wan_dit(cls, dit: WanModel):
       dit.__class__ = cls

       #cast all of the blocks to a new type
       for blk in dit.blocks:
           blk = TracksAwareDiTBlock.from_wan_dit_block(blk)
       return dit

class TracksAwareDiTBlock(DiTBlock):
    #from wan dit for reference:
    # self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
    #     approximate='tanh'), nn.Linear(ffn_dim, dim))
    @classmethod
    def from_wan_dit_block(cls, blk: DiTBlock):
       blk.__class__ = cls
       blk.track_conditioning_layer1 = nn.Linear(5120,64)
       blk.track_conditioning_layer2 = nn.Linear(64,5120)
       blk.tracks_residual_parameter = torch.nn.Parameter(torch.zeros(1).to(torch.bfloat16))
       return blk

    def tracks_conditioning(self, x, tracks):
        out = self.track_conditioning_layer1(x.float())
        out = torch.relu(out)
        out = self.track_conditioning_layer2(out)
        return out.to(x.dtype)

    def forward(self, x, context, t_mod, freqs, tracks=None):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, freqs)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        tracks_conditioned_x = input_x + self.tracks_conditioning(x, tracks
            )*self.tracks_residual_parameter
        x = x + gate_mlp * self.ffn(tracks_conditioned_x)
        return x
