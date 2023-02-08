import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from retrieval_head import Head
from pytorchvideo.models.hub import i3d_r50
import numpy as np
# from models.ConvGRU import *
from math import ceil 

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim)).cuda()
        self.slots_sigma = torch.randn(1, 1, dim).cuda()
        self.slots_sigma = nn.Parameter(self.slots_sigma.absolute())

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        # self.tem_gru = nn.GRUCell(dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        mu = self.slots_mu.expand(1, self.num_slots, -1)
        sigma = self.slots_sigma.expand(1, self.num_slots, -1)
        slots = torch.normal(mu, sigma)

        slots = slots.contiguous()
        self.register_buffer("slots", slots)

    def get_attention(self, slots, inputs):
        slots_prev = slots
        b, n, d = inputs.shape
        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)
        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        # print(dots.shape)
        attn_ori = dots.softmax(dim=1) + self.eps
        attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)
        slots = self.gru(
            updates.reshape(-1, d),
            slots_prev.reshape(-1, d)
        )

        slots = slots.reshape(b, -1, d)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn_ori

    def forward(self, inputs, num_slots = None):
        b, nf,  n, d = inputs.shape
        slots_out = []
        attns = []
        slots = self.slots.expand(b,-1,-1)
        # pre-attention for the first frame
        slots, _ = self.get_attention(slots, inputs[:,0,:,:])
        # cur_slots = slots
        for f in range(nf):
            # pre_slots = cur_slots
            # for i in range(3):
            cur_slots, cur_attn = self.get_attention(slots,inputs[:,f,:,:])
                # cur_slots, cur_attn = self.get_attention(cur_slots,inputs[:,f,:,:])
            # slots_out.append(cur_slots)
            # cur_slots = self.tem_gru(cur_slots.reshape(-1, d),pre_slots.reshape(-1, d))
            # cur_slots = cur_slots.reshape(b, -1, d)
            slots_out.append(cur_slots)
            attns.append(cur_attn)
            slots = cur_slots
        slots_out = torch.stack([slot for slot in slots_out])
        slots_out = slots_out.permute(1,0,2,3)
        attns = torch.stack([attn for attn in attns])
        attns = attns.permute(1,0,2,3)
        return slots_out, attns


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.register_buffer("grid", build_grid(resolution))
    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


class SLOT_VIDEO(nn.Module):
    def __init__(self, args, num_ego_class, num_actor_class, num_slots=21):
        super(SLOT_VIDEO, self).__init__()
        self.hidden_dim = 512
        self.hidden_dim2 = 256
        # self.slot_hidden_dim = 512
        self.resnet = i3d_r50(True)
        # self.resnet = self.resnet.blocks[:2]
        if args.backbone == 'i3d-2':
            self.resnet = self.resnet.blocks[:-2]
            self.resolution = (16, 48)
            self.in_c = 1024
        elif args.backbone == 'i3d-1':
            self.resnet = self.resnet.blocks[:-1]
            self.in_c = 2048
            self.resolution = (8, 24)
        self.head = Head(self.hidden_dim2, num_ego_class, num_actor_class)
        # self.model.blocks[-1] = nn.Sequential(
        #                         nn.Dropout(p=0.5, inplace=False),
        #                         # nn.Linear(in_features=2304, out_features=400, bias=True),
        #                         self.head,
        #                         )
        # 64 192
        self.encoder_pos = SoftPositionEmbed(self.in_c, self.resolution)
        self.fc1 = nn.Linear(self.in_c, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim2)
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=self.hidden_dim2,
            eps = 1e-8, 
            hidden_dim = self.hidden_dim2) 
        self.LN = nn.LayerNorm([self.resolution[0]*self.resolution[1], self.in_c]) 
        self.drop = nn.Dropout(p=0.5)         

    def forward(self, x):
        seq_len = len(x)
        batch_size = x[0].shape[0]
        height, width = x[0].shape[2], x[0].shape[3]
        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
            # l, b, c, h, w
            x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
        # num_block = len(self.resnet.blocks)
        # [bs, c, n, w, h]
        for i in range(len(self.resnet)):
            # x = self.resnet.blocks[i](x)
            x = self.resnet[i](x)
        new_seq_len = x.shape[2]
        new_h, new_w = x.shape[3], x.shape[4]
        # [bs, c, n, w, h]

        x = torch.permute(x, (0, 2, 3, 4, 1))
        # [bs, n, w, h, c]
        x = torch.reshape(x, (batch_size*new_seq_len, new_h, new_w, -1))
        # [bs*n, h, w, c]
        x = self.encoder_pos(x)
        # [bs*n, h*w, c]
        x = torch.flatten(x, 1, 2)
        x = self.LN(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  
        # [bs, n, h*w, c]
        x = x.view(batch_size, new_seq_len, -1, self.hidden_dim2)
        # print(x.shape)
        x, attn_masks = self.slot_attention(x)
        x = torch.sum(x, 1)



        # ego, x = self.head(slots_ori)
        x = self.drop(x)
        x = self.head(x)

        return x
        # return ego, x
