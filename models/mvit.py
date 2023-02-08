from pytorchvideo.models.hub import mvit_base_16x4
import torch
import torch.nn as nn
import torch.nn.functional as F
from retrieval_head import Head

# model = mvit_base_16x4(True)
# # for i, b in enumerate(model.blocks):
# #     print(i)
# #     print(b)

# print(1)
# print(model.norm_embed)
# print(2)
# print(model.head)
# B, C, T, H, W = 2, 3, 16, 224, 224
# input_tensor = torch.zeros(B, C, T, H, W)

# output = model(input_tensor)



class MViT(nn.Module):
	def __init__(self, num_ego_class, num_actor_class):
		super(MViT, self).__init__()
		self.model = mvit_base_16x4(True)
		self.cls = Head(768, num_ego_class, num_actor_class)
		self.head = nn.Sequential(
								nn.Dropout(p=0.5, inplace=False),
								# nn.Linear(in_features=2304, out_features=400, bias=True),
								self.cls,
								)
		# for i, b in enumerate(self.model.blocks):
		# 	print(i)
		# 	print(b)
        # self.pool = nn.AdaptiveAvgPool3d(output_size=1)
	def forward(self, x):
		seq_len = len(x)
		batch_size = x[0].shape[0]
		height, width = x[0].shape[2], x[0].shape[3]
		if isinstance(x, list):
			x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
			# l, b, c, h, w
			x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
		num_block = len(self.model.blocks)
		# print(x.shape)

		x = self.model.patch_embed(x)
		x = self.model.cls_positional_encoding(x)
		x = self.model.pos_drop(x)

		thw = self.model.cls_positional_encoding.patch_embed_shape()
		for blk in self.model.blocks:
		    x, thw = blk(x, thw)
		x = self.model.norm_embed(x)

		# https://github.com/facebookresearch/pytorchvideo/blob/702f9f42569598c5cce8c5e2dd7e37c3d6c46efd/pytorchvideo/models/head.py#L11
		x = x[:, 0]
		# x = torch.reshape(x, (batch_size, 768))
		ego, x = self.head(x)

		return ego, x