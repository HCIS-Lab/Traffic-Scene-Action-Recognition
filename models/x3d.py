import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from retrieval_head import Head

class X3D(nn.Module):
    def __init__(self, num_ego_class, num_actor_class):
        super(X3D, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
        # for i, b in enumerate(self.model.blocks):
        #     print(i)
        #     print(b)
        self.projection = nn.Sequential(
                nn.Conv3d(192, 432, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
                nn.BatchNorm3d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.AvgPool3d(kernel_size=(16, 3, 3), stride=1, padding=0),
                nn.Conv3d(432, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                    )

        self.model.blocks[-1] = self.projection
        # # b, n, c, h, w
        # # b, c, h, w, n
        # # b, n, c, h, w
        # x = x.permute((0, 2, 3, 4, 1))
        # x = self.proj(x)
        # x = x.permute((0, 4, 1, 2, 3))

        self.head = Head(2048, num_ego_class, num_actor_class)
        self.model.blocks.append(self.head)
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)

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
        for i in range(num_block-1):
            x = self.model.blocks[i](x)
            # print(x.shape)
        x = self.pool(x)
        x = torch.reshape(x, (batch_size, 2048))
        ego, x = self.model.blocks[-1](x)
        
        return ego, x
