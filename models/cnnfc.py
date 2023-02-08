import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from MaskFormer.demo.demo import get_maskformer
from retrieval_head import Head



class CNNFC(nn.Module):
    def __init__(self, num_ego_class, num_actor_class, road, num_seq=12, use_backbone=False):
        super(CNNFC, self).__init__()
        self.use_backbone = use_backbone
        if self.use_backbone:
            self.backbone = get_maskformer()
            self.in_channel = 3
        else:
            self.in_channel = 2048
        self.road = road
        # self.seq_len = num_seq
        # print(self.backbone.backbone.res5)
        self.conv1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                    )

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        self.head = Head(num_seq*256, num_ego_class, num_actor_class)



    def forward(self, x):

        seq_len = len(x)
        batch_size = x[0].shape[0]
        height, width = x[0].shape[2], x[0].shape[3]

        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
            x = torch.permute(x, (1,0,2,3,4)) #[b, v, 2048, h, w]
            x = torch.reshape(x, (batch_size*seq_len, self.in_channel, height, width)) #[b, 2048, h, w]

        x = self.backbone.backbone(x)['res5']
        x = self.conv1(x) #[b, 256, h, w]
        height, width = x.shape[2], x.shape[3]
        x = self.pool(x)
        x = torch.reshape(x, (batch_size, seq_len, 256))
        x = torch.reshape(x, (batch_size, seq_len*256))
        x = self.dropout(x)
        ego, actor = self.head(x)

        return ego, actor
