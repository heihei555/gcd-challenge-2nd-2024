import os
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from models import vision_transformer as vits

class DINO(nn.Module):
    def __init__(self, args):
        super(DINO, self).__init__()
        self.k = args.k
        self.backbone = self.init_backbone()
        self.img_projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim, out_dim=args.feat_dim, nlayers=args.num_mlp_layers)

    def forward(self, image):
        feat = self.backbone(image)
        feat = self.img_projection_head(feat)
        feat = F.normalize(feat, dim=-1)
            
        return feat

    def init_backbone(self):
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        for m in model.parameters():
            m.requires_grad = False

        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= 23:
                    m.requires_grad = True

        return model