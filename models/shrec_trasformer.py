import torch
import torch.nn as nn
from models.attention import EncoderSelfAttention

class ShrecTrasformer(nn.Module):
    """Multi Modal model for gesture recognition on 3 channel"""
    def __init__(self, in_planes: int, out_planes: int, **kwargs):
        super(ShrecTrasformer, self).__init__()

        self.in_planes = in_planes
        self.self_attention = EncoderSelfAttention(self.in_planes, 64, 64, **kwargs)
        self.classifier = nn.Linear(self.in_planes, out_planes)


    def forward(self, x):
        shape = x.shape
        x = x.view(shape[0], shape[1], -1)
        x = self.self_attention(x)
        x = self.classifier(x)
        return x