from torch.autograd import Function
import torch.nn as nn

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

# class ShuffleDiscriminator(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(ShuffleDiscriminator, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)
#         self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        
#     def forward(self, x, type='static'):
#         if type == 'static':
#             # Flatten: BCHW -> B,C*H*W
#             x = x.view(x.size(0), -1)
#         elif type == 'dynamic':  # 这种版本尝试使用dynamic的特征作为负样本，来计算triplet loss，但是这种对比损失的设计就会很奇怪
#             # Flatten: BCTHW -> B,C*T*H*W
#             x = self.temporal_pool(x).squeeze(2) # [b, c, h, w]
#             x = x.view(x.size(0), -1)
#         output = self.fc(x)
#         return output

class ShuffleDiscriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ShuffleDiscriminator, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # Flatten: BCHW -> B,C*H*W
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output
