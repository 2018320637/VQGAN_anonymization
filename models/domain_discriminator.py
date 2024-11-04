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

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, domain_num):
        super(DomainDiscriminator, self).__init__()
        self.domain_fc = nn.Linear(input_dim, domain_num)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        
    def forward(self, x, alpha=0.5):
        # Pooling: B,C,T,H,W -> B,C,H,W
        x = self.temporal_pool(x).squeeze(2) # [b, c, h, w]
        # Flatten: B,C,H,W -> B,C*H*W
        x = x.view(x.size(0), -1)
        reverse_feature = ReverseLayerF.apply(x, alpha)
        output = self.domain_fc(reverse_feature)
        return output
