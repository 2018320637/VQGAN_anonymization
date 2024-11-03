import torch
from .valid_models import valid_models
import torch.nn as nn
from torchvision.models import resnet50, vgg16, resnet18, resnet101
from torchvision.models.vision_transformer import vit_b_16, vit_b_32
from .r3d_classifier import r2plus1d_18
from torch.autograd import Function

# This has all the strings of the torch hub video models.
# https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/README.md
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class PrivacyClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, num_domains=2):
        super(PrivacyClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
        self.domain_fc = nn.Linear(feature_dim, num_domains)
        
    def forward(self, x, alpha=0.5):
        class_output = self.fc(x)
        reverse_feature = ReverseLayerF.apply(x, alpha)
        domain_output = self.domain_fc(reverse_feature)
        return class_output, domain_output

def set_requires_grad(model, flag: bool):
    for param in model.parameters():
        param.requires_grad = flag

def build_model_privacy(architecture, pretrained, num_classes, train_backbone):
    if architecture == "resnet101":
        model = resnet101(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.fc = nn.Linear(2048, num_classes)

    elif architecture == "resnet50":
        model = resnet50(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.fc = nn.Linear(2048, num_classes)

    elif architecture == "resnet18":
        model = resnet18(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.fc = nn.Linear(512, num_classes)

    elif architecture == "vgg16":
        model = vgg16(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif architecture == "vit":
        model = vit_b_16(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.heads.head = nn.Linear(768, num_classes)

    elif architecture == "vit_b_32":
        model = vit_b_32(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.heads.head = nn.Linear(768, num_classes)
    else:
        raise ValueError("unsupported architecture")

    return model

def build_model_privacy_domain(architecture, pretrained, num_classes, train_backbone):
    if architecture == "resnet101":
        model = resnet101(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.fc = nn.Identity()
        privacy_model = PrivacyClassifier(2048, num_classes)

    elif architecture == "resnet50":
        model = resnet50(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.domain_fc = nn.Linear(2048, 2)
        model.fc = nn.Identity()
        privacy_model = PrivacyClassifier(2048, num_classes)

    elif architecture == "resnet18":
        model = resnet18(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.fc = nn.Identity()
        privacy_model = PrivacyClassifier(512, num_classes)

    elif architecture == "vgg16":
        model = vgg16(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.classifier[6] = nn.Identity()
        privacy_model = PrivacyClassifier(4096, num_classes)
    elif architecture == "vit":
        model = vit_b_16(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.heads.head = nn.Identity()
        privacy_model = PrivacyClassifier(768, num_classes)

    elif architecture == "vit_b_32":
        model = vit_b_32(pretrained=pretrained)
        set_requires_grad(model, train_backbone)
        model.heads.head = nn.Identity()
        privacy_model = PrivacyClassifier(768, num_classes)
    else:
        raise ValueError("unsupported architecture")

    return model, privacy_model

def build_model(architecture, pretrained, num_classes, train_backbone):
    if architecture == "fast_r50":
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", "slowfast_r50", pretrained=pretrained
        )
        set_requires_grad(model, train_backbone)
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_features, num_classes)
        model.blocks[-1].activation = nn.Identity()
        print(model)
        # and then route the slow output into the nirvana.
        print("printing trainable parameters")

    elif architecture == "mvit_base_16x4":
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", architecture, pretrained=pretrained
        )
        set_requires_grad(model, train_backbone)
        in_features = model.head.proj.in_features
        model.head.proj = nn.Linear(in_features, num_classes)
        print("done")

    elif "e2s_x3d" in architecture:
        model = E2SX3D(
            rgb_arch=architecture,
            flow_arch=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            train_rgbbackbone=train_backbone,
            train_flowbackbone=train_backbone,
        )
    elif "r2plus1d_r18" in architecture:
        model = r2plus1d_18(pretrained = True, progress = False)
        set_requires_grad(model, train_backbone)
        model.fc = nn.Linear(512, num_classes)

    elif architecture in valid_models.keys():
        # model = torch.hub.load(
        #     "facebookresearch/pytorchvideo", architecture, pretrained=pretrained
        # )
        model = torch.hub.load(
            "/home/zhiwei/.cache/torch/hub/facebookresearch_pytorchvideo_main", architecture, 
            pretrained= True, trust_repo= True, source= 'local'
        )
        in_features = model.blocks[-1].proj.in_features
        set_requires_grad(model, train_backbone)
        model.blocks[-1].proj = nn.Linear(in_features, num_classes)
        model.blocks[-1].activation = nn.Identity()
    else:
        raise ValueError("Unknown architecture.")
    return model


class E2SX3D(nn.Module):
    def init_backbone(self, arch, num_classes, pretrained, train_backbone):
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            arch.replace("e2s_", ""),
            pretrained=pretrained,
        )
        in_features = model.blocks[-1].proj.in_features
        set_requires_grad(model, train_backbone)
        model.blocks[-1].proj = nn.Linear(in_features, num_classes)
        model.blocks[-1].activation = nn.Identity()
        return model

    def __init__(
        self,
        rgb_arch,
        flow_arch,
        num_classes,
        pretrained,
        train_rgbbackbone,
        train_flowbackbone,
    ):
        super(E2SX3D, self).__init__()
        self.rgb_arch = rgb_arch
        self.flow_arch = flow_arch
        self.rgbstream = self.init_backbone(
            rgb_arch, num_classes, pretrained, train_rgbbackbone
        )
        self.flowstream = self.init_backbone(
            flow_arch, num_classes, pretrained, train_flowbackbone
        )
        self.head = nn.Linear(2 * num_classes, num_classes)

    def forward(self, rgbs, flows):
        fs = self.flowstream(flows)
        aps = self.rgbstream(rgbs)
        x = torch.cat([fs, aps], dim=1)
        x = self.head(x)
        return x