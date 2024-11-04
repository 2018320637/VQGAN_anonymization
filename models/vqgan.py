import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import LPIPS, Codebook, Codebook2D

def silu(x):
    return x*torch.sigmoid(x)

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.args = args
        self.embed_dim_static = args.embedding_dim_static
        self.embed_dim_dynamic = args.embedding_dim_dynamic
        self.embedding_dim = self.embed_dim_static + self.embed_dim_dynamic
        self.n_codes_static = args.n_codes_static
        self.n_codes_dynamic = args.n_codes_dynamic

        self.encoder = Encoder(args.n_hiddens, args.downsample, args.image_channels, args.norm_type, args.padding_type)
        self.decoder = Decoder(args.n_hiddens, args.downsample, args.image_channels, args.norm_type)
        self.enc_out_ch = self.encoder.out_channels

        self.pre_vq_conv_static = torch.nn.Conv2d(self.enc_out_ch, args.embedding_dim_static, 1)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.pre_vq_conv_dynamic = SamePadConv3d(self.enc_out_ch, args.embedding_dim_dynamic, 1, padding_type=args.padding_type)
        self.post_vq_conv = SamePadConv3d(self.embedding_dim, self.enc_out_ch, 1)
        
        self.codebook_dynamic = Codebook(args.n_codes_dynamic, args.embedding_dim_dynamic, no_random_restart=args.no_random_restart, restart_thres=args.restart_thres)
        self.codebook_static = Codebook2D(args.n_codes_static, args.embedding_dim_static, no_random_restart=args.no_random_restart, restart_thres=args.restart_thres)

        self.gan_feat_weight = args.gan_feat_weight
        self.image_discriminator = NLayerDiscriminator(args.image_channels, args.disc_channels, args.disc_layers)
        self.video_discriminator = NLayerDiscriminator3D(args.image_channels, args.disc_channels, args.disc_layers)
        
        if args.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif args.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.image_gan_weight = args.image_gan_weight
        self.video_gan_weight = args.video_gan_weight
        self.perceptual_weight = args.perceptual_weight
        self.l1_weight = args.l1_weight

    def forward(self, x, optimizer_idx=None, log_image=False):
        B, C, T, H, W = x.shape
        
        encoder_output = self.encoder(x)
        z_dynamic = self.pre_vq_conv_dynamic(encoder_output)
        e_bs, e_c, e_t, e_h, e_w = encoder_output.shape
        z_static = self.pre_vq_conv_static(encoder_output.permute(0, 2, 1, 3, 4).reshape(-1, e_c, e_h, e_w))
        z_static = z_static.reshape(e_bs, e_t, self.embed_dim_static, e_h, e_w).permute(0, 2, 1, 3, 4)
        z_static = self.temporal_pool(z_static).squeeze(2) # [b, c, h, w]
        vq_output_static = self.codebook_static(z_static)
        vq_output_dynamic = self.codebook_dynamic(z_dynamic)

        static_embeddings = vq_output_static['embeddings'].unsqueeze(2).expand(-1, -1, e_t, -1, -1)
        static_encodings = vq_output_static['encodings']
        dynamic_embeddings = vq_output_dynamic['embeddings']
        dynamic_encodings = vq_output_dynamic['encodings']
        vq_output = {
            'embeddings': torch.cat([static_embeddings, dynamic_embeddings], dim=1),
            'encodings_static': static_encodings,
            'encodings_dynamic': dynamic_encodings,
        }
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))

        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight

        frame_idx = torch.randint(0, T, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if optimizer_idx == 0:
            # autoencoder
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight

            logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon)
            logits_video_fake, pred_video_fake = self.video_discriminator(x_recon)
            g_image_loss = -torch.mean(logits_image_fake)
            g_video_loss = -torch.mean(logits_video_fake)
            g_loss = self.image_gan_weight*g_image_loss + self.video_gan_weight*g_video_loss
            
            aeloss = g_loss

            # gan feature matching loss
            image_gan_feat_loss = 0
            video_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.image_gan_weight > 0:
                logits_image_real, pred_image_real = self.image_discriminator(frames)
                for i in range(len(pred_image_fake)-1):
                    image_gan_feat_loss += feat_weights * F.l1_loss(pred_image_fake[i], pred_image_real[i].detach()) * (self.image_gan_weight>0)
            if self.video_gan_weight > 0:
                logits_video_real, pred_video_real = self.video_discriminator(x)
                for i in range(len(pred_video_fake)-1):
                    video_gan_feat_loss += feat_weights * F.l1_loss(pred_video_fake[i], pred_video_real[i].detach()) * (self.video_gan_weight>0)

            gan_feat_loss = self.gan_feat_weight * (image_gan_feat_loss + video_gan_feat_loss)
            commitment_loss_static = vq_output_static['commitment_loss']
            commitment_loss_dynamic = vq_output_dynamic['commitment_loss']
            return recon_loss, commitment_loss_static, commitment_loss_dynamic, aeloss, perceptual_loss, gan_feat_loss

        elif optimizer_idx == 1:
            # discriminator
            logits_image_real, _ = self.image_discriminator(frames.detach())
            logits_video_real, _ = self.video_discriminator(x.detach())

            logits_image_fake, _ = self.image_discriminator(frames_recon.detach())
            logits_video_fake, _ = self.video_discriminator(x_recon.detach())

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
            discloss = self.image_gan_weight*d_image_loss + self.video_gan_weight*d_video_loss

            return discloss
        
        elif optimizer_idx == 2:

            return vq_output_static['embeddings'], vq_output_dynamic['embeddings']
        
        else:
            perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight
            return recon_loss, x_recon, vq_output_static, vq_output_dynamic, perceptual_loss



def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=3, norm_type='group', padding_type='replicate'):
        super(Encoder, self).__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()

        self.conv_first = SamePadConv3d(image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        for i in range(max_ds):
            block = nn.Module()
            in_channels = n_hiddens * 2**i
            out_channels = n_hiddens * 2**(i+1)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            block.down = SamePadConv3d(in_channels, out_channels, 4, stride=stride, padding_type=padding_type)
            block.res = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type), 
            SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group'):
        super(Decoder, self).__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        
        in_channels = n_hiddens*2**max_us
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type),
            SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i ==0 else n_hiddens*2**(max_us-i+1)
            out_channels = n_hiddens*2**(max_us-i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose3d(in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            block.res2 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_last = SamePadConv3d(out_channels, image_channel, kernel_size=3)
    def forward(self, x):
        h = self.final_block(x)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='replicate'):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type)
        self.conv2 = SamePadConv3d(out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h

# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super(SamePadConv3d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super(SamePadConvTranspose3d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
    # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), None


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), None