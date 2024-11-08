# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import imageio

import math
import numpy as np
import skvideo.io

import sys
import pdb as pdb_original

class ForkedPdb(pdb_original.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb_original.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin



# Shifts src_tf dim to dest dim
# i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)
def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


def adopt_weight(global_step, threshold=0, value=0.):
    weight = 1
    if global_step < threshold:
        weight = value
    return weight


def save_video_grid(video, fname, nrow=None, fps=6):
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1)
    video = (video.cpu().numpy() * 255).astype('uint8')
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * nrow + padding,
                           (padding + w) * ncol + padding, c), dtype='uint8')
    # print(video_grid.shape)
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    video = []
    for i in range(t):
        video.append(video_grid[i])
    imageio.mimsave(fname, video, fps=fps)
    # skvideo.io.vwrite(fname, video_grid, inputdict={'-r': '5'})
    print('saved videos to', fname)


def comp_getattr(args, attr_name, default=None):
    if hasattr(args, attr_name):
        return getattr(args, attr_name)
    else:
        return default


def visualize_tensors(t, name=None, nest=0):
    if name is not None:
        print(name, "current nest: ", nest)
    print("type: ", type(t))
    if 'dict' in str(type(t)):
        print(t.keys())
        for k in t.keys():
            if t[k] is None:
                print(k, "None")
            else:
                if 'Tensor' in str(type(t[k])):
                    print(k, t[k].shape)
                elif 'dict' in str(type(t[k])):
                    print(k, 'dict')
                    visualize_tensors(t[k], name, nest + 1)
                elif 'list' in str(type(t[k])):
                    print(k, len(t[k]))
                    visualize_tensors(t[k], name, nest + 1)
    elif 'list' in str(type(t)):
        print("list length: ", len(t))
        for t2 in t:
            visualize_tensors(t2, name, nest + 1)
    elif 'Tensor' in str(type(t)):
        print(t.shape)
    else:
        print(t)
    return ""


import torch.nn as nn
from torch.autograd import Variable
import shutil
import torch.nn.functional as F
from PIL import Image, ImageDraw
import torch
import socket
import numpy as np
import scipy.misc

hostname = socket.gethostname()


def sprites_loaddata(path, Src_domain, Tar_domain, seed=0):
    directions = ['front', 'left', 'right']
    actions = ['spellcard', 'thrust', 'walk', 'slash', 'shoot']
    import time
    start = time.time()
    X_src = []
    X_tar = []

    y_attribute_src = []
    y_attribute_tar = []
    y_action_src = []
    y_action_tar = []
    for act in range(len(actions)):
        for i in range(len(directions)):
            label = 3 * act + i
            print(actions[act], directions[i], act, i, label)
            x = np.load(path + Src_domain + '/%s_%s_frames_data.npy' %
                        (actions[act], directions[i]))
            X_src.append(x)
            y = np.load(path + Tar_domain + '/%s_%s_frames_data.npy' %
                        (actions[act], directions[i]))
            X_tar.append(y)

            a = np.load(path + Src_domain + '/%s_%s_attributes_data.npy' %
                        (actions[act], directions[i]))
            y_attribute_src.append(a[:, 0, :])
            d = np.zeros([a.shape[0]])
            d[:] = label
            y_action_src.append(d)

            a = np.load(path + Tar_domain + '/%s_%s_attributes_data.npy' %
                        (actions[act], directions[i]))
            y_attribute_tar.append(a[:, 0, :])
            d = np.zeros([a.shape[0]])
            d[:] = label
            y_action_tar.append(d)

    X_src = np.concatenate(X_src, axis=0)
    X_src = X_src.transpose((0, 1, 4, 2, 3))
    X_tar = np.concatenate(X_tar, axis=0)
    X_tar = X_tar.transpose((0, 1, 4, 2, 3))
    np.random.seed(seed)
    ind = np.random.permutation(X_src.shape[0])
    X_src = X_src[ind]

    y_attribute_src = np.concatenate(y_attribute_src, axis=0)
    y_action_src = np.concatenate(y_action_src)
    y_attribute_src = y_attribute_src[ind]
    y_action_src = y_action_src[ind]

    ind = np.random.permutation(X_tar.shape[0])
    X_tar = X_tar[ind]

    y_attribute_tar = np.concatenate(y_attribute_tar, axis=0)
    y_action_tar = np.concatenate(y_action_tar)
    y_attribute_tar = y_attribute_tar[ind]
    y_action_tar = y_action_tar[ind]

    end = time.time()
    print('data loaded in %.2f seconds...' % (end - start))

    return X_src, X_tar, y_attribute_src, y_attribute_tar, y_action_src, y_action_tar


def find_pseudo(target_data, target_index, psuedo_label, psuedo_index):
    intersection = list(set(target_index.numpy()) & set(psuedo_index.numpy()))
    tar_sub_index = [target_index.tolist().index(intersection[i])
                     for i in range(len(intersection))]
    pse_sub_index = [psuedo_index.tolist().index(intersection[i])
                     for i in range(len(intersection))]

    return target_data[tar_sub_index, :, :], psuedo_label[pse_sub_index, :]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, is_pretrain, path_exp, filename='checkpoint.pth.tar'):
    if is_pretrain:
        path_file = path_exp + 'model_pretrain.pth.tar'
        torch.save(state, path_file)
    else:
        # path_file = path_exp + filename
        # torch.save(state, path_file)
        # if is_best:
        #     path_best = path_exp + 'model_best.pth.tar'
        #     shutil.copyfile(path_file, path_best)
        if is_best:
            path_best = path_exp + 'model_best.pth.tar'
            torch.save(state, path_best)


def attentive_entropy(pred, pred_domain):
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
    entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
    weights = 1 + entropy
    loss = torch.mean(
        weights * torch.sum(-softmax(pred) * logsoftmax(pred), 1))
    return loss


def adjust_learning_rate_loss(optimizer, decay, stat_current, stat_previous, op):
    ops = {'>': (lambda x, y: x > y), '<': (lambda x, y: x < y),
           '>=': (lambda x, y: x >= y), '<=': (lambda x, y: x <= y)}
    if ops[op](stat_current, stat_previous):
        for param_group in optimizer.param_groups:
            param_group['lr'] /= decay


def adjust_learning_rate_dann(optimizer, p, opt):
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr / (1. + 10 * p) ** 0.75


def adjust_learning_rate(optimizer, decay):
    """Sets the learning rate to the initial LR decayed by 10 """
    for param_group in optimizer.param_groups:
        param_group['lr'] /= decay


def print_log(print_string, log=None):
    print("{}".format(print_string))
    if log is not None:
        log = open(log, 'a')
        log.write('{}\n'.format(print_string))
        log.close()


def logsumexp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        raise ValueError('Must specify the dimension.')


def log_density(sample, mu, logsigma):
    mu = mu.type_as(sample)
    logsigma = logsigma.type_as(sample)
    c = torch.Tensor([np.log(2 * np.pi)]).type_as(sample.data)
    inv_sigma = torch.exp(-logsigma)
    tmp = (sample - mu) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * logsigma + c)


def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M+1] = 1 / N
    W.view(-1)[1::M+1] = strat_weight
    W[M-1, 0] = strat_weight
    return W.log()


def loss_fn_new(original_seq, recon_seq, 
                f_mean, f_logvar, 
                z_post_mean, z_post_logvar, z_post,
                z_prior_mean, z_prior_logvar, z_prior
    ):
    batch_size = original_seq.size(0)
    mse = F.mse_loss(recon_seq, original_seq, reduction='sum')/batch_size
    f_mean = f_mean.view((-1, f_mean.shape[-1]))
    f_logvar = f_logvar.view((-1, f_logvar.shape[-1]))
    
    kld_f = -0.5 * \
        torch.sum(1 + f_logvar - torch.pow(f_mean, 2) -
                  torch.exp(f_logvar))/batch_size

    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar + ((z_post_var +
                            torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)/batch_size

    return {'mse': mse, 'kld_f': kld_f, 'kld_z': kld_z}


def save_np_img(fname, x):
    if x.shape[0] == 1:
        x = np.tile(x, (3, 1, 1))
    img = scipy.misc.toimage(x, high=255*x.max(), channel_axis=0)
    img.save(fname)


def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    return scipy.misc.toimage(tensor.numpy(), high=255*float(tensor.max()), channel_axis=0)


def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0, 0, 0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)


def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
