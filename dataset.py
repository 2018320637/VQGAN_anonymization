import os
import os.path
from PIL import Image
import numpy as np
from numpy.random import randint
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pickle


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file, num_dataload,
                 num_segments=3, new_length=1, modality='RGB', sample_every_n_frames = 2,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False, opts=None, data_name=None):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.sample_every_n_frames = sample_every_n_frames
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_dataload = num_dataload
        self.triple = opts.triple
        self.input_H = opts.reso_h
        self.input_W = opts.reso_w
        self.data_name = data_name
        self.transform = transforms.Compose([
            transforms.Resize((self.input_W, self.input_H)),
            transforms.ToTensor()
        ])

        if self.modality == 'RGBDiff' or self.modality == 'RGBDiff2' or self.modality == 'RGBDiffplus':
            self.new_length += 1

        if self.data_name == 'hmdb':
            with open('/home/zhiwei/source/dataset/hmdb51_privacy_label.pickle', 'rb') as f:
                self.privacy_data = pickle.load(f)
        elif self.data_name == 'ucf':
            with open('/home/zhiwei/source/dataset/ucf101_privacy_label.pickle', 'rb') as f:
                self.privacy_data = pickle.load(f)

        self._parse_list()

    def _process_image(self, img):
        h, w = img.height, img.width
        if h > w:
            half = (h - w) // 2
            cropsize = (0, half, w, half + w) 
        elif w > h:
            half = (w - h) // 2
            cropsize = (half, 0, half + h, h)

        if h != w:
            img = img.crop(cropsize)

        img = img.resize((self.input_W, self.input_H),Image.LANCZOS)
        img = np.asarray(img, dtype=np.float32)
        img /= 255.
        img = img - 0.5
        img_tensor = torch.from_numpy(img)
        return img_tensor

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff' or self.modality == 'RGBDiff2' or self.modality == 'RGBDiffplus':
            img_path = os.path.join(directory, self.image_tmpl.format(idx))
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self._process_image(img)
                img_tensor = img_tensor.permute(2, 0, 1)
                return [img_tensor]

            except:
                img = [torch.zeros(3, self.input_W, self.input_H)]
                print('Error loading: {}'.format(img_path))
            return img

        elif self.modality == 'Flow':
            x_feat = torch.load(os.path.join(directory, self.image_tmpl.format('x', idx)))
            y_feat = torch.load(os.path.join(directory, self.image_tmpl.format('y', idx)))

            return [x_feat, y_feat]

    def _parse_list(self):
        base_path = '/home/zhiwei/source/'
        self.video_list = [VideoRecord([os.path.join(base_path, x.strip().split(' ')[0])] + x.strip().split(' ')[1:])
                   for x in open(self.list_file)]
        n_repeat = self.num_dataload//len(self.video_list)
        n_left = self.num_dataload % len(self.video_list)
        self.video_list = self.video_list*n_repeat + self.video_list[:n_left]

    def _sample_indices(self, record):
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames -
                              self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _sample_indices_every_n_frames(self, record):
        n_frames_interval = self.num_segments * self.sample_every_n_frames
        if record.num_frames > n_frames_interval + 1:
            start_idx = randint(0, record.num_frames - 1 - n_frames_interval)
            end_idx = start_idx + n_frames_interval
            return list(range(start_idx + 1, end_idx + 1, self.sample_every_n_frames))
        else:
            offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        num_min = self.num_segments + self.new_length - 1
        num_select = record.num_frames - self.new_length + 1

        if record.num_frames >= num_min:
            tick = float(num_select) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * float(x))
                               for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):
        num_min = self.num_segments + self.new_length - 1
        num_select = record.num_frames - self.new_length + 1

        if record.num_frames >= num_min:
            tick = float(num_select) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * float(x))
                               for x in range(self.num_segments)])
        else:
            id_select = np.array([x for x in range(num_select)])
            id_expand = np.ones(self.num_segments-num_select,
                                dtype=int)*id_select[id_select[0]-1]
            offsets = np.append(id_select, id_expand)

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        segment_indices = self._sample_indices_every_n_frames(record)
        data_ancher, label_ancher, privacy_label = self.get(record, segment_indices)

        if self.triple:
            perm = np.random.permutation(data_ancher.shape[0])
            data_pos = data_ancher[perm]

            index_neg = np.random.randint(self.num_dataload)
            record_neg = self.video_list[index_neg]

            if not self.test_mode:
                segment_indices_neg = self._sample_indices(
                    record_neg) if self.random_shift else self._get_val_indices(record_neg)
            else:
                segment_indices_neg = self._get_test_indices(record_neg)

            data_neg, _, _ = self.get(record_neg, segment_indices_neg)

            return data_ancher, label_ancher,privacy_label, data_pos, data_neg

        return data_ancher, label_ancher, privacy_label

    def get(self, record, indices):

        frames = list()

        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                frames.extend(seg_imgs)

                if p < record.num_frames:
                    p += 1

        process_data = torch.stack(frames)
        privacy_label = self.privacy_data[record.path.split('/')[-1].split('.')[0]][0]
        privacy_label = torch.Tensor(np.array(privacy_label))

        return process_data, record.label, privacy_label

    def __len__(self):
        return len(self.video_list)
