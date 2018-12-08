import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import copy
import random

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        import accimage
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def make_dataset(video_listfile, sample_duration, n_samples_for_each_video):
    dataset = []
    video_list = []
    with open(video_listfile, 'r') as f:
        for lines in f.readlines():
            video_list.append(lines.strip('\n').split(' '))
    for video_path, n_frames, label in video_list:
        n_frames = int(n_frames)
        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'label': int(label)
        }
        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1, math.ceil((n_frames - 1 - sample_duration) /
                                        (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for i in range(1, n_frames, step):
                sample_i = copy.deepcopy(sample)
                sample_i['frame_indices'] = list(range(i, min(n_frames, i + sample_duration)))
                dataset.append(sample_i)
    return dataset

class Video(data.Dataset):
    def __init__(self, video_listfile,
                 spatial_transform=None, temporal_transform=None,
                 sample_duration=16, n_samples_for_each_video=1, get_loader=get_default_video_loader):
        self.data = make_dataset(video_listfile, sample_duration, n_samples_for_each_video)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        video_path = self.data[index]['video']
        label = self.data[index]['label']
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(video_path, frame_indices)
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip, label, video_path

    def __len__(self):
        return len(self.data)
