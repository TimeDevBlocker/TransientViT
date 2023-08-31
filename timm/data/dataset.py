""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
from typing import Optional
import os
import torch
import torch.utils.data as data
import random
from PIL import Image
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from .readers import create_reader
import numpy as np
from .mwr_utils import get_rho, get_age_bounds

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            img_mode='RGB',
            transform=None,
            target_transform=None,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.reader[index]

        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.img_mode and not self.load_bytes:
            img = img.convert(self.img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)

class ImageFileList(data.Dataset):
    def __init__(
            self, 
            root, 
            split='train',
            class_map=None,
            img_mode='RGB',
            load_bytes=False,
            transform=None, 
            target_transform=None):

        self.root = root
        self.split = split
        self.class_map = class_map
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        
        self.imlist = self.flist_reader()	
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def flist_reader(self):
        flist_path = os.path.join(self.root, f'{self.split}.txt')
        imlist = []
        with open(flist_path, 'r') as rf:
            for line in rf.readlines():
                # print(line)
                impath, imlabel = line.strip().rsplit(' ', maxsplit=1)
                imlist.append((impath, int(imlabel))) 
        return imlist

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        # img_path = os.path.join(self.root,impath)
        img_path = impath
        try:
            img = Image.open(img_path)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {img_path}). {str(e)}')

        if self.img_mode and not self.load_bytes:
            img = img.convert(self.img_mode)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imlist)

class ImageFileList_Multi(data.Dataset):
    def __init__(
            self, 
            root, 
            split='train',
            class_map=None,
            img_mode='RGB',
            load_bytes=False,
            transform=None, 
            target_transform=None,
            super_img=False):

        self.root = root
        self.split = split
        self.class_map = class_map
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.super_img = super_img
        self.super_img_dir = '/public/191-aiprime/jiawei.dong/projects/kats/kats_downloader/imgData'
        self.imlist = self.flist_reader()	
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def flist_reader(self):
        flist_path = os.path.join(self.root, f'{self.split}.txt')
        imlist = []
        with open(flist_path, 'r') as rf:
            for line in rf.readlines():
                # print(line)
                impath, imlabel, imcnt = line.strip().rsplit(' ', maxsplit=2)
                imlist.append((impath, int(imlabel), int(imcnt))) 
        return imlist

    def __getitem__(self, index):
        impre, target, imcnt = self.imlist[index]
        # img_path = os.path.join(self.root,impath)
        imgs = []
        # for super image
        if self.super_img:
            # /public/191-aiprime/jiawei.dong/projects/kats/kats_cutout/20230529/True/20230529_6230529192349000499
            _, img_date, img_class, img_name = impre.rsplit('/', 3)
            img_date, img_id = img_name.split('_')
            img_path = os.path.join(self.super_img_dir, img_date, str(img_class), img_id+'.jpg')
            # img_path = f'{self.super_img_dir}/{img_date}/{str(img_class)}/{img_id}.jpg'
            try:
                img = Image.open(img_path)
            except Exception as e:
                _logger.warning(f'Skipped sample (index {index}, file {img_path}). {str(e)}')

            if self.img_mode and not self.load_bytes:
                img = img.convert(self.img_mode)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target
        
        # for 6 channel 
        else:
            idxs = np.random.choice(imcnt, 2)
            for i in idxs:
                if imcnt > 1:
                    img_path = f'{impre}_{i}.jpg'
                else:
                    img_path = f'{impre}.jpg'
                try:
                    img = Image.open(img_path)
                except Exception as e:
                    _logger.warning(f'Skipped sample (index {index}, file {img_path}). {str(e)}')
                
                width, height = img.size
                new_w, new_h = 48, 48
                # 裁剪图像
                if width > new_w or height > new_h:
                    left = (width - new_w)/2
                    top = (height - new_h)/2
                    right = (width + new_w)/2
                    bottom = (height + new_h)/2
                    img = img.crop((left, top, right, bottom))

                if self.img_mode and not self.load_bytes:
                    img = img.convert(self.img_mode)

                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            imgs = torch.cat(imgs, dim=0)
            return imgs, target

    def __len__(self):
        return len(self.imlist)

class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            is_training=False,
            batch_size=None,
            seed=42,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                reader,
                root=root,
                split=split,
                is_training=is_training,
                batch_size=batch_size,
                seed=seed,
                repeats=repeats,
                download=download,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.reader:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.reader, '__len__'):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
        if hasattr(self.reader, 'set_loader_cfg'):
            self.reader.set_loader_cfg(num_workers=num_workers)

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)

class DogAgeImageFileList(data.Dataset):
    avg = 0
    std = 0

    def __init__(
            self, 
            root, 
            split='train',
            class_map=None,
            img_mode='RGB',
            load_bytes=False,
            transform=None, 
            target_transform=None,
            reweight='sqrt_inv',
            lds=True, 
            lds_kernel='gaussian', 
            lds_ks=5, 
            lds_sigma=2):

        self.root = root
        self.split = split
        self.class_map = class_map
        self.load_bytes = load_bytes
        self.img_mode = img_mode

        self.istest = 'ensemble' in self.split
        self.imlist = self.flist_reader(istest=self.istest)

        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

        self.age_avg = 56.61900934391641
        self.age_diff = 190

        self.log_max = np.log(190)
        self.log_min = np.log(1)
        # self.weights = None
        # if 'train' in self.split:
        #     self.weights = self._prepare_weights(reweight=reweight,
        #                                         lds=lds,
        #                                         lds_kernel=lds_kernel,
        #                                         lds_ks=lds_ks, 
        #                                         lds_sigma=lds_sigma)


        if 'train' in self.split:
            age_sum = sum([x[1] for x in self.imlist])
            labels = [x[1] for x in self.imlist]
            self.age_diff = 190
            DogAgeImageFileList.avg = age_sum / len(self.imlist)
            DogAgeImageFileList.std = np.std(labels)
            print(f'avg: {DogAgeImageFileList.avg}, std: {DogAgeImageFileList.std}')

    def age_norm(self, x):
        # return x
        # x = np.log(x)
        # x = (x - self.log_min) / self.log_max
        # x = (x - self.age_avg) / self.age_diff
        # x = self.gaussian(x)
        x = (x - DogAgeImageFileList.avg) / DogAgeImageFileList.std
        return x
    
    
    def gaussian(self, x):
        x = np.exp(-np.power(x - DogAgeImageFileList.avg, 2.) / (2 * np.power(DogAgeImageFileList.std, 2.)))
        x = (1 / (np.log2(2 * np.pi) * DogAgeImageFileList.std)) * x
        return x

    def flist_reader(self, istest=False):
        flist_path = os.path.join(self.root, f'{self.split}.txt')
        imlist = []
        with open(flist_path, 'r') as rf:
            for line in rf.readlines():
                # print(line)
                if not istest:
                    impath, imlabel = line.strip().rsplit(' ', maxsplit=1)
                    imlist.append((impath, int(imlabel))) 
                else:
                    impath, imlabel = line.strip().rsplit('\t', maxsplit=1)
                    impath = os.path.join('/public/share/challenge_dataset/AFAC-Track3-2023/testset', impath)
                    imlist.append((impath, float(imlabel))) 
        return imlist

    def __getitem__(self, index):
        # index = index % len(self.imlist)

        impath, target = self.imlist[index]

        # if 'val' not in self.split:
        #     target = self.age_norm(target)

        # img_path = os.path.join(self.root,impath)

        try:
            img = Image.open(impath)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {impath}). {str(e)}')

        if self.img_mode and not self.load_bytes:
            img = img.convert(self.img_mode)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])
        
        levels = torch.ones(191, dtype=torch.float)
        if not self.istest:
            levels = self.cvt_target_2levels(target)
        
        # return img, target
        
        # target -= 1 for softmax loss
        if 'val' not in self.split:
            target -= 1

        return img, target, levels
    

    def cvt_target_2levels(self, target):
        levels = [1]*target + [0]*(191 - target)
        levels = torch.tensor(levels, dtype=torch.float32)
        return levels

    # def _prepare_weights(self, reweight='sqrt_inv', max_target=192, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    #     assert reweight in {'none', 'inverse', 'sqrt_inv'}
    #     assert reweight != 'none' if lds else True, \
    #         "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

    #     value_dict = {x: 0 for x in range(max_target)}
    #     # labels = self.df['age'].values
    #     labels = [x[1] for x in self.imlist]
    #     for label in labels:
    #         value_dict[min(max_target - 1, int(label))] += 1
    #     if reweight == 'sqrt_inv':
    #         value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    #     elif reweight == 'inverse':
    #         value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
    #     num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
    #     if not len(num_per_label) or reweight == 'none':
    #         return None
    #     print(f"Using re-weighting: [{reweight.upper()}]")

    #     if lds:
    #         lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
    #         print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
    #         smoothed_value = convolve1d(
    #             np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
    #         num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

    #     weights = [np.float32(1 / x) for x in num_per_label]
    #     scaling = len(weights) / np.sum(weights)
    #     weights = [scaling * x for x in weights]
        return weights

    def __len__(self):
        return len(self.imlist)
    

# def get_lds_kernel_window(kernel, ks, sigma):
#     assert kernel in ['gaussian', 'triang', 'laplace']
#     half_ks = (ks - 1) // 2
#     if kernel == 'gaussian':
#         base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
#         kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
#     elif kernel == 'triang':
#         kernel_window = triang(ks)
#     else:
#         laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
#         kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

#     return kernel_window


class DogBodyHeadFileList(DogAgeImageFileList):
    avg = 0
    std = 0
    def __init__(
            self, 
            root, 
            split='train',
            class_map=None,
            img_mode='RGB',
            load_bytes=False,
            transform=None, 
            target_transform=None,
            reweight='sqrt_inv',
            lds=True, 
            lds_kernel='gaussian', 
            lds_ks=5, 
            lds_sigma=2):
        super(DogBodyHeadFileList, self).__init__(root, 
                                                split,
                                                class_map,
                                                img_mode,
                                                load_bytes,
                                                transform, 
                                                target_transform,
                                                reweight,
                                                lds, 
                                                lds_kernel, 
                                                lds_ks, 
                                                lds_sigma)

    def __getitem__(self, index):
        # index = index % len(self.imlist)

        impath, target = self.imlist[index]


        
        
        #     target = self.age_norm(target)

        # img_path = os.path.join(self.root,impath)
        impath_head = impath.replace('/public/share/challenge_dataset/AFAC-Track3-2023/trainset', '/public/share/challenge_dataset/AFAC-Track3-2023/dog_head/trainset')
        try:
            img = Image.open(impath)
            img_head = Image.open(impath_head)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {impath}). {str(e)}')

        if self.img_mode and not self.load_bytes:
            img = img.convert(self.img_mode)
            img_head = img_head.convert(self.img_mode)

        if self.transform is not None:
            img = self.transform(img)
            img_head = self.transform(img_head)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        
        

        # weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])
        levels = self.cvt_target_2levels(target)
        # return img, target
        img_all = torch.cat((img, img_head), dim=0)
        
        

        return img_all, target, levels