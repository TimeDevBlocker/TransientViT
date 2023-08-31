import os
from PIL import Image
import itertools
import torch
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

from .mwr_utils import get_age_bounds, get_rho, get_age_bounds

from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

train_transform = transforms.Compose([
            transforms.Resize([384, 384]),
            # transforms.CenterCrop(320), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),            
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

test_transform = transforms.Compose([
            transforms.Resize([384, 384]),
            # transforms.CenterCrop(320), 
            transforms.ToTensor(),            
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

class MWR(Dataset):
    def __init__(self, 
                 root, 
                 split='train',
                 class_map=None,
                 img_mode='RGB',
                 load_bytes=False,
                 transform=None, 
                 target_transform=None,
                 tau=5):
        
        self.root = root
        self.split = split
        self.class_map = class_map
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        
        self.imlist = self.flist_reader()	
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0


        self.pg = PairGenerator(train_data=self.imlist, tau=tau)
        self.idx_min, self.idx_max, self.idx_test, self.label = None, None, None, None

        self.mean = torch.as_tensor(imagenet_stats['mean'], dtype=torch.float32)
        self.std = torch.as_tensor(imagenet_stats['mean'], dtype=torch.float32)

    def flist_reader(self):
        flist_path = os.path.join(self.root, f'{self.split}.txt')
        imlist = []
        with open(flist_path, 'r') as rf:
            for line in rf.readlines():
                # print(line)
                impath, imlabel = line.strip().rsplit(' ', maxsplit=1)
                imlist.append((impath, int(imlabel))) 
        return imlist

    def __getitem__(self, idx):
        if 'train' in self.split:
            base_min_name, base_min_age = self.imlist[self.idx_min[idx]]
            base_max_name, base_max_age = self.imlist[self.idx_max[idx]]
            test_name, test_age = self.imlist[self.idx_test[idx]]

            label = self.imlist[idx][1]

            base_min_image = Image.open(base_min_name)
            base_max_image = Image.open(base_max_name)
            test_image = Image.open(test_name)

            base_min_image_trans = train_transform(base_min_image)
            base_max_image_trans = train_transform(base_max_image)
            test_image_trans = train_transform(test_image)

            sample = dict()
            sample['base_min_impath'] = base_min_name
            sample['base_max_impath'] = base_max_name
            sample['test_impath'] = test_name

            sample['base_min_age'] = base_min_age
            sample['base_max_age'] = base_max_age
            sample['test_age'] = test_age

            sample['label'] = label

            # sample['base_min_image_orig'] = np.array(base_min_image)
            # sample['base_max_image_orig'] = np.array(base_max_image)
            # sample['test_image_orig'] = np.array(test_image)

            sample['base_min_image'] = base_min_image_trans
            sample['base_max_image'] = base_max_image_trans
            sample['test_image'] = test_image_trans

        else:
            test_name, test_age = self.imlist(idx)
            test_image = Image.open(test_name)

            test_image_trans = test_transform(test_image)

            sample = dict()
            sample['test_impath'] = test_name
            sample['test_age'] = test_age
            # sample['test_image_orig'] = np.array(test_image)
            sample['test_image'] = test_image_trans

        return sample

    def __len__(self):
        return len(self.imlist)

    def get_pair_lists(self):
        self.idx_min, self.idx_max, self.idx_test, self.label = self.pg.get_ternary_mwr_pairs(rank=[x[1] for x in self.imlist])


# class MorphRef(Dataset):
#     def __init__(self, 
#                  root, 
#                  split='train',
#                  class_map=None,
#                  img_mode='RGB',
#                  load_bytes=False,
#                  transform=None, 
#                  target_transform=None,
#                  tau=5):
        
#         self.root = root
#         self.split = split
#         self.class_map = class_map
#         self.load_bytes = load_bytes
#         self.img_mode = img_mode
        
#         self.imlist = self.flist_reader()	
#         self.transform = transform
#         self.target_transform = target_transform
#         self._consecutive_errors = 0

#         self.tau = tau

#         train_data, _, reg_bound, sampling, sample_rate = data_select(cfg)
#         if mwr_sampling:

#             if os.path.exists(cfg.sampled_refs):
#                 train_data = pd.read_csv(cfg.sampled_refs)
#                 print('Load sampled datalist: ', cfg.sampled_refs)
#             else:
#                 data_train_age = train_data['age'].to_numpy().astype('int')
#                 data_train_age_unique = np.unique(data_train_age)

#                 sampled_idx = []

#                 ### Sample data from training set ###
#                 for tmp in data_train_age_unique:
#                     valid_idx = np.where(data_train_age == tmp)[0]
#                     sample_num = int(len(valid_idx) * sample_rate + 0.5)

#                     if sample_num < 1:
#                         sample_num = 1
#                         idx_choice = np.random.choice(valid_idx, sample_num, replace=True)
#                     else:
#                         idx_choice = np.random.choice(valid_idx, sample_num, replace=False)

#                     sampled_idx.append(idx_choice.tolist())

#                 sampled_idx = np.array(sum(sampled_idx, []))
#                 train_data_sampled = train_data.iloc[sampled_idx].reset_index(drop=True)

#                 train_data_sampled.to_csv(cfg.sampled_refs)
#                 print('Save sampled datalist to', cfg.sampled_refs)

#         self.df_base = train_data

#     def __getitem__(self, idx):
#         ref_name = os.path.join(self.image_dir, self.df_base['filename'].iloc[idx])
#         ref_image = Image.open(ref_name)
#         ref_image_trans = ImgAugTransform_Test(ref_image).astype(np.float32) / 255.
#         ref_image_trans = torch.from_numpy(np.transpose(ref_image_trans, (2, 0, 1)))

#         dtype = ref_image_trans.dtype
#         mean = torch.as_tensor(imagenet_stats['mean'], dtype=dtype, device=ref_image_trans.device)
#         std = torch.as_tensor(imagenet_stats['mean'], dtype=dtype, device=ref_image_trans.device)
#         ref_image_trans.sub_(mean[:, None, None]).div_(std[:, None, None])

#         sample = dict()
#         sample['ref_impath'] = ref_name
#         sample['ref_image_orig'] = np.array(ref_image)
#         sample['ref_image'] = ref_image_trans

#         return sample

#     def __len__(self):
#         return len(self.df_base)


class MWRRefSampling(Dataset):
    def __init__(self, 
                 root, 
                 split='train',
                 class_map=None,
                 img_mode='RGB',
                 load_bytes=False,
                 transform=None, 
                 target_transform=None,
                 tau=5):
        
        self.root = root
        self.split = split
        self.class_map = class_map
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        
        self.imlist = self.flist_reader()	
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

        self.tau = tau
        self.ref_sampling(pair_num=5)

    def flist_reader(self):
        flist_path = os.path.join(self.root, f'{self.split}.txt')
        imlist = []
        with open(flist_path, 'r') as rf:
            for line in rf.readlines():
                # print(line)
                impath, imlabel = line.strip().rsplit(' ', maxsplit=1)
                imlist.append((impath, int(imlabel))) 
        return imlist

    def __getitem__(self, idx):
        ref_name = self.imlist[idx][0]
        ref_image = Image.open(ref_name)

        ref_image_trans = test_transform(ref_image)

        sample = dict()
        sample['ref_impath'] = ref_name
        sample['ref_image_orig'] = np.array(ref_image)
        sample['ref_image'] = ref_image_trans

        return sample

    def __len__(self):
        return len(self.base_filename_np)

    def ref_sampling(self, pair_num=5):
        print(f'Sampling References / Sampling Num {pair_num}')
        base_age = np.array([x[1] for x in self.imlist])
        base_age_unique, base_age_num = np.unique(base_age, return_counts=True)
        lower_bounds, upper_bounds = get_age_bounds(int(base_age_unique.max()), base_age_unique, self.tau)

        base_min_idx_list = list()
        base_max_idx_list = list()

        base_min_age_list = list()
        base_max_age_list = list()

        for base_min_age in base_age_unique:

            base_min_idx = np.where(base_age == base_min_age)[0]

            base_max_age = upper_bounds[base_min_age]
            base_max_idx = np.where(base_max_age == base_age)[0]

            pair_list = np.array(list(itertools.product(base_min_idx, base_max_idx)))
            pair_idx = np.arange(len(pair_list))

            if len(pair_list) >= pair_num:
                pair_idx_selected = np.random.choice(pair_idx, pair_num, replace=False)
            else:
                pair_idx_selected = np.random.choice(pair_idx, pair_num, replace=True)

            base_min_idx_list.extend(pair_list[:, 0][pair_idx_selected].tolist())
            base_max_idx_list.extend(pair_list[:, 1][pair_idx_selected].tolist())

            base_min_age_list.extend([base_min_age] * pair_num)
            base_max_age_list.extend([base_max_age] * pair_num)

        base_min_idx_list = np.array(base_min_idx_list)
        base_max_idx_list = np.array(base_max_idx_list)

        self.base_min_age_np = np.array(base_min_age_list)
        self.base_max_age_np = np.array(base_max_age_list)

        base_unique_idx = np.unique(np.concatenate([base_min_idx_list, base_max_idx_list]))
        self.base_filename_np = [x[0] for x in self.imlist]
        self.base_age_np = [x[1] for x in self.imlist]

        mapping = dict()
        for i, unique_idx in enumerate(base_unique_idx):
            mapping[str(unique_idx)] = i

        self.base_min_idx_np = np.array(list(map(lambda x: mapping[str(x)], base_min_idx_list)))
        self.base_max_idx_np = np.array(list(map(lambda x: mapping[str(x)], base_max_idx_list)))

        print(f'Sampling Finished / Total Pair Num {len(self.base_min_idx_np)}')




class PairGenerator:
    def __init__(self,train_data, tau=0.2):
        self.tau = tau
        self.data = train_data

    def get_ternary_mwr_pairs(self, rank):

        rank = np.array(rank)

        sample_idx = [[] for _ in range(3)]
        sample_label = []

        rank_unique, rank_num = np.unique(rank, return_counts=True)
        rank_min, rank_max = rank_unique.min(), rank_unique.max()

        idx = np.arange(len(rank))
        np.random.shuffle(idx)

        delta = self.tau - 0.05

        lower_bounds, upper_bounds = get_age_bounds(int(rank_max), rank_unique, self.tau)
        lower_bounds_delta, upper_bounds_delta = get_age_bounds(int(rank_max), rank_unique, delta)

        for min_base_idx in idx:
            min_base_rank = rank[min_base_idx]

            #max_base_rank = rank_unique[np.abs(np.log(rank_unique) - (np.log(min_base_rank) + self.tau)).argmin()]
            max_base_rank = upper_bounds[min_base_rank]
            max_base_idx = np.where(max_base_rank == rank)[0]
            max_base_idx = np.random.choice(max_base_idx)

            while True:

                flag = np.random.rand()
                if flag < 0.1:
                    start = rank_min
                    end = max(rank_min, lower_bounds_delta[int(min_base_rank)])
                    test_rank = np.where((rank_unique >= start) & (rank_unique < end))[0]

                elif (flag >= 0.1) & (flag < 0.3):
                    start = max(rank_min, lower_bounds_delta[int(min_base_rank)])
                    end = max(rank_min, upper_bounds_delta[int(min_base_rank)])
                    test_rank = np.where((rank_unique >= start) & (rank_unique < end))[0]

                elif (flag >= 0.3) & (flag < 0.7):
                    start = max(rank_min, min_base_rank)
                    end = min(rank_max, max_base_rank)
                    test_rank = np.where((rank_unique >= start) & (rank_unique < end))[0]

                elif (flag >= 0.7) & (flag < 0.9):
                    start = max(rank_min, lower_bounds_delta[int(max_base_rank)])
                    end = min(rank_max, upper_bounds_delta[int(max_base_rank)])
                    test_rank = np.where((rank_unique >= start) & (rank_unique < end))[0]

                elif flag >= 0.9:
                    start = min(rank_max, upper_bounds_delta[int(max_base_rank)])
                    end = rank_max
                    test_rank = np.where((rank_unique > start) & (rank_unique <= end))[0]

                if len(test_rank) == 0:
                    continue

                else:
                    test_rank = np.random.choice(test_rank)
                    test_rank = rank_unique[test_rank]

                    test_idx = np.where(rank == test_rank)[0]

                    if len(test_idx) > 0:

                        label = get_rho(np.array([min_base_rank]), 
                                        np.array([max_base_rank]), 
                                        np.array([test_rank]))[0]
                        break

            test_idx = np.random.choice(test_idx)

            sample_idx[0].append(min_base_idx)
            sample_idx[1].append(max_base_idx)
            sample_idx[2].append(test_idx)
            sample_label.append(label)

        permute = np.arange(len(sample_idx[0]))
        np.random.shuffle(permute)
        sample_idx_min = np.array(sample_idx[0])[permute]
        sample_idx_max = np.array(sample_idx[1])[permute]
        sample_idx_test = np.array(sample_idx[2])[permute]
        sample_label = np.array(sample_label)[permute]

        return sample_idx_min, sample_idx_max, sample_idx_test, sample_label




# if __name__ == '__main__':
#     from configs.config_v1 import ConfigV1 as Config
#     from torch.utils.data import DataLoader

#     pass