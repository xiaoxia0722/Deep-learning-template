# encoding: utf8
# @Author: qiaoyongchao
# @Email: qiaoyc2023@163.com


from src.utils import *

import torch.utils.data as data


def prepare_datasets(args):
    """
    dataset加载封装
    :param args:
    :return:
    """
    # train dataset
    train_dataset = prepare_dataset(args, split='train')
    # val dataset
    val_dataset = prepare_dataset(args, split='val')
    # test dataset
    test_dataset = prepare_dataset(args, split='test')
    return train_dataset, val_dataset, test_dataset


def prepare_dataset(configs, split):
    """
    dataset相关处理封装，可在此加入transforms等相关工作
    :param configs:
    :param split:
    :return:
    """
    # train dataset
    dataset = Dataset(split=split, configs=configs)
    return dataset


################################################################
#                    Dataset
################################################################
class Dataset(data.Dataset):
    """
    数据集的封装
    """
    def __init__(self, split, configs=None):
        if split == 'train':
            self.data = []
        elif split == 'val':
            self.data = []
        else:
            self.data = []

    def __getitem__(self, index):
        imgs, label = self.data[index]
        return imgs, label

    def __len__(self):
        return len(self.data)


