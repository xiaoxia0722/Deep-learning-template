# encoding: utf8
# @Author: qiaoyongchao
# @Email: qiaoyc2023@163.com

import yaml
from torch import distributed as dist


def get_config(args):
    """
    将args和yml合并，args优先
    :param args:
    :return:
    """
    config_file = args.config
    with open(config_file, 'r', encoding='utf8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    if args.mode:
        configs['mode'] = args.mode

    if configs['mode'] == 'train':
        if args.epochs:
            configs['epochs'] = args.epochs

        if args.gpus:
            configs['gpus'] = args.gpus

        if args.output:
            configs['output'] = args.output

        if args.batch_size:
            configs['batch_size'] = args.batch_size

        if args.resume_epoch:
            configs['resume_epoch'] = args.resume_epoch

        if args.resume_model_path:
            configs['resume_model_path'] = args.resume_model_path

        if args.test_interval:
            configs['test_interval'] = args.test_interval

        if args.resume_epoch:
            configs['resume_epoch'] = args.resume_epoch

        if args.resume_model_path:
            configs['resume_model_path'] = args.resume_model_path

    elif configs['mode'] != 'other':
        configs['load_dir'] = args.load_dir

    return configs


# save and load models
def load_opt_weights(optimizer, weights):
    """
    加载优化器权重的封装方法
    :param optimizer:
    :param weights:
    :return:
    """
    optimizer.load_state_dict(weights)
    return optimizer


def save_args(save_path, args):
    """
    保存配置参数yml文件
    :param save_path:
    :param args:
    :return:
    """
    fp = open(save_path, 'w')
    fp.write(yaml.dump(args))
    fp.close()


# DDP utils
def get_rank():
    """
    用于判断多GPU时的当前线程
    :return:
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def load_model_weights(model, weights, multi_gpus, train=True):
    """
    封装的模型加载方法
    :param model:
    :param weights:
    :param multi_gpus:
    :param train:
    :return:
    """
    if list(weights.keys())[0].find('module')==-1:
        pretrained_with_multi_gpu = False
    else:
        pretrained_with_multi_gpu = True
    if (multi_gpus==False) or (train==False):
        if pretrained_with_multi_gpu:
            state_dict = {
                key[7:]: value
                for key, value in weights.items()
            }
        else:
            state_dict = weights
    else:
        state_dict = weights
    model.load_state_dict(state_dict)
    return model
