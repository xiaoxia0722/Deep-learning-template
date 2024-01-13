# encoding: utf8
# @Author: qiaoyongchao
# @Email: qiaoyc2023@163.com
import json

from torch.utils.tensorboard import SummaryWriter

from src.train import *
from src.dataset import *

import os
import time
import random
import datetime
import argparse
from loguru import logger

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


parser = argparse.ArgumentParser(description="Deep learning template ")

# 需要使用的实时配置参数
parser.add_argument("--config", help='配置文件(可指定参数优先于配置文件参数)', default='./configs/bird.yml')
parser.add_argument("--mode", help='代码运行模式(train或者test)', choices=['test', 'train'])
parser.add_argument("--epochs", help="训练时的轮数", type=int)
parser.add_argument("--batch_size", help="训练时每步的批次", type=int)
parser.add_argument("--test_interval", help="测试频率", type=int)
parser.add_argument("--local_rank", default=-1, type=int, help="pytorch多gpu训练时的节点")
parser.add_argument("--gpus", help='需要使用的gpu/cpu')
parser.add_argument("--output", help='保存文件目录')
parser.add_argument("--resume_epoch", help='训练的起始epoch', type=int)
parser.add_argument("--resume_model_path", help='训练的预加载模型')

args = parser.parse_args()

logger.info(args)

configs = get_config(args)

random.seed(configs['manual_seed'])
torch.manual_seed(configs['manual_seed'])
torch.cuda.manual_seed_all(configs['manual_seed'])

# 多GPU训练判断
many_gpus = False
local_rank = 0
gpus = False
if configs['gpus'] in ('cpu', '-1') or not torch.cuda.is_available():
    device = torch.device('cpu')
elif ',' not in configs['gpus']:
    device = torch.device(int(configs['gpus']))
    gpus = True
else:
    local_rank = int(configs['local_rank'])
    torch.distributed.init_process_group(backend='nccl')
    # local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    many_gpus = True
    gpus = True

# 处理日志文件
if not os.path.exists(configs['output']):
    logger.warning("输出目录{}不存在, 开始创建...".format(configs['output']))
    os.makedirs(configs['output'])

logger.add(os.path.join(configs['output'], 'log_{}_{}.log'.format(configs['mode'], datetime.datetime.now().strftime("%Y%m%d%H%M%S"))))

logger.info("配置文件名为:{}, 路径为:{}, 配置如下:\n{}".format(configs['config_name'], args.config, configs))

configs['many_gpus'] = many_gpus

logger.info("gpu是否可用:{}, gpu是否使用:{}, 多gpu是否使用:{}, gpu使用信息:{}".format(torch.cuda.is_available(), gpus, many_gpus, device))

logger.info("初始化数据集....")

# 创建dataset(如果没有test_dataset, 就将所有的test_dataset修改为val_test)
train_dataset, val_dataset, test_dataset = prepare_datasets(configs)

logger.info("训练数据大小:{}, 验证数据大小:{}, 测试数据大小:{}".format(len(train_dataset), len(val_dataset), len(test_dataset)))

logger.info("初始化dataloader...")
train_sampler = None
if many_gpus:
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], drop_last=True, num_workers=configs['num_workers'], sampler=train_sampler)
else:
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], drop_last=True, num_workers=configs['num_workers'], shuffle=True)

val_dataloader = DataLoader(val_dataset, batch_size=configs['batch_size'], num_workers=configs['num_workers'])
test_dataloader = DataLoader(test_dataset, batch_size=configs['batch_size'], num_workers=configs['num_workers'])

# 创建训练器
trainer = Trainer(configs, device, logger)

# 计算模型参数
if configs['params'] and (not many_gpus or local_rank == 0):
    params_dict = trainer.get_params()
    logger.info("模型的参数量为:{:.2f}M\tflops:{:.2f}G".format(params_dict['params'] / 1e6, params_dict['flops'] / 1e9))

writer = None
fixed_img = None
fixed_sent = None
fixed_z = None

# 创建tensorboard
if configs['tensorboard']:
    logger.info("创建tensorboard writer...")
    writer = SummaryWriter(os.path.join(configs['output'], 'logs'))


# 断点训练的历史训练数据加载
best_f1 = -1
start_epoch = 1
loss_list = []
if configs['resume_epoch'] != 1:
    start_epoch = configs['resume_epoch'] + 1
    path = os.path.join(configs['resume_model_path'], 'state_epoch_%03d.pth' % (configs['resume_epoch']))
    trainer.load_model_opt(path)
    with open(os.path.join(configs['resume_model_path'], 'loss.json'), 'r', encoding='utf8') as f:
        loss_list = json.load(f)
    loss_list = loss_list[:configs['resume_epoch']]
    best_f1 = min([c['f1'] for c in loss_list if 'f1' in c])

# 保存配置文件
if not many_gpus or configs.get('local_rank', -1) == 0:
    save_args(os.path.join(trainer.configs['output'], 'configs.yml'), trainer.configs)

logger.info("开始训练...")

# 模型训练
torch.cuda.empty_cache()
start_time = time.time()
for epoch in range(start_epoch, configs['epochs'] + 1):
    if many_gpus and get_rank() != 0:
        train_sampler.set_epoch(epoch)
    torch.cuda.empty_cache()    # 清除cuda缓存，释放资源
    loss_dict = trainer.train(epoch, train_dataloader)
    loss_dict['epoch'] = epoch
    if not many_gpus or configs.get('local_rank', -1) == 0:
        if writer:
            writer.add_scalar('loss', sum(loss_dict['losses']) / len(loss_dict['losses']), epoch)

        # 数据验证
        if epoch % configs['test_interval'] == 0 or epoch == configs['epochs'] or (configs['resume_epoch'] != 1 and epoch == start_epoch):
            torch.cuda.empty_cache()
            f1 = trainer.val(val_dataloader)
            loss_dict['f1'] = f1
            if f1 <= best_f1:
                logger.info("epoch:{}, new f1:{}, old f1:{}, 保存最优模型".format(epoch, f1, best_f1))
                trainer.save_model('best', epoch)
                best_fid = f1
            all_times = time.time() - start_time
            logger.info("val epoch:{} => f1:{}, best f1:{}, 当前耗时为:{}h, 预计总耗时为:{}h, 剩余耗时为:{}h".format(epoch, f1, best_f1, all_times / 3600, all_times / (epoch - start_epoch + 1) * (configs['epochs'] - start_epoch) / 3600, all_times / (epoch - start_epoch + 1) * (configs['epochs'] - epoch) / 3600))
            if writer:
                writer.add_scalar('f1', f1, epoch)

        if epoch % configs['save_interval'] == 0 or epoch == configs['epochs']:
            logger.info("epoch:{}, 保存模型...".format(epoch))
            trainer.save_model('epoch', epoch)

        logger.info("保存最新模型...")
        trainer.save_model('latest', epoch)

        loss_list.append(loss_dict)

        with open(os.path.join(configs['output'], 'loss.json'), 'w', encoding='utf8') as f:
            json.dump(loss_list, f, ensure_ascii=False, indent=4)

# 使用最优模型测试效果
trainer.load_model_opt(os.path.join(configs['output'], 'latext.pth'))
f1 = trainer.val(val_dataloader)

logger.info("result of test's F1 Score:{}".format(f1))


