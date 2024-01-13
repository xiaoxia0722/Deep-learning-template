# encoding: utf8
# @Author: qiaoyongchao
# @Email: qiaoyc2023@163.com

from src.models import *
from src.dataset import *
from src.losses import *

import time

import torch
from thop import profile


class Trainer:
    def __init__(self, configs, device, logger):
        """
        模型训练器，主要包括模型的创建、优化器的创建、模型的训练、验证等
        :param configs:
        :param device:
        :param logger:
        """
        self.configs = configs
        self.device = device
        self.logger = logger
        self.model = None
        self.opt = None
        logger.info("初始化模型...")
        self.init_model()

        logger.info("初始化优化器...")
        self.init_opt()
        self.criterion = nn.CrossEntropyLoss()

    def init_model(self):
        """
        初始化模型
        :return:
        """
        self.model = Model()

    def init_opt(self):
        """
        初始化优化器
        :return:
        """
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.configs['opt']['lr'], betas=(self.configs['opt']['beta1'], self.configs['opt']['beta2']))

    def train(self, epoch, train_loader):
        """
        模型训练并输出训练过程中的信息
        :param epoch:
        :param train_loader:
        :return:
        """
        lens = len(train_loader)
        start = time.time()
        losses = []
        for step, batch in enumerate(train_loader):
            imgs, label = batch

            pred = self.model(imgs)

            loss = self.criterion(pred, imgs)
            self.opt.zero_grad()

            loss.backward()

            if step % self.configs['print_step'] == 0:
                end = time.time()
                all_time = end - start
                self.logger.info("epoch:{}[step:{}/{}] => loss:{}, 当前epoch耗时为:{}min, 预计总耗时为:{}min, 剩余耗时为:{}min".format(epoch, step, lens, loss.item(), all_time / 60, all_time / (step + 1) * lens / 60, all_time / (step + 1) * (lens - step - 1) / 60))
                losses.append(loss.item())
        return {
            'losses': losses,
            'time': time.time() - start
        }

    def val(self, val_dataloader):
        """
        模型验证并返回验证结果
        :param val_dataloader:
        :return:
        """
        torch.cuda.empty_cache()
        f1 = None   # f1的计算方式
        return f1

    def load_model_opt(self, path):
        """
        模型和优化器加载
        :param path:
        :return:
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.model = load_model_weights(self.model, checkpoint['model'], self.configs['many_gpus'])
        self.opt = load_opt_weights(self.opt, checkpoint['optimizers'])

    def save_model(self, mode, epoch):
        """
        保存模型
        :param mode:
        :param epoch:
        :return:
        """
        state = {'model': self.model.state_dict(),
                 'optimizers': self.opt.state_dict(),
                 'epoch': epoch}
        if mode == 'epoch':
            torch.save(state, '%s/state_epoch_%03d.pth' % (self.configs['output'], epoch))
        else:
            torch.save(state, '%s/state_%s.pth' % (self.configs['output'], mode))

    def get_params(self):
        """
        计算模型params和flops
        :return:
        """
        with torch.no_grad():
            imgs = torch.randn((1, 3, 256, 256)).to(self.device)
            flops, params = profile(self.model, inputs=(imgs))

        return {
            'params': params,
            'flops': flops
        }
