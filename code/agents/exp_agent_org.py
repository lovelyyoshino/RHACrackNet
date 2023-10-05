"""
Capsule agent
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')#上一目录

import time
import shutil
import logging
from thop import profile
from os.path import join
from os import makedirs
from tqdm import tqdm
import pandas as pd
from util.numpy_utils import tensor2numpy
from tensorboardX import SummaryWriter

import torch
from torch import nn
import torchvision
# import ttach as tta

import numpy as np
from torch.utils.data import DataLoader

from agents.base_agent import BaseAgent

from dataset.utils.get_datasets import get_datasets
from dataset.utils.get_data_loaders import get_data_loaders

from models.utils.get_model import get_model

from losses.utils.get_losses import get_loss_module

from optimizers.utils.get_lr_schedule import get_lr_schedule
from optimizers.utils.get_optimizer import get_optimizer

from metrics.average_meter import AverageMeter
from metrics.calculate_metrics import calculate_metrics, calculate_metrics_threshold
from metrics.binary_confusion_matrix import get_threshold_binary_confusion_matrix 
from util.get_analytic_plot import get_analytic_plot
from util.print_cuda_statistic import print_cuda_statistics

from torchsummary import summary
# set True for consistence input size only
torch.backends.cudnn.benchmark = True


class ExpAgent(BaseAgent):
    """
    Agent

    Args:
        config (config node object): the given config for the agent
    """

    def __init__(self, config):
        super(ExpAgent, self).__init__(config)

        self.logger = logging.getLogger('Exp Agent')#logging.getLogger(name)方法进行初始化

        self.summ_writer = SummaryWriter(log_dir=self.config.env.summ_dir,
                                         comment='Exp Agent')#设置tensorboardX的访问路径

        self.is_cuda_available = torch.cuda.is_available()#判断是否可以使用cuda
        self.use_cuda = self.is_cuda_available and self.config.env.use_cuda#是否使用cuda

        if self.use_cuda:
            self.device = torch.device('cuda:' + self.config.env.cuda_id)#cuda编号
            self.logger.info('Agent running on CUDA')#输出log
            torch.manual_seed(self.config.env.seed)#随机初始化种子
            torch.cuda.manual_seed_all(self.config.env.seed)#为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
            print_cuda_statistics()

        else:
            self.device = torch.device('cpu')
            self.logger.info('Agent running on CPU')
            torch.manual_seed(self.config.env.seed)

        self.train_set,_ = get_datasets(self.config.data.dataset_train_name,self.config.data.data_root)#导入名称和路径
        self.valid_set,self.num_returns = get_datasets(self.config.data.dataset_valid_name,self.config.data.data_root)
        self.logger.info('processing: get datasets')
        self.train_loader = DataLoader(dataset=self.train_set,
                              batch_size=self.config.data.batch_size,
                              shuffle=self.config.data.shuffle, sampler=None,
                              batch_sampler=None, num_workers=0,
                              pin_memory=self.config.data.pin_memory,
                              drop_last=self.config.data.drop_last)#导入模型
        self.valid_loader = DataLoader(dataset=self.valid_set,
                              batch_size=self.config.data.valid_batch_size,
                              shuffle=self.config.data.shuffle, sampler=None,
                              batch_sampler=None, num_workers=0,
                              pin_memory=self.config.data.pin_memory,
                              drop_last=self.config.data.drop_last)
        
        self.logger.info('processing: dataset loader')

        self.current_epoch = 0#当前的epoch
        self.current_iteration = 0#当前迭代，1个iteration等于使用batchsize个样本训练一次；
        self.best_valid_metric = 0#最佳有效指标

        self.model = get_model(self.config)#获取当前模型
        flops, params = profile(self.model, inputs=(torch.randn(1, 3, 256, 256),), verbose=False)#profile（模型，输入数据）
        # print(model)
        print('Random Path of the Supernet: Params: %.2fM, Flops:%.2fM' % ((params / 1e6), (flops / 1e6)))#输出flops和params
        self.logger.info(self.model)
        unet=self.model.to(self.device)
        # summary(unet,(3,572,572))
        self.model = self.model.to(self.device)#将模型转为cuda版本

        self.loss = get_loss_module(self.config)#获取loss模型
        self.logger.info('processing: get loss module')
        self.loss = self.loss.to(self.device)#将模型转为cuda版本

        self.optimizer = get_optimizer(self.config,
                                       self.model.parameters())#获取优化器

        self.scheduler = get_lr_schedule(self.config,
                                         self.optimizer)#返回学习率计划对象

        # 尝试加载现有的ckpt以恢复中断的训练
        self.resume(self.config.ckpt.ckpt_name)

    def resume(self, ckpt_name='ckpt.pth'):
        ckpt_path = join(self.config.env.ckpt_dir, ckpt_name)#导入模型路径
        try:
            self.load_ckpt(ckpt_path)#判断是否能导入pth文件
        except:
            self.logger.info('Can not load ckpt at "%s"', ckpt_path)

    def load_ckpt(self, ckpt_path, strict=False):#使用给定的ckpt_name加载检查点
        """
        Load checkpoint with given ckpt_name

        Args:
            ckpt_path (string): the path to ckpt
            strict (bool): whether or not to strictly load ckpt
        """

        try:
            self.logger.info('Loading ckpt from %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device)#加载模型

            self.current_epoch = ckpt['current_epoch']#获取ckpt的当前epoch
            self.current_iteration = ckpt['current_iteration']##获取ckpt的当前iteration

            self.model.load_state_dict(ckpt['model_state_dict'], strict=strict)# 接着就可以将模型参数load进模型。
            # NOTE
#            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            self.logger.info('Successfully loaded ckpt from %s at '
                             'epoch %d, iteration %d', ckpt_path,
                             self.current_epoch, self.current_iteration)
            self.logger.info(
                'Loaded initial learning rate %f from ckpt',
                ckpt['optimizer_state_dict']['param_groups'][0]['lr']
            )

        except OSError:
            self.logger.warning('No ckpt exists at "%s". Skipping...',
                                ckpt_path)

    def save_ckpt(self, ckpt_name='ckpt.pth', is_best=False):#将代理程序模型的当前state_dict保存到ckpt_path
        """
        Save the current state_dict of agent model to ckpt_path

        Args:
            ckpt_name (string, optional): the name of the current state_dict to
                 save as
            is_best (bool, optional): indicator for whether the model is best
        """
        state = {'current_epoch': self.current_epoch,
                 'current_iteration': self.current_iteration,
                 'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict()}

        ckpt_path = join(self.config.env.ckpt_dir, ckpt_name)#保存的路径
        torch.save(state, ckpt_path)#保存的信息

        if is_best:
            best_ckpt_path = join(self.config.env.ckpt_dir,
                                  'best_' + ckpt_name)
            shutil.copyfile(ckpt_path, best_ckpt_path)

    def run(self):#导入后启用run
        """
        The main operator of agent
        """
        try:
            if self.config.agent.mode == 'valid':
                self.validate()#显示valid信息

            elif self.config.agent.mode == 'train':
                self.train()#显示train+valid信息

            else:
                self.logger.error('Running mode %s not implemented',
                                  self.config.agent.mode)
                raise NotImplementedError

        except KeyboardInterrupt:
            self.logger.info('Agent interrupted by CTRL+C...')

    def train(self):
        """
        Main training loop
        """
        for i in range(self.current_epoch, self.config.optimizer.max_epoch):#从当前epoch开始训练
            # 训练一个epoch
            self.train_one_epoch()

            valid_epoch_loss, AP = self.validate()#获得预测的结果

            self.scheduler.step(valid_epoch_loss)#lr学习率,当epoch每过scheduler.step时,学习率都变为初始学习率的gamma倍

            is_best = AP > self.best_valid_metric#判断是否AP是最优的
            if is_best:
                self.best_valid_metric = AP
            self.save_ckpt(is_best=is_best)#是否需要保存最优
    
            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        """
        if not self.config.data.drop_last:
            iteration_per_epoch = int(np.ceil(len(self.train_set) /
                                              self.train_loader.batch_size))
        else:
            iteration_per_epoch = int(len(self.train_set) /
                                      self.train_loader.batch_size)

        # 初始训练批次
        tqdm_batch = tqdm(iterable=self.train_loader,
                          total=iteration_per_epoch,
                          desc='Train epoch {}'.format(self.current_epoch))

        # 将模型设置为训练模式
        self.model.train()#必须要

        # initialize average meters
        epoch_loss = AverageMeter()

        epoch_acc = AverageMeter()
        epoch_recall = AverageMeter()
        epoch_precision = AverageMeter()
        epoch_specificity = AverageMeter()
        epoch_f1_score = AverageMeter()
        epoch_iou = AverageMeter()

        epoch_auroc = AverageMeter()
        
        
        
        # 在一个时期的迭代中训练循环
        for images, targets, *_ in tqdm_batch:
            images = images.to(self.device)
            targets = targets.to(self.device)

            # 前向传播
            preds = self.model(images)#导入模型，到forward
            #print(images.shape,preds.shape,">>>>>",targets.shape)
            # 损失函数
            curr_loss = self.loss(preds, targets)

            if torch.isnan(curr_loss):
                self.logger.error('Loss is NaN during training...')
                raise RuntimeError

            # 当前指标
            (curr_acc, curr_recall,curr_specificity, curr_precision, 
             curr_f1_score, curr_iou,curr_auroc) = calculate_metrics(self.config, preds, targets,self.device)

            # 优化器
            self.optimizer.zero_grad()
            curr_loss.backward()
            self.optimizer.step()

            # update average meter
            epoch_loss.update(curr_loss.item())

            epoch_acc.update(curr_acc)
            epoch_recall.update(curr_recall)
            epoch_precision.update(curr_precision)
            epoch_specificity.update(curr_specificity)
            epoch_f1_score.update(curr_f1_score)
            epoch_iou.update(curr_iou)

            epoch_auroc.update(curr_auroc)

            self.current_iteration += 1

        tqdm_batch.close()
        self.summ_writer.add_scalar('train/learning_rate',
                                    self.optimizer.param_groups[0]['lr'],
                                    self.current_epoch, time.time())

        self.summ_writer.add_scalar('train/loss', epoch_loss.val,
                                    self.current_epoch, time.time())

        self.summ_writer.add_scalar('train/accuracy', epoch_acc.val,
                                    self.current_epoch, time.time())
        self.summ_writer.add_scalar('train/recall', epoch_recall.val,
                                    self.current_epoch, time.time())
        self.summ_writer.add_scalar('train/precision', epoch_precision.val,
                                    self.current_epoch, time.time())
        self.summ_writer.add_scalar('train/specificity', epoch_specificity.val,
                                    self.current_epoch, time.time())
        self.summ_writer.add_scalar('train/f1_score', epoch_f1_score.val,
                                    self.current_epoch, time.time())
        self.summ_writer.add_scalar('train/IOU', epoch_iou.val,
                                    self.current_epoch, time.time())
        self.summ_writer.add_scalar('train/AUROC', epoch_auroc.val,
                                    self.current_epoch, time.time())

        self.logger.info('Train epoch: %d | lr: %f | loss: %f',
                         self.current_epoch,
                         self.optimizer.param_groups[0]['lr'],
                         epoch_loss.val)
        self.logger.info('specificity: %f | recall: %f | precision: %f'
                         'f1_score: %f | IOU: %f | AUROC: %f',
                         epoch_specificity.val,
                         epoch_recall.val, epoch_precision.val,
                         epoch_f1_score.val, epoch_iou.val,
                         epoch_auroc.val)


    


    def validate(self):
        """
        Model validation
        """
        if not self.config.data.drop_last:
            iteration_per_epoch = int(np.ceil(len(self.valid_set) /
                                              self.valid_loader.batch_size))
        else:
            iteration_per_epoch = int(len(self.valid_set) /
                                      self.valid_loader.batch_size)

        with torch.no_grad():#不要求计算梯度                    
            
            tqdm_batch = tqdm(self.valid_loader,
                              total=iteration_per_epoch,
                              desc='Valid epoch {}'.format(
                                  self.current_epoch))#Tqdm 是 Python 进度条库

            # 将模型设置为评估模式
            self.model.eval()#当用于inference时不要忘记添加model.eval()
            # self.tta_model = tta.SegmentationTTAWrapper(self.model, tta.aliases.d4_transform(), merge_mode='mean')#添加的TTA策略
            # train函数是模型训练的入口。首先一些变量的更新采用自定义的AverageMeter类来管理，后面会介绍该类的定义。然后model.train()是设置为训练模式。
            epoch_loss = AverageMeter()
            epoch_acc = AverageMeter()
            epoch_recall = AverageMeter()
            epoch_precision = AverageMeter()
            epoch_specificity = AverageMeter()
            epoch_f1_score = AverageMeter()
            epoch_iou = AverageMeter()

            epoch_auroc = AverageMeter()
            
            fusion_mat_all = torch.empty(0).to(self.device)
            if self.num_returns == 2:
                for images, targets, *_ in tqdm_batch:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    # 前向传播得到的预测模型
                    preds = self.model(images)  # 导入模型，到forward
                    # preds = self.tta_model(images)
                    (curr_acc, curr_recall,curr_specificity, curr_precision,
                        curr_f1_score, curr_iou,curr_auroc) = calculate_metrics(self.config, preds, targets,self.device)#获得对应的评价指标
                    fusion_mat_t = get_threshold_binary_confusion_matrix(preds,targets,self.device,pixel=2)
                    fusion_mat = fusion_mat_t[None,:]
                    fusion_mat_all = torch.cat((fusion_mat_all,fusion_mat),0)
                    # 损失函数
                    curr_loss = self.loss(preds, targets)
                    
                    
                    if torch.isnan(curr_loss):
                        self.logger.error('Loss is NaN during validation...')
                        raise RuntimeError

                    # 更新平均参数
                    epoch_loss.update(curr_loss.item())
                    epoch_acc.update(curr_acc)
                    epoch_recall.update(curr_recall)
                    epoch_precision.update(curr_precision)
                    epoch_specificity.update(curr_specificity)
                    epoch_f1_score.update(curr_f1_score)
                    epoch_iou.update(curr_iou)
        
                    epoch_auroc.update(curr_auroc)
                    self.add_visual_log(images, targets, preds,
                                        self.config.metrics.plot_every_epoch)#通过summ_writer添加视觉图
            else:
                for images, targets,mask in tqdm_batch:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    mask = mask.to(self.device)

                    # 前向传播得到的预测模型
                    #preds = self.tta_model(images)
                    preds = self.model(images)
                    preds = preds*mask
                    (curr_acc, curr_recall,curr_specificity, curr_precision, 
                        curr_f1_score, curr_iou,curr_auroc) = calculate_metrics(self.config, preds, targets,self.device)#获得对应的评价指标
                    fusion_mat_t = get_threshold_binary_confusion_matrix(preds,targets,self.device,pixel=2)
                    fusion_mat = fusion_mat_t[None,:]
                    fusion_mat_all = torch.cat((fusion_mat_all,fusion_mat),0)
                    # 损失函数
                    curr_loss = self.loss(preds, targets)
                    
                    
                    if torch.isnan(curr_loss):
                        self.logger.error('Loss is NaN during validation...')
                        raise RuntimeError

                    # 更新平均参数
                    epoch_loss.update(curr_loss.item())
                    epoch_acc.update(curr_acc)
                    epoch_recall.update(curr_recall)
                    epoch_precision.update(curr_precision)
                    epoch_specificity.update(curr_specificity)
                    epoch_f1_score.update(curr_f1_score)
                    epoch_iou.update(curr_iou)
        
                    epoch_auroc.update(curr_auroc)
                    self.add_visual_log(images, targets, preds,
                                        self.config.metrics.plot_every_epoch)#通过summ_writer添加视觉图

            tqdm_batch.close()
            
            ODS, OIS, AIU, AP, accuracy,recall,specificity,dsc = calculate_metrics_threshold(fusion_mat_all)
            self.summ_writer.add_scalar('valid/loss', epoch_loss.val,
                                    self.current_epoch, time.time())

            self.summ_writer.add_scalar('valid/accuracy', epoch_acc.val,
                                        self.current_epoch, time.time())
            self.summ_writer.add_scalar('valid/recall', epoch_recall.val,
                                        self.current_epoch, time.time())
            self.summ_writer.add_scalar('valid/precision', epoch_precision.val,
                                        self.current_epoch, time.time())
            self.summ_writer.add_scalar('valid/specificity', epoch_specificity.val,
                                        self.current_epoch, time.time())
            self.summ_writer.add_scalar('valid/f1_score', epoch_f1_score.val,
                                        self.current_epoch, time.time())
            self.summ_writer.add_scalar('valid/IOU', epoch_iou.val,
                                        self.current_epoch, time.time())
            self.summ_writer.add_scalar('valid/AUROC', epoch_auroc.val,
                                        self.current_epoch, time.time())
            self.summ_writer.add_scalar('valid/ODS', ODS,
                                        self.current_epoch, time.time())
            self.summ_writer.add_scalar('valid/OIS', OIS,
                                        self.current_epoch, time.time())
            self.summ_writer.add_scalar('valid/AP', AP,
                                        self.current_epoch, time.time())
            self.summ_writer.add_scalar('valid/DSC', dsc,
                                        self.current_epoch, time.time())
            self.logger.info('Valid epoch: %d | lr: %f | loss: %f',
                                 self.current_epoch,
                                 self.optimizer.param_groups[0]['lr'],
                                 epoch_loss.val)

            self.logger.info('specificity: %f | acc: %f |recall: %f | precision: %f'
                         'f1_score: %f | IOU: %f | AUROC: %f| ODS: %f| OIS: %f| AP: %f | DSC: %f',
                         epoch_specificity.val,epoch_acc.val,
                         epoch_recall.val, epoch_precision.val,
                         epoch_f1_score.val, epoch_iou.val,
                         epoch_auroc.val,ODS,OIS,AP,dsc)

        return epoch_loss.val, epoch_f1_score.val

    def add_visual_log(self, images, targets, preds, every_epoch=50):
        """
        Add visual plots by summary writer
        """
        def add_plots(summ_writer, images, tag, global_step,
                      nrow=2, dataformats='CHW'):
            images_grid = torchvision.utils.make_grid(
                images, nrow=nrow, padding=2, normalize=True, range=None,
                scale_each=True, pad_value=0
            )#make_grid的作用是将若干幅图像拼成一幅图像，显示分割结果
            summ_writer.add_image(tag=tag, img_tensor=images_grid,
                                  global_step=global_step,
                                  walltime=time.time(),
                                  dataformats=dataformats)#使用summ_writer.add_image添加图片

        if (self.current_epoch % every_epoch == 0 or
                self.config.agent.mode == 'valid'):

            add_plots(summ_writer=self.summ_writer, images=images,
                      tag='valid/images',
                      global_step=self.current_epoch)

            add_plots(summ_writer=self.summ_writer, images=targets,
                      tag='valid/targets',
                      global_step=self.current_epoch)

            add_plots(summ_writer=self.summ_writer, images=preds,
                      tag='valid/preds', global_step=self.current_epoch)

            analytic = get_analytic_plot(preds, targets, self.device, self.config.metrics.pixel, self.config.metrics.threshold)#得到分析图片，并将false_positive, true_positive, false_negative拼接
            add_plots(summ_writer=self.summ_writer, images=analytic,
                      tag='valid/analytic', global_step=self.current_epoch)

    def inference(self, dataset, num_return, out_folder):

        def save_image(tensor, name, idx, out_dir):
            save_path = join(out_dir, '{}-{:02d}.png'.format(name, idx))#导入路径
            torchvision.utils.save_image(tensor, save_path)#保存路径

        out_dir = join(self.config.env.out_dir, out_folder)#self.config.env.out_dir在get_config.py中申明
        out_dir_image = join(self.config.env.out_dir, out_folder,'image')
        out_dir_prediction = join(self.config.env.out_dir, out_folder,'prediction')
        out_dir_analytic = join(self.config.env.out_dir, out_folder,'analytic')
        out_dir_target = join(self.config.env.out_dir, out_folder,'target')
        try:
            makedirs(out_dir)
            makedirs(out_dir_image)
            makedirs(out_dir_prediction)
            makedirs(out_dir_analytic)
            makedirs(out_dir_target)
        except:
            pass

        batch_size = 1
        iteration_per_epoch = int(np.ceil(len(dataset) / batch_size))#每一个epoch中的iteration
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False, sampler=None,
            batch_sampler=None, num_workers=0,
            pin_memory=False, drop_last=False,
        )#加载路径

        with torch.no_grad():
            tqdm_batch = tqdm(data_loader,
                              total=iteration_per_epoch,
                              desc='Inference epoch {}'.format(
                                  self.current_epoch))
            self.model.eval()
            

            #metrics_lists = []
            
            count = 0
            
            epoch_acc = AverageMeter()
            epoch_recall = AverageMeter()
            epoch_precision = AverageMeter()
            epoch_specificity = AverageMeter()
            epoch_f1_score = AverageMeter()
            epoch_iou = AverageMeter()
    
            epoch_auroc = AverageMeter()
            fusion_mat_all = torch.empty(0).to(self.device)
            if num_return == 2:
                for image, target,*_ in tqdm_batch:
                    #print(image.dtype)
                    image = image.to(self.device)
                    target = target.to(self.device)
                    # self.tta_model = tta.SegmentationTTAWrapper(self.model, tta.aliases.d4_transform(), merge_mode='mean')#添加的TTA策略
                    # 前向传播
                    prediction = self.model(image)
                    # prediction = self.tta_model(image)
                    (curr_acc, curr_recall,curr_specificity, curr_precision, 
                        curr_f1_score, curr_iou,curr_auroc) = calculate_metrics(self.config, prediction, target,self.device)
                    fusion_mat_t = get_threshold_binary_confusion_matrix(prediction,target,self.device,pixel=2)
                    fusion_mat = fusion_mat_t[None,:]
                    fusion_mat_all = torch.cat((fusion_mat_all,fusion_mat),0)
                    epoch_acc.update(curr_acc)
                    epoch_recall.update(curr_recall)
                    epoch_precision.update(curr_precision)
                    epoch_specificity.update(curr_specificity)
                    epoch_f1_score.update(curr_f1_score)
                    epoch_iou.update(curr_iou)
        
                    epoch_auroc.update(curr_auroc)
            
                    analytic = get_analytic_plot(prediction, target, self.device, self.config.metrics.pixel, 
                                            self.config.metrics.threshold)
                    save_image(image, 'image', count, out_dir_image)
                    save_image(target, 'target', count, out_dir_target)
                    save_image(prediction, 'prediction', count, out_dir_prediction)
                    save_image(analytic, 'analytic', count, out_dir_analytic)
                    count += 1

            else:
                for image, target,mask in tqdm_batch:
                    #print(image.dtype)
                    image = image.to(self.device)
                    target = target.to(self.device)
                    mask = mask.to(self.device)
                    # self.tta_model = tta.SegmentationTTAWrapper(self.model, tta.aliases.d4_transform(), merge_mode='mean')#添加的TTA策略
                    # 前向传播
                    prediction = self.model(image)
                    # prediction = self.tta_model(image)
                    prediction = prediction*mask
                    (curr_acc, curr_recall,curr_specificity, curr_precision, 
                        curr_f1_score, curr_iou,curr_auroc) = calculate_metrics(self.config, prediction, target,self.device)
                    fusion_mat_t = get_threshold_binary_confusion_matrix(prediction,target,self.device,pixel=2)
                    fusion_mat = fusion_mat_t[None,:]
                    fusion_mat_all = torch.cat((fusion_mat_all,fusion_mat),0)
                    epoch_acc.update(curr_acc)
                    epoch_recall.update(curr_recall)
                    epoch_precision.update(curr_precision)
                    epoch_specificity.update(curr_specificity)
                    epoch_f1_score.update(curr_f1_score)
                    epoch_iou.update(curr_iou)
        
                    epoch_auroc.update(curr_auroc)
            
                    analytic = get_analytic_plot(prediction, target, self.device, self.config.metrics.pixel, 
                                            self.config.metrics.threshold)
                    save_image(image, 'image', count, out_dir_image)
                    save_image(target, 'target', count, out_dir_target)
                    save_image(prediction, 'prediction', count, out_dir_prediction)
                    save_image(analytic, 'analytic', count, out_dir_analytic)
                    count += 1

                
            ODS, OIS, AIU, AP, accuracy,recall,specificity,dsc = calculate_metrics_threshold(fusion_mat_all)
            self.logger.info('specificity: %f | acc: %f | recall: %f | precision: %f'
                        'f1_score: %f | IOU: %f | AUROC: %f| ODS: %f| OIS: %f| AP: %f | DSC: %f |',
                        epoch_specificity.val,epoch_acc.val,
                        epoch_recall.val, epoch_precision.val,
                        epoch_f1_score.val, epoch_iou.val,
                        epoch_auroc.val,ODS,OIS,AP,dsc)
            
            performance_df = pd.DataFrame(
                data=[[epoch_acc.val, epoch_recall.val,epoch_specificity.val, epoch_precision.val, 
                    epoch_f1_score.val, epoch_iou.val,epoch_auroc.val]],
                columns=['acc', 'recall',
                        'specificity', 'precision', 'f1_score', 'iou',
                        'auroc']

            )
            performance_csv_path = join(out_dir, 'performance.csv')
            performance_df.to_csv(performance_csv_path)
            
            

    def finalize(self):#完成操作员和数据加载器的所有操作
        """
        Finalizes all the operations of the operator and the data loader
        """
        self.logger.info('Running finalize operation...')
        self.summ_writer.close()

        self.resume('best_ckpt.pth')#重新加载best_ckpt.pth

        def transform_no_aug(image, annot, split_mode=None):
#            image = torchvision.transforms.functional.resize(
#                image, size=(512, 512))
#            annot = torchvision.transforms.functional.resize(
#                annot, size=(512, 512))
#            annot_aux = torchvision.transforms.functional.resize(
#                annot_aux, size=(512, 512))

            image = torchvision.transforms.functional.to_tensor(image)
            annot = torchvision.transforms.functional.to_tensor(annot)
            #annot_aux = torchvision.transforms.functional.to_tensor(annot_aux)

            return image, annot
        
        train_set,num_train_set_return = get_datasets(self.config.data.dataset_train_name, self.config.data.data_root)#对应dataset和图片数目
        valid_set,num_valid_set_return = get_datasets(self.config.data.dataset_valid_name, self.config.data.data_root)

        self.inference(train_set, num_train_set_return, self.config.data.dataset_train_name)
        self.inference(valid_set, num_valid_set_return, self.config.data.dataset_valid_name)



