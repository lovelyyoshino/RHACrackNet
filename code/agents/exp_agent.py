"""
Capsule agent
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')

import time
import shutil
import logging
from os.path import join
from os import makedirs
from tqdm import tqdm
import pandas as pd
from util.numpy_utils import tensor2numpy
from tensorboardX import SummaryWriter

import torch
from torch import nn
import torchvision

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

        self.logger = logging.getLogger('Exp Agent')

        self.summ_writer = SummaryWriter(log_dir=self.config.env.summ_dir,
                                         comment='Exp Agent')

        self.is_cuda_available = torch.cuda.is_available()
        self.use_cuda = self.is_cuda_available and self.config.env.use_cuda

        if self.use_cuda:
            self.device = torch.device('cuda:' + self.config.env.cuda_id)
            self.logger.info('Agent running on CUDA')
            torch.manual_seed(self.config.env.seed)
            torch.cuda.manual_seed_all(self.config.env.seed)
            print_cuda_statistics()

        else:
            self.device = torch.device('cpu')
            self.logger.info('Agent running on CPU')
            torch.manual_seed(self.config.env.seed)

        self.train_set,_ = get_datasets(self.config.data.dataset_train_name,self.config.data.data_root)
        self.valid_set,_ = get_datasets(self.config.data.dataset_valid_name,self.config.data.data_root)
        self.logger.info('processing: get datasets')
        self.train_loader = DataLoader(dataset=self.train_set,
                              batch_size=self.config.data.batch_size,
                              shuffle=self.config.data.shuffle, sampler=None,
                              batch_sampler=None, num_workers=0,
                              pin_memory=self.config.data.pin_memory,
                              drop_last=self.config.data.drop_last)
        self.valid_loader = DataLoader(dataset=self.valid_set,
                              batch_size=self.config.data.valid_batch_size,
                              shuffle=self.config.data.shuffle, sampler=None,
                              batch_sampler=None, num_workers=0,
                              pin_memory=self.config.data.pin_memory,
                              drop_last=self.config.data.drop_last)
        
        self.logger.info('processing: dataset loader')

        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_metric = 0

        self.model = get_model(self.config)
        self.model = self.model.to(self.device)

        self.loss = get_loss_module(self.config)
        self.logger.info('processing: get loss module')
        self.loss = self.loss.to(self.device)

        self.optimizer = get_optimizer(self.config,
                                       self.model.parameters())

        self.scheduler = get_lr_schedule(self.config,
                                         self.optimizer)

        # try to load existing ckpt to resume a interrupted training
        self.resume(self.config.ckpt.ckpt_name)

    def resume(self, ckpt_name='ckpt.pth'):
        ckpt_path = join(self.config.env.ckpt_dir, ckpt_name)
        try:
            self.load_ckpt(ckpt_path)
        except:
            self.logger.info('Can not load ckpt at "%s"', ckpt_path)

    def load_ckpt(self, ckpt_path, strict=False):
        """
        Load checkpoint with given ckpt_name

        Args:
            ckpt_path (string): the path to ckpt
            strict (bool): whether or not to strictly load ckpt
        """

        try:
            self.logger.info('Loading ckpt from %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device)

            self.current_epoch = ckpt['current_epoch']
            self.current_iteration = ckpt['current_iteration']

            self.model.load_state_dict(ckpt['model_state_dict'], strict=strict)
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

    def save_ckpt(self, ckpt_name='ckpt.pth', is_best=False):
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

        ckpt_path = join(self.config.env.ckpt_dir, ckpt_name)
        torch.save(state, ckpt_path)

        if is_best:
            best_ckpt_path = join(self.config.env.ckpt_dir,
                                  'best_' + ckpt_name)
            shutil.copyfile(ckpt_path, best_ckpt_path)

    def run(self):
        """
        The main operator of agent
        """
        try:
            if self.config.agent.mode == 'valid':
                self.validate()

            elif self.config.agent.mode == 'train':
                self.train()

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
        for i in range(self.current_epoch, self.config.optimizer.max_epoch):
            # train one epoch
            self.train_one_epoch()
            if i %10 == 0:
                valid_epoch_loss, AP = self.validate()

                self.scheduler.step(valid_epoch_loss)

                is_best = AP > self.best_valid_metric
                if is_best:
                    self.best_valid_metric = AP
                self.save_ckpt(is_best=is_best)
    
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

        # init train batch
        tqdm_batch = tqdm(iterable=self.train_loader,
                          total=iteration_per_epoch,
                          desc='Train epoch {}'.format(self.current_epoch))

        # set model into train mode
        self.model.train()

        # initialize average meters
        epoch_loss = AverageMeter()

        epoch_acc = AverageMeter()
        epoch_recall = AverageMeter()
        epoch_precision = AverageMeter()
        epoch_specificity = AverageMeter()
        epoch_f1_score = AverageMeter()
        epoch_iou = AverageMeter()

        epoch_auroc = AverageMeter()
        
        
        
        # stare the training loop over iterations of one epoch
        for images, targets, *_ in tqdm_batch:
            images = images.to(self.device)
            targets = targets.to(self.device)

            # forward propagation
            preds = self.model(images)

            # loss function
            curr_loss = self.loss(preds, targets)

            if torch.isnan(curr_loss):
                self.logger.error('Loss is NaN during training...')
                raise RuntimeError

            # current metrics
            (curr_acc, curr_recall,curr_specificity, curr_precision, 
             curr_f1_score, curr_iou,curr_auroc) = calculate_metrics(self.config, preds, targets,self.device)

            # optimizer
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

        with torch.no_grad():                        
            
            tqdm_batch = tqdm(self.valid_loader,
                              total=iteration_per_epoch,
                              desc='Valid epoch {}'.format(
                                  self.current_epoch))

            # set model into evaluation mode
            self.model.eval()

            # initialize average meters
            epoch_loss = AverageMeter()
            
            fusion_mat_all = torch.empty(0).to(self.device)
            for images, targets, *_ in tqdm_batch:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # forward propagation
                preds = self.model(images)
                
                fusion_mat_t = get_threshold_binary_confusion_matrix(preds,targets,self.device,pixel=2)
                fusion_mat = fusion_mat_t[None,:]
                fusion_mat_all = torch.cat((fusion_mat_all,fusion_mat),0)
                # loss function
                curr_loss = self.loss(preds, targets)

                if torch.isnan(curr_loss):
                    self.logger.error('Loss is NaN during validation...')
                    raise RuntimeError

                # update average meter
                epoch_loss.update(curr_loss.item())

                self.add_visual_log(images, targets, preds,
                                    self.config.metrics.plot_every_epoch)

            tqdm_batch.close()
            
            ODS, OIS, AIU, AP, accuracy,recall,specificity = calculate_metrics_threshold(fusion_mat_all)
            
            self.summ_writer.add_scalar('valid/loss',
                                        epoch_loss.val,
                                        self.current_epoch,
                                        time.time())
            
            self.summ_writer.add_scalar('valid/ODS',
                                        ODS,
                                        self.current_epoch,
                                        time.time())
            self.summ_writer.add_scalar('valid/OIS',
                                        OIS,
                                        self.current_epoch,
                                        time.time())
            self.summ_writer.add_scalar('valid/AIU',
                                        AIU,
                                        self.current_epoch,
                                        time.time())
            self.summ_writer.add_scalar('valid/AP',
                                        AP,
                                        self.current_epoch,
                                        time.time())
            self.summ_writer.add_scalar('valid/accuracy',
                                        accuracy,
                                        self.current_epoch,
                                        time.time())
            
            self.summ_writer.add_scalar('valid/recall',
                                        recall,
                                        self.current_epoch,
                                        time.time())

            self.summ_writer.add_scalar('valid/specificity',
                                        specificity,
                                        self.current_epoch,
                                        time.time())
            self.logger.info('Valid epoch: %d | lr: %f | loss: %f',
                             self.current_epoch,
                             self.optimizer.param_groups[0]['lr'],
                             epoch_loss.val)

            self.logger.info('ODS: %f | OIS: %f | AIU: %f | AP: %f'
                             'accuracy: %f | recall: %f | specificity: %f',
                             ODS, OIS, AIU, AP, accuracy,recall,specificity)

        return epoch_loss.val, AP

    def add_visual_log(self, images, targets, preds, every_epoch=50):
        """
        Add visual plots by summary writer
        """
        def add_plots(summ_writer, images, tag, global_step,
                      nrow=2, dataformats='CHW'):
            images_grid = torchvision.utils.make_grid(
                images, nrow=nrow, padding=2, normalize=True, range=None,
                scale_each=True, pad_value=0
            )
            summ_writer.add_image(tag=tag, img_tensor=images_grid,
                                  global_step=global_step,
                                  walltime=time.time(),
                                  dataformats=dataformats)

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

            analytic = get_analytic_plot(preds, targets, self.device, self.config.metrics.pixel, self.config.metrics.threshold)
            add_plots(summ_writer=self.summ_writer, images=analytic,
                      tag='valid/analytic', global_step=self.current_epoch)

    def inference(self, dataset, num_return, out_folder):

        def save_image(tensor, name, idx, out_dir):
            save_path = join(out_dir, '{}-{:02d}.png'.format(name, idx))
            torchvision.utils.save_image(tensor, save_path)

        out_dir = join(self.config.env.out_dir, out_folder)
        try:
            makedirs(out_dir)
        except:
            pass

        batch_size = 1
        iteration_per_epoch = int(np.ceil(len(dataset) / batch_size))
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False, sampler=None,
            batch_sampler=None, num_workers=0,
            pin_memory=False, drop_last=False,
        )

        with torch.no_grad():
            tqdm_batch = tqdm(data_loader,
                              total=iteration_per_epoch,
                              desc='Inference epoch {}'.format(
                                  self.current_epoch))
            self.model.eval()
            

            #metrics_lists = []
            
            count = 0
            fusion_mat_all = torch.empty(0).to(self.device)
            if num_return == 2:                
                for image, target in tqdm_batch:
                    image = image.to(self.device)
                    target = target.to(self.device)
    
                    # forward propagation
                    prediction = self.model(image)
    
                    fusion_mat_t = get_threshold_binary_confusion_matrix(prediction, 
                                                                         target, self.device, pixel=self.config.metrics.pixel)
                    fusion_mat = fusion_mat_t[None,:]
                    fusion_mat_all = torch.cat((fusion_mat_all,fusion_mat),0)

                    analytic = get_analytic_plot(prediction, target, self.device, self.config.metrics.pixel, 
                                             self.config.metrics.threshold)
                    save_image(image, 'image', count, out_dir)
                    save_image(target, 'target', count, out_dir)
                    save_image(prediction, 'prediction', count, out_dir)
                    save_image(analytic, 'analytic_pixel2', count, out_dir)
                    count += 1
            else:
                for image, target, mask in tqdm_batch:
                    image = image.to(self.device)
                    target = target.to(self.device)
                    mask = mask.to(self.device)
                    # forward propagation
                    prediction = self.model(image)
                    prediction = prediction*mask
                    fusion_mat_t = get_threshold_binary_confusion_matrix(prediction, 
                                                                         target, self.device, pixel=self.config.metrics.pixel)
                    fusion_mat = fusion_mat_t[None,:]
                    fusion_mat_all = torch.cat((fusion_mat_all,fusion_mat),0)

                    analytic = get_analytic_plot(prediction, target, self.device, self.config.metrics.pixel, 
                                             self.config.metrics.threshold)
                    save_image(image, 'image', count, out_dir)
                    save_image(target, 'target', count, out_dir)
                    save_image(prediction, 'prediction', count, out_dir)
                    save_image(analytic, 'analytic_pixel2', count, out_dir)
                    count += 1
                    
            ODS, OIS, AIU, AP, accuracy,recall,specificity = calculate_metrics_threshold(fusion_mat_all)
            
            print('ODS=%.3f, OIS=%.3f, AIU=%.3f, AP=%.3f, accuracy=%.3f, recall=%.3f ,specificity=%.3f.'%(ODS, OIS, AIU, AP, accuracy,recall,specificity))

            performance_df = pd.DataFrame(
                data=[[ODS, OIS, AIU, AP, accuracy,recall,specificity]],
                columns = ['ODS','OIS','AIU','AP','accuracy','recall','specificity']
            )
            performance_csv_path = join(out_dir, 'performance.csv')
            performance_df.to_csv(performance_csv_path)
            
            

    def finalize(self):
        """
        Finalizes all the operations of the operator and the data loader
        """
        self.logger.info('Running finalize operation...')
        self.summ_writer.close()

        self.resume('best_ckpt.pth')

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
        
        train_set,num_train_set_return = get_datasets(self.config.data.dataset_train_name, self.config.data.data_root, transforms=transform_no_aug)
        valid_set,num_valid_set_return = get_datasets(self.config.data.dataset_valid_name, self.config.data.data_root, transforms=transform_no_aug)
        valid2_set,num_valid2_set_return = get_datasets(self.config.data.dataset_valid2_name, self.config.data.data_root)
        self.inference(train_set, num_train_set_return, self.config.data.dataset_train_name)
        self.inference(valid_set, num_valid_set_return, self.config.data.dataset_valid_name)
        self.inference(valid2_set, num_valid2_set_return, self.config.data.dataset_valid2_name)


