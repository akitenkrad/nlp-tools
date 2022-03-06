from typing import Callable, Tuple, Any
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.utils import Config, is_colab, Phase
from utils.watchers import LossWatcher
from datasets.base import BaseDataset

if is_colab():
    from tqdm.notebook import tqdm
    print('running on google colab -> use tqdm.notebook')
else:
    from tqdm import tqdm

class BaseModel(ABC, nn.Module):

    def __init__(self, config:Config, name:str):
        super().__init__()
        self.config = config
        self.config.add_logger('train', silent=True)
        self.name = name

    @abstractmethod
    def build(self):
        '''build a model'''
        raise NotImplementedError()
    
    @abstractmethod
    def step(self, x:torch.Tensor, y:torch.Tensor, loss_func:Callable) -> Tuple[float, torch.Tensor]:
        '''calculate output and loss
        
        Args:
            x (torch.Tensor): input
            y (torch.Tensor): label
            loss_func (Callable): loss function

        Returns:
            loss, output of the model
        '''
        raise NotImplementedError()

    @abstractmethod
    def step_wo_loss(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        '''calculate output without loss
        
        Args:
            x (torch.Tensor): input
            y (torch.Tensor): label

        Returns:
            output of the model
        '''
        raise NotImplementedError()

    def validate(self, epoch:int, valid_dl:DataLoader, loss_func:Callable):
        loss_watcher = LossWatcher('loss')
        with tqdm(valid_dl, total=len(valid_dl), desc=f'[Epoch {epoch:4d} - Validate]', leave=False) as valid_it:
            for x, y in valid_it:
                with torch.no_grad():
                    loss, out = self.step(x, y, loss_func)
                    loss_watcher.put(loss.item())
        return loss_watcher.mean

    def save_model(self, name:str):
        '''save model to config.weights.log_weights_dir
        
        Args:
            name (str): name of the model to save
        '''
        save_dir = Path(self.config.weights.log_weights_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(save_dir / name))

    def forward(self, x):
        '''execute forward process in the PyTorch module'''
        pass

    def fit(self, ds:BaseDataset, optimizer:optim.Optimizer, lr_scheduler:Any, loss_func:Callable):
        '''train the model
        
        Args:
            ds (BaseDataset): dataset.
            optimizer (optim.Optimizer): optimizer.
            lr_scheduler (Any): scheduler for learning rate. ex. optim.lr_scheduler.ExponentialLR.
            loss_func (Callable): loss function
        '''
        self.train().to(self.config.train.device)
        global_step = 0
        tensorboard_dir = self.config.log.log_dir / 'tensorboard' / f'exp_{self.config.now().strftime("%Y%m%d-%H%M%S")}'

        with SummaryWriter(str(tensorboard_dir)) as tb_writer:
            kfold = KFold(n_splits=self.config.train.k_folds, shuffle=True)
            with tqdm(enumerate(kfold.split(ds)), total=self.config.train.k_folds, desc=f'[Fold {0:2d}]') as fold_it:
                for fold, (train_indices, valid_indices) in fold_it:
                    fold_it.set_description(f'[Fold {fold:2d}]')

                    # prepare dataloader
                    train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
                    valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_indices)
                    train_dl = DataLoader(ds, batch_size=self.config.train.batch_size, sampler=train_subsampler)
                    valid_dl = DataLoader(ds, batch_size=self.config.train.batch_size, sampler=valid_subsampler)

                    # initialize learning rate
                    optimizer.param_groups[0]['lr'] = self.config.train.lr

                    # Epoch roop
                    with tqdm(range(self.config.train.epochs), total=self.config.train.epochs, desc=f'[Fold {fold:2d} | Epoch   0]', leave=False) as epoch_it:
                        valid_loss_watcher = LossWatcher('valid_loss', patience=self.config.train.early_stop_patience)
                        for epoch in epoch_it:
                            loss_watcher = LossWatcher('loss')

                            # Batch roop
                            with tqdm(enumerate(train_dl), total=len(train_dl), desc=f'[Epoch {epoch:3d} | Batch {0:3d}]', leave=False) as batch_it:
                                for batch, (x, y) in batch_it:

                                    # process model and calculate loss
                                    loss, out = self.step(x, y, loss_func)

                                    # update parameters
                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()

                                    # put logs
                                    tb_writer.add_scalar('train loss', loss.item(), global_step)
                                    loss_watcher.put(loss.item())
                                    batch_it.set_description(f'[Epoch {epoch:3d} | Batch {batch:3d}] Loss: {loss.item():.3f}')

                                    # global step for summary writer
                                    global_step += 1

                                    if batch > 0 and batch % self.config.train.logging_per_batch == 0:
                                        log = f'[Fold {fold:02d} | Epoch {epoch:03d} | Batch {batch:05d}/{len(train_dl):05d} ({(batch/len(train_dl)) * 100.0:.2f}%)] Loss:{loss.item():.3f}'
                                        self.config.log.train.info(log)

                            # evaluation
                            val_loss = self.validate(epoch, valid_dl, loss_func)

                            # step scheduler
                            lr_scheduler.step()

                            # update iteration description
                            desc = f'[Fold {fold:2d} | Epoch {epoch:3d}] Train Loss: {loss_watcher.mean:.5f} | Valid Loss:{val_loss:.5f}'
                            epoch_it.set_description(desc)

                            # logging
                            last_lr = lr_scheduler.get_last_lr()[0]
                            log = f'[Fold {fold:2d} / Epoch {epoch:3d}] Train Loss: {loss_watcher.mean:.5f} | Valid Loss:{val_loss:.5f} | LR: {last_lr:.7f}'
                            self.config.log.train.info(log)
                            tb_writer.add_scalar('valid loss', val_loss, global_step)

                            # save best model
                            if valid_loss_watcher.is_best:
                                self.save_model(f'{self.name}_best.pt')
                            valid_loss_watcher.put(val_loss)

                            # save model regularly
                            if epoch % 5 == 0:
                                self.save_model(f'{self.name}_f{fold}e{epoch}.pt')

                            # early stopping
                            if valid_loss_watcher.early_stop:
                                self.config.log.train.info(f'====== Early Stopping @epoch: {epoch} @Loss: {valid_loss_watcher.best_score:5.10f} ======')
                                break

                            # backup files
                            if self.config.backup.backup:
                                self.config.log.train.info('start backup process')
                                self.config.backup_logs()
                                self.config.log.train.info(f'finished backup process: backup logs -> {str(Path(self.config.backup.backup_dir).resolve().absolute())}')

                        self.save_model(f'{self.name}_last_f{fold}.pt')

                            # end of Batch
                    # end of Epoch
            # end of k-fold
            self.config.backup_logs()

    def find_lr(self, ds:BaseDataset, optimizer:optim.Optimizer, loss_func:Callable, init_value:float=1e-8, final_value:float=10.0, beta:float=0.98):
        self.train().to(self.config.train.device)
        self.config.add_logger('lr_finder', silent=True)
        ds.to_train()
        dl = DataLoader(ds, batch_size=32)
        num = len(dl) - 1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.0
        best_loss = 0.0
        losses = []
        log_lrs = []

        with tqdm(enumerate(dl), total=len(dl), desc='[B:{:05d}] lr:{:.8f} best_loss:{:.3f}'.format(0, lr, -1)) as it:
            for idx, (x, y) in it:

                # process model and calculate loss
                loss, out = self.step(x, y, loss_func)

                # compute the smoothed loss
                avg_loss = beta * avg_loss + (1-beta) * loss.item()
                smoothed_loss = avg_loss / (1 - beta**(idx+1))

                # stop if the loss is exploding
                if idx > 0 and smoothed_loss > 4 * best_loss:
                    break

                # record the best loss
                if smoothed_loss < best_loss or idx == 0:
                    best_loss = smoothed_loss

                # store the values
                losses.append(smoothed_loss)
                log_lrs.append(lr)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update progress bar
                desc = '[B:{:05d}] lr:{:.10f} best_loss:{:.6f} loss:{:.6f}'.format(idx+1, lr, best_loss, loss.item())
                it.set_description(desc)
                self.config.log.lr_finder.info(desc)

                # update learning rate
                lr *= mult
                optimizer.param_groups[0]['lr'] = lr

            # save figure
            save_path = self.config.log.log_dir / 'lr_finder' / 'lr_loss_curve.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.plot(log_lrs[10:-5], losses[10:-5])
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.savefig(str(save_path))
            self.config.log.lr_finder.info(f'saved -> {str(save_path.resolve().absolute())}')

    def predict(self, ds:BaseDataset, phase:Phase) -> Tuple[torch.Tensor, torch.Tensor]:
        '''predict

        Args:
            ds (BaseDataset): instance of BaseDataset
            phase (Phase): one of phases

        Returns:
            outputs, labels
        '''
        self.eval().to(self.config.train.device)
        ds.to(phase)
        dl = DataLoader(ds, batch_size=self.config.train.batch_size)

        results = []
        labels = []
        for x, y in tqdm(dl, total=len(dl)):
            with torch.no_grad():
                out = self.step_wo_loss(x, y)
            results.append(out.cpu().numpy().copy())
            labels.append(y.cpu().numpy().copy())
        
        results = np.concatenate(results, axis=0)
        labels = np.concatenate(labels, axis=0)
        return results, labels

    def describe(self, ds:BaseDataset):
        x, y = next(iter(DataLoader(ds, batch_size=1)))
        self.config.describe_model(self, input_size=x.shape)
