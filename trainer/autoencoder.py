import torch
import torch.nn as nn
import torch.optim as optim
from datasets.base import BaseDataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from utils.utils import Config, is_notebook
from utils.watchers import LossWatcher

from trainer.base import BaseTrainer

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Trainer(BaseTrainer):
    def __init__(
        self,
        config: Config,
        name: str,
        model: nn.Module,
        dataset: BaseDataset,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
    ):
        super().__init__(config, name, model, dataset, optimizer, lr_scheduler)

    def fit(self):

        # configuration
        self.config.add_logger("train", silent=True)
        self.model.train().to(self.config.train.device)
        global_step = 0
        epochs = self.config.train.epochs
        loss_function = nn.BCELoss()

        # initialize learning rate
        self.optimizer.param_groups[0]["lr"] = self.config.train.lr

        kfold = KFold(n_splits=self.config.train.k_folds, shuffle=True)
        with tqdm(
            enumerate(kfold.split(self.dataset)), total=self.config.train.k_folds, desc=self.iterdesc(fold=0)
        ) as fold_it:
            for fold_index, (train_indices, valid_indices) in fold_it:
                fold_it.set_description(self.iterdesc(fold=fold_index))

                # prepare dataloader
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
                valid_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
                train_dl = DataLoader(self.dataset, batch_size=self.config.train.batch_size, sampler=train_subsampler)
                valid_dl = DataLoader(self.dataset, batch_size=self.config.train.batch_size, sampler=valid_subsampler)

                # Epoch loop
                with tqdm(
                    range(epochs), total=epochs, desc=self.iterdesc(fold=fold_index, epoch=0), leave=False
                ) as epoch_it:
                    for epoch_index in epoch_it:
                        loss_watcher = LossWatcher("train_loss")

                        # Batch loop
                        with tqdm(
                            train_dl, desc=self.iterdesc(fold=fold_index, epoch=epoch_index, batch=0), leave=False
                        ) as batch_it:
                            for batch_index, (x, y) in enumerate(batch_it):

                                x = x.to(self.config.train.device)
                                y = y.to(self.config.train.device)

                                out = self.model(x)

                                self.optimizer.zero_grad()
                                loss = loss_function(out, x)
                                loss.backward()
                                self.optimizer.step()

                                # put logs
                                loss_watcher.put(loss.item())
                                batch_it.set_description(
                                    self.iterdesc(
                                        fold=fold_index, epoch=epoch_index, batch=batch_index, loss=loss.item()
                                    )
                                )

                                # global step for summary writer
                                global_step += 1

                                if batch_index > 0 and batch_index % self.config.train.logging_per_batch == 0:
                                    log = self.iterdesc(
                                        fold=fold_index,
                                        epoch=epoch_index,
                                        batch=batch_index,
                                        batch_total=len(train_dl),
                                        loss=loss.item(),
                                    )
                                    self.config.log.train.info(log)

                        # validation loop
                        with tqdm(
                            valid_dl, desc=self.iterdesc(fold=fold_index, epoch=epoch_index, batch=0), leave=False
                        ) as valid_it:
                            for valid_index, (x, y) in enumerate(batch_it):

                                x = x.to(self.config.train.device)
                                y = y.to(self.config.train.device)
                                out = self.model(x)
                                loss = loss_function(out, x)
                                loss_watcher.put(loss.item())

                                valid_it.set_description(
                                    self.iterdesc(
                                        fold=fold_index, epoch=epoch_index, batch=valid_index, loss=loss.item()
                                    )
                                )

                        # step scheduler
                        self.lr_scheduler.step()

                        # update iteration description
                        epoch_it.set_description(
                            self.iterdesc(fold=fold_index, epoch=epoch_index, loss=loss_watcher.mean)
                        )

                        # logging
                        last_lr = self.lr_scheduler.get_last_lr()[0]
                        log = self.iterdesc(fold=fold_index, epoch=epoch_index, loss=loss_watcher.mean, lr=last_lr)
                        self.config.log.train.info(log)

                        # save best model
                        if loss_watcher.is_best:
                            self.save_model(f"{self.name}_best.pt")

                        # save model regularly
                        if epoch_index % 5 == 0:
                            self.save_model(f"{self.name}_f{fold_index}_e{epoch_index}.pt")
                # end of Epoch
                self.save_model(f"{self.name}_last_at_f{fold_index}.pt")
        # end of k-fold
        self.save_model(f"{self.name}_last.pt")
