"""PyTorch trainer module.

- Author: Jongkuk Lim, Junghoon Kim
- Contact: lim.jeikei@gmail.com, placidus36@gmail.com
"""

import os
import shutil
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import optuna
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from tqdm import tqdm

from src.utils.torch_utils import save_model

from src.sam import SAM

def knowledge_distillation_loss(logits, labels, teacher_logits):
        alpha = 0.3
        T = 15
        
        student_loss = F.cross_entropy(input=logits, target=labels)
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits/T, dim=1), F.softmax(teacher_logits/T, dim=1)) * (T * T)
        total_loss =  (1. - alpha)*student_loss + alpha*distillation_loss
 
        return total_loss

def _get_n_data_from_dataloader(dataloader: DataLoader) -> int:
    """Get a number of data in dataloader.

    Args:
        dataloader: torch dataloader

    Returns:
        A number of data in dataloader
    """
    if isinstance(dataloader.sampler, SubsetRandomSampler):
        n_data = len(dataloader.sampler.indices)
    elif isinstance(dataloader.sampler, SequentialSampler):
        n_data = len(dataloader.sampler.data_source)
    else:
        n_data = len(dataloader) * dataloader.batch_size if dataloader.batch_size else 1

    return n_data


def _get_n_batch_from_dataloader(dataloader: DataLoader) -> int:
    """Get a batch number in dataloader.

    Args:
        dataloader: torch dataloader

    Returns:
        A batch number in dataloader
    """
    n_data = _get_n_data_from_dataloader(dataloader)
    n_batch = dataloader.batch_size if dataloader.batch_size else 1

    return n_data // n_batch


def _get_len_label_from_dataset(dataset: Dataset) -> int:
    """Get length of label from dataset.

    Args:
        dataset: torch dataset

    Returns:
        A number of label in set.
    """
    if isinstance(dataset, torchvision.datasets.ImageFolder) or isinstance(
        dataset, torchvision.datasets.vision.VisionDataset
    ):
        return len(dataset.classes)
    elif isinstance(dataset, torch.utils.data.Subset):
        return _get_len_label_from_dataset(dataset.dataset)
    else:
        raise NotImplementedError


class TorchTrainer:
    """Pytorch Trainer."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        hyperparams: dict,
        model_path: str,
        scaler=None,
        device: torch.device = "cpu",
        verbose: int = 1,
        trial: Optional[optuna.trial.Trial] = None,
        flag = False
    ) -> None:
        """Initialize TorchTrainer class.

        Args:
            model: model to train
            criterion: loss function module
            optimizer: optimization module
            device: torch device
            verbose: verbosity level.
        """
        # taco.yaml을 사용한 경우 (hyperparameter search를 하지 않은 경우)
        if "optimizer" not in hyperparams.keys():
            hyperparams["optimizer"] = "sgd"
            hyperparams["scheduler"] = "onecycle"

        # define optimizer & scheduler
        if hyperparams["optimizer"] == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=hyperparams["INIT_LR"], momentum=0.9, weight_decay=5e-4)
            self.sam = SAM(model.parameters(), optim.SGD, lr=hyperparams["INIT_LR"], momentum=0.9)
            self.sam_flag = flag
            print(f'use sam !!! : {self.sam_flag}')
        elif hyperparams["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=hyperparams["INIT_LR"], weight_decay=5e-4)
        elif hyperparams["optimizer"] == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=hyperparams["INIT_LR"], weight_decay=5e-4)
        else : 
            optimizer = hyperparams["optimizer"]

        if hyperparams["scheduler"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams['EPOCHS'])
        elif hyperparams["scheduler"] == "reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold_mode='abs',min_lr=1e-6)
        elif hyperparams["scheduler"] == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hyperparams["INIT_LR"],
                                                            steps_per_epoch=20851//hyperparams["BATCH_SIZE"],
                                                            epochs=hyperparams["EPOCHS"], pct_start=0.05)
        elif hyperparams["scheduler"] == "None":
            scheduler = None
        else : 
            scheduler = hyperparams["scheduler"]
        
        print(hyperparams)

        self.model = model
        self.model_path = model_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.verbose = verbose
        self.device = device
        self.trial = trial

    def train(
        self,
        train_dataloader: DataLoader,
        n_epoch: int,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[float, float]:
        """Train model.
        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation

        Returns:
            loss and accuracy
        """
        best_test_acc = -1.0
        best_test_f1 = -1.0
        num_classes = _get_len_label_from_dataset(train_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        for epoch in range(n_epoch):
            running_loss, correct, total = 0.0, 0, 0
            preds, gt = [], []
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            self.model.train()
            for batch, (data, labels) in pbar:
                data, labels = data.to(self.device), labels.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                outputs = torch.squeeze(outputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.sam_flag :
                        self.sam.first_step(zero_grad=True)
                        self.criterion(self.model(data), labels).backward()
                        self.sam.second_step(zero_grad=True)
                    else:
                        self.optimizer.step()

                

                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                preds += pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                    f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
                )
            pbar.close()

            _, test_f1, test_acc = self.test(
                model=self.model, test_dataloader=val_dataloader
            )

            if self.trial is not None:
                self.trial.report(test_f1, epoch)
                if self.trial.should_prune():
                    print("Unpromising Trial.")
                    raise optuna.TrialPruned()
                # print("Promising Trial!")

            if self.scheduler is not None:
                if self.scheduler == "cosine":
                    self.scheduler.step()
                elif self.scheduler == "reduce":
                    self.scheduler.step(test_f1)
            
            if best_test_f1 > test_f1:
                continue
            best_test_acc = test_acc
            best_test_f1 = test_f1
            print(f"Model saved. Current best test f1: {best_test_f1:.3f}")
            save_model(
                model=self.model,
                path=self.model_path,
                data=data,
                device=self.device,
            )

        return best_test_acc, best_test_f1

    def KD_train(
        self,
        train_dataloader: DataLoader,
        n_epoch: int,
        val_dataloader: Optional[DataLoader] = None,
        teacher : nn.Module = None
    ) -> Tuple[float, float]:
        """Train model.

        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation

        Returns:
            loss and accuracy
        """
        best_test_acc = -1.0
        best_test_f1 = -1.0
        num_classes = _get_len_label_from_dataset(train_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        for epoch in range(n_epoch):
            running_loss, correct, total = 0.0, 0, 0
            preds, gt = [], []
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            
            self.model.train()
            teacher.eval()
            
            for batch, (data, labels) in pbar:
                data, labels = data.to(self.device), labels.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        teacher_outputs = teacher(data)
                        student_outputs = self.model(data)
                else:
                    teacher_outputs = teacher(data)
                    student_outputs = self.model(data)
                student_outputs = torch.squeeze(student_outputs)
                teacher_outputs = torch.squeeze(teacher_outputs)

                loss = knowledge_distillation_loss(student_outputs, labels, teacher_outputs)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                _, pred = torch.max(student_outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                preds += pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                    f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
                )
            pbar.close()

            _, test_f1, test_acc = self.test(
                model=self.model, test_dataloader=val_dataloader
            )

            if self.trial is not None:
                self.trial.report(test_f1, epoch)
                if self.trial.should_prune():
                    print("Unpromising Trial.")
                    raise optuna.TrialPruned()
                # print("Promising Trial!")

            if self.scheduler is not None:
                if self.scheduler == "cosine":
                    self.scheduler.step()
                elif self.scheduler == "reduce":
                    self.scheduler.step(test_f1)
            
            if best_test_f1 > test_f1:
                continue
            best_test_acc = test_acc
            best_test_f1 = test_f1
            print(f"Model saved. Current best test f1: {best_test_f1:.3f}")
            save_model(
                model=self.model,
                path=self.model_path,
                data=data,
                device=self.device,
            )

        return best_test_acc, best_test_f1

    @torch.no_grad()
    def test(
        self, model: nn.Module, test_dataloader: DataLoader
    ) -> Tuple[float, float, float]:
        """Test model.

        Args:
            test_dataloader: test data loader module which is a iterator that returns (data, labels)

        Returns:
            loss, f1, accuracy
        """

        n_batch = _get_n_batch_from_dataloader(test_dataloader)

        running_loss = 0.0
        preds = []
        gt = []
        correct = 0
        total = 0

        num_classes = _get_len_label_from_dataset(test_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        model.to(self.device)
        model.eval()
        for batch, (data, labels) in pbar:
            data, labels = data.to(self.device), labels.to(self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
            else:
                outputs = model(data)
            outputs = torch.squeeze(outputs)
            running_loss += self.criterion(outputs, labels).item()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()
            pbar.update()
            pbar.set_description(
                f" Val: {'':5} Loss: {(running_loss / (batch + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
                f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
            )
        loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        f1 = f1_score(
            y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0
        )
        return loss, f1, accuracy


def count_model_params(
    model: torch.nn.Module,
) -> int:
    """Count model's parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
