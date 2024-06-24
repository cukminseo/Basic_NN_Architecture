import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
from torchmetrics.classification import (
    ConfusionMatrix,
    Recall,
    Precision,
    Accuracy,
    F1Score,
)
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler



class Trainer:
    def __init__(self, config, model, device):
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.MultiMarginLoss()  # 손실함수로 MultiMarginLoss 설정

        self.model = model
        self.device = device
        self.model.to(self.device)

        # 손실함수로 adam 설정
        # self.optim = Adam(lr=config.lr, params=self.model.parameters())
        self.optim = optim.SGD(self.model.parameters(), lr=config.lr, momentum=0.9)
        # self.optim = optim.NAdam(lr=config.lr, params=self.model.parameters())

        # # 학습률 스케줄러 추가
        # self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.1, patience=5, verbose=True)

        self.acc = Accuracy(
            num_classes=self.config.num_classes, average="weighted", task="multiclass"
        ).to(device)
        self.precision = Precision(
            num_classes=self.config.num_classes, average="weighted", task="multiclass"
        ).to(device)
        self.recall = Recall(
            num_classes=self.config.num_classes, average="weighted", task="multiclass"
        ).to(device)
        self.f1 = F1Score(
            num_classes=self.config.num_classes, average="weighted", task="multiclass"
        ).to(device)

        self.c_mat = ConfusionMatrix(
            task="multiclass", num_classes=self.config.num_classes
        ).to(device)

    def train(self, epoch, train_dataset: Dataset):
        self.model.train()
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        data_iter = tqdm(train_loader, desc=f"EP:{epoch}_train", total=len(train_loader), bar_format="{l_bar}{r_bar}")

        # 성능 지표를 저장할 리스트 초기화
        avg_loss = []
        avg_acc = []
        avg_precision = []
        avg_recall = []
        avg_f1 = []

        for idx, batch in enumerate(data_iter):
            # 튜플 처리
            inputs, labels = batch

            # 데이터 타입 확인 및 변환
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optim.zero_grad()

            output = self.model(inputs)
            loss = self.criterion(output, labels.long())
            preds = torch.argmax(output, dim=1)
            preds = preds.to(self.device)

            acc = self.acc(preds, labels)
            recall = self.recall(preds, labels)
            precision = self.precision(preds, labels)
            f1 = self.f1(preds, labels)

            avg_loss.append(loss.item())
            avg_acc.append(acc.item())
            avg_recall.append(recall.item())
            avg_precision.append(precision.item())
            avg_f1.append(f1.item())

            loss.backward()
            self.optim.step()

            post_fix = {
                "acc": acc.item(),
                "loss": loss.item(),
            }

            data_iter.set_postfix(post_fix)
            torch.cuda.empty_cache()  # GPU 메모리 해제

        # 에포크별 평균 성능 지표 계산
        avg_loss = np.mean(avg_loss)
        avg_acc = np.mean(avg_acc)
        avg_precision = np.mean(avg_precision)
        avg_recall = np.mean(avg_recall)
        avg_f1 = np.mean(avg_f1)

        return {
            "loss": avg_loss,
            "acc": avg_acc,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
        }

    def eval(self, epoch, val_dataset):
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        data_iter = tqdm(val_loader, desc=f"EP:{epoch}_valid", total=len(val_loader), bar_format="{l_bar}{r_bar}")

        # 에포크별 평균 성능 지표 계산
        avg_loss = []
        avg_acc = []
        avg_precision = []
        avg_recall = []
        avg_f1 = []

        c_mat = None
        for idx, batch in enumerate(data_iter):
            # 튜플 처리
            inputs, labels = batch

            # 데이터 타입 확인 및 변환
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                output = self.model(inputs)
                loss = self.criterion(output, labels.long())
                preds = torch.argmax(output, dim=1)
                preds = preds.to(self.device)

            acc = self.acc(preds, labels)
            recall = self.recall(preds, labels)
            precision = self.precision(preds, labels)
            f1 = self.f1(preds, labels)

            avg_loss.append(loss.item())
            avg_acc.append(acc.item())
            avg_recall.append(recall.item())
            avg_precision.append(precision.item())
            avg_f1.append(f1.item())

            if c_mat is None:
                c_mat = self.c_mat(preds, labels)
            else:
                c_mat += self.c_mat(preds, labels)

        # Add this line to update the progress bar for validation
        post_fix = {
            "acc": acc.item(),
            "loss": loss.item(),
        }
        data_iter.set_postfix(post_fix)
        torch.cuda.empty_cache()  # GPU 메모리 해제

        # 에포크별 평균 성능 지표 계산
        avg_loss = np.mean(avg_loss)
        avg_acc = np.mean(avg_acc)
        avg_precision = np.mean(avg_precision)
        avg_recall = np.mean(avg_recall)
        avg_f1 = np.mean(avg_f1)

        # # 학습률 스케줄러 업데이트
        # self.scheduler.step(avg_loss)

        return {
            "loss": avg_loss,
            "acc": avg_acc,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
        }, c_mat
