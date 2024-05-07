import torch
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


class Trainer:
    def __init__(self, config, model, device):
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

        self.model = model
        self.device = device
        self.model.to(self.device)

        # 손실함수로 adam 설정
        self.optim = Adam(lr=config.lr, params=self.model.parameters())

        self.acc = Accuracy(
            num_classes=self.config.num_classes, average="macro", task="multiclass"
        ).to(device)
        self.precision = Precision(
            num_classes=self.config.num_classes, average="macro", task="multiclass"
        ).to(device)
        self.recall = Recall(
            num_classes=self.config.num_classes, average="macro", task="multiclass"
        ).to(device)
        self.f1 = F1Score(
            num_classes=self.config.num_classes, average="macro", task="multiclass"
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
            if isinstance(self.model, nn.DataParallel):
                primary_device = f"cuda:{self.model.device_ids[0]}"
            else:
                primary_device = self.device

            # 딕셔너리 처리
            #batch = {k: v.to(primary_device) for k, v in batch.items()}
            # 튜플 처리
            inputs, labels = batch
            inputs, labels = inputs.to(primary_device), labels.to(primary_device)

            self.optim.zero_grad()
            # 기존 코드 (잘못된 접근)
            # batch_inputs = batch_inputs.unsqueeze(1)

            # 변경 후
            inputs, labels = batch
            batch_inputs = inputs.to(primary_device)
            batch_labels = labels.to(primary_device)
            # batch_inputs = inputs.unsqueeze(1)  # 필요한 경우

            output = self.model(batch_inputs)

            loss = self.criterion(output, batch_labels.long())
            preds = torch.argmax(output, dim=1)

            acc = self.acc(preds, batch_labels)
            recall = self.recall(preds, batch_labels)
            precision = self.precision(preds, batch_labels)
            f1 = self.f1(preds, batch_labels)

            avg_loss.append(loss.item())
            avg_acc.append(acc.item())
            avg_recall.append(recall.item())
            avg_precision.append(precision.item())
            avg_f1.append(f1.item())

            loss.backward()
            self.optim.step()

            post_fix = {
                "loss": loss.item(),
                "acc": acc.item(),
                "precision": precision.item(),
                "recall": recall.item(),
                "f1-score": f1.item(),
            }

            data_iter.set_postfix(post_fix)

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
         def eval(self, epoch, val_dataset):
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        data_iter = tqdm(val_loader, desc=f"EP:{epoch}_valid", total=len(val_loader))

        # 에포크별 평균 성능 지표 계산
        avg_loss = []
        avg_acc = []
        avg_precision = []
        avg_recall = []
        avg_f1 = []

        c_mat = None
        for idx, batch in enumerate(data_iter):
            if isinstance(self.model, nn.DataParallel):
                primary_device = f"cuda:{self.model.device_ids[0]}"
            else:
                primary_device = self.device

            # 딕셔너리 처리
            #batch = {k: v.to(primary_device) for k, v in batch.items()}
            # 튜플 처리
            inputs, labels = batch
            inputs, labels = inputs.to(primary_device), labels.to(primary_device)

            with torch.no_grad():
                # 기존 코드 (잘못된 접근)
                # batch_inputs = batch_inputs.unsqueeze(1)

                # 변경 후
                inputs, labels = batch
                batch_inputs = inputs.to(primary_device)
                batch_labels = labels.to(primary_device)
                # batch_inputs = inputs.unsqueeze(1)  # 필요한 경우
                output = self.model(batch_inputs)

                loss = self.criterion(output, batch_labels.long())
                preds = torch.argmax(output, dim=1)



            loss = self.criterion(output, batch_labels.long())

            acc = self.acc(output, batch_labels)
            recall = self.recall(output, batch_labels)
            precision = self.precision(output, batch_labels)
            f1 = self.f1(output, batch_labels)

            avg_loss.append(loss.item())
            avg_acc.append(acc.item())
            avg_recall.append(recall.item())
            avg_precision.append(precision.item())
            avg_f1.append(f1.item())

            if c_mat is None:
                c_mat = self.c_mat(output, batch_labels)
            else:
                c_mat += self.c_mat(output, batch_labels)

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
        }, c_mat


if __name__ == "__main__":
    print(torch.cuda.is_available())
