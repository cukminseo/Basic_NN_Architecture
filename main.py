"""
Created by minseo on 2024-03-12.
Description: main code
"""
import argparse

import torch
import sklearn.model_selection
import sys
import gc
import os
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils import tensorboard
from torch.utils.data import TensorDataset
# 이미지 데이터 가져오기 위한 호출
from PIL import Image


# 내 파일
import utils
import training
from model.VGG16 import VGG16

utils.fix_seed(42)

# argparse
config = argparse.ArgumentParser()
config.add_argument("--batch_size", default=8, type=int)
config.add_argument("--lr", default=0.0005, type=float)
config.add_argument("--gpus", default="0", type=str)
config.add_argument("--epoch", default=200, type=int)
config.add_argument("--eawrlystop_patience", default=20, type=int)
config.add_argument("--train_size", default=0.8, type=float)
config.add_argument("--val_size", default=0.2, type=float)
config.add_argument("--num_classes", default=3, type=int)
config.add_argument("--model", default="VGG16", type=str)
# config.add_argument("--DEBUG", default=True, type=bool)

config = config.parse_args()
debug = True
# GPU 설정
if torch.cuda.is_available():
    device_ids = list(map(int, config.gpus.split(',')))  # "0,1,2" 같은 문자열을 [0, 1, 2] 리스트로 변환
    device = torch.device(f"cuda:{device_ids[0]}")  # 첫 번째 GPU를 기본 디바이스로 설정
    if debug:
        print("CUDA is working.")
else:
    print("CUDA is not available.")
    sys.exit(1)  # CUDA 사용 불가 시 오류 메시지 출력 후 프로그램 종료

# 학습 폴드 설정
k_folds = 5
if debug:
    print("use StratifiedShuffleSplit...")
# sss 선언
sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=k_folds, test_size=config.val_size, random_state=0)
# 데이터 root 위치
data_path = os.path.join(os.getcwd(), "application", "Breast-Ultrasound-Images-classification", "Dataset_BUSI_with_GT")
# 데이터 불러오기
file_path_list, y = utils.get_data_list(data_path)

#------샘플링 방법 추가가능

#------

if debug:
    print(f"select {len(file_path_list)} files.(After sampling)")

# 이미지 최대 가로 길이와 세로길이 구하기
max_width, max_height = utils.get_image_size_statistics(file_path_list)

print("loading data from directory ...")

# png파일 로드 및 최대 가로세로 크기에 맞춰서 패딩
x = np.stack(
    [utils.load_and_pad_image(file_path, target_size=(244, 244)) for file_path in tqdm(file_path_list)],
    axis=0
)
if debug:
    print(f"done. one tensor shape : {x[0].shape}")
    print(f"done. stacked tensor shape : {x.shape}")

# 그레이스케일에 대응하기 위해 차원을 (N, H, W)에서 (N, 1, H, W)로 변경하여 1채널로 만듦
x = np.expand_dims(x, axis=1)

# 폴드별로 학습진행
for i, (train_index, test_index) in enumerate(sss.split(x, y)):
    print("=========================================")
    print("====== K Fold Validation step => %d/%d =======" % (i + 1, k_folds))
    print("=========================================")

    # 모델 생성 및 DataParallel 설정
    model = VGG16(num_classes=len(np.unique(y)))

    # 병렬처리
    if torch.cuda.is_available() and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        model = model.to(device)

    trainer = training.Trainer(config, model, device)
    if debug:
        print(f"TRAIN: total {len(train_index)} sample")
        print(f"TEST: total {len(test_index)} sample")

    X_train, X_test = np.array(x)[train_index], np.array(x)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

    stopping_check = torch.inf
    patience = 0
    checkpoint_score = 0

    # tensorboard 로그 기록
    writer = tensorboard.SummaryWriter(f"./logs/{config.model}_fold{i + 1}_batch{config.batch_size}_lr{config.lr}")

    if debug:
        print(f"X_train.shape : {X_train.shape}")
        print(f"X_test.shape : {X_test.shape}")

    # numpy 배열을 PyTorch 텐서로 변환
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()


    # 에포크별 학습 진행
    for epoch in range(config.epoch):

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_result = trainer.train(epoch, train_dataset)
        print(train_result)
        eval_result, c_mat = trainer.eval(epoch, test_dataset)

        # tensorboard for training result
        writer.add_scalar("loss/train", train_result["loss"], epoch)
        writer.add_scalar("acc/train", train_result["acc"], epoch)
        writer.add_scalar("precision/train", train_result["precision"], epoch)
        writer.add_scalar("recall/train", train_result["recall"], epoch)
        writer.add_scalar("f1-score/train", train_result["f1"], epoch)

        # tensorboard for validation result
        writer.add_scalar("loss/val", eval_result["loss"], epoch)
        writer.add_scalar("acc/val", eval_result["acc"], epoch)
        writer.add_scalar("precision/val", eval_result["precision"], epoch)
        writer.add_scalar("recall/val", eval_result["recall"], epoch)
        writer.add_scalar("f1-score/val", eval_result["f1"], epoch)

        # 결과 출력
        print(f"{epoch} train result:", train_result)
        print(f"{epoch} val result:", eval_result)

        # tensorboard for confusion matrix
        ax = utils.plot_confusion_matrix(c_mat.cpu().numpy())
        cm = ax.get_figure()
        if not os.path.exists("./confusion_matrix"):
            os.makedirs("./confusion_matrix")

        eval_acc = eval_result["acc"]
        eval_loss = eval_result["loss"]
        plt.savefig(
            f"./confusion_matrix/epoch{epoch}_{config.model}_fold{i + 1}_batch{config.batch_size}_lr{config.lr}_acc{eval_acc:.2f}.png")
        writer.add_figure("Confusion Matrix", cm, epoch)
        if stopping_check < eval_loss:
            patience += 1

        stopping_check = eval_loss
        if not os.path.exists("./output"):
            os.makedirs("./output")
        if checkpoint_score < eval_acc:
            torch.save(trainer.model.state_dict(),
                       f"./output/epoch{epoch}_{config.model}_fold{i + 1}_batch{config.batch_size}_lr{config.lr}_acc{eval_acc:.2f}.ckpt")
            torch.save(
                c_mat,
                f"./confusion_matrix/epoch{epoch}_{config.model}_fold{i + 1}_batch{config.batch_size}_lr{config.lr}_acc{eval_acc:.2f}.pth")

        if patience == config.eawrlystop_patience:
            print("early stopping at", epoch)
            break

    writer.close()

    del X_train, X_test, y_train, y_test
    gc.collect()
