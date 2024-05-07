"""
Created by minseo on 2024-03-12.
Description: main code
"""
import argparse

import torch
import utils
import timm
import sklearn.model_selection
import sys
import training
import gc
import os
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils import tensorboard
from torch.utils.data import TensorDataset
from model.VGG16 import VGG16

labels = [i + 1 for i in range(30)]
utils.fix_seed(42)

# argparse
config = argparse.ArgumentParser()
config.add_argument("--batch_size", default=256, type=int)
config.add_argument("--num_workers", default=4, type=int)
config.add_argument("--lr", default=0.00005, type=float)
config.add_argument("--gpus", default="0", type=str)
config.add_argument("--epoch", default=200, type=int)
config.add_argument("--patience", default=20, type=int)
config.add_argument("--train_samples", default=8000, type=int)
config.add_argument("--val_samples", default=2000, type=int)
config.add_argument("--num_classes", default=30, type=int)
config.add_argument("--model", default="vit_base_patch16_224", type=str)

config = config.parse_args()


# GPU 설정
if torch.cuda.is_available():
    device_ids = list(map(int, config.gpus.split(',')))  # "0,1,2" 같은 문자열을 [0, 1, 2] 리스트로 변환
    device = torch.device(f"cuda:{device_ids[0]}")  # 첫 번째 GPU를 기본 디바이스로 설정
else:
    print("CUDA is not available.")
    sys.exit(1)  # CUDA 사용 불가 시 오류 메시지 출력 후 프로그램 종료


# 학습 폴드 설정
k_folds = 5

print("loading data ...")

sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=k_folds, test_size=0.2, random_state=0)

# 데이터 위치
data_path = "./data/spectrogram/librosa_stft_data_vec_tol0.6"
# 데이터 불러오기
data_name_list, data_path_list, y = utils.get_data_list(data_path)

# stride = 20
# # stride = 2
# selected_data_name_list = []
# selected_data_path_list = []
# selected_y = []
# for i in range(0, len(data_name_list), stride):
#     select = np.random.randint(i, min(i + stride, len(data_name_list)))
#     selected_data_name_list.append(data_name_list[select])
#     selected_data_path_list.append(data_path_list[select])
#     selected_y.append(y[select])
# data_name_list = selected_data_name_list
# data_path_list = selected_data_path_list
# y = selected_y

print(f"get {len(data_name_list)} files")
print("loading data from directory ...")


# # np파일 로드
# x = np.stack(
#     [np.load(data_path) for data_path in tqdm(data_path_list)],
#     axis=0,
# )

# np.load 대신 pd.read_csv 사용하여 데이터 불러오기
x = np.stack(
    [pd.read_csv(data_path).to_numpy() for data_path in tqdm(data_path_list)],
    axis=0,
)


print(f"done. tensor shape : {x.shape}")


# 폴드별로 학습진행
for i, (train_index, test_index) in enumerate(sss.split(x, y)):

    # 모델 생성 및 DataParallel 설정
    model = timm.create_model(config.model, num_classes=config.num_classes, pretrained=True)

    if torch.cuda.is_available() and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        model = model.to(device)

    trainer = training.Trainer(config, model, device)


    print("TRAIN:\n", train_index, "\nTEST:\n", test_index)
    X_train, X_test = np.array(x)[train_index], np.array(x)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
    print(y)
    print("=========================================")
    print("====== K Fold Validation step => %d/%d =======" % (i + 1, k_folds))
    print("=========================================")

    stopping_check = torch.inf
    patience = 0
    checkpoint_score = 0

    # tensorboard 로그 기록
    writer = tensorboard.SummaryWriter(f"./logs/{config.model}_fold{i + 1}_batch{config.batch_size}_lr{config.lr}")

    print(f"X_train.shape : {X_train.shape}")
    print(f"X_test.shape : {X_test.shape}")

    # 데이터 리사이징 전에 채널 차원을 추가
    X_train_expanded = X_train[:, np.newaxis, :, :]  # (데이터 개수, 1, 높이, 너비) 형태로 변경
    X_test_expanded = X_test[:, np.newaxis, :, :]  # (데이터 개수, 1, 높이, 너비) 형태로 변경

    # 예시: 이미지 데이터가 (데이터 개수, 1, 높이, 너비)인 경우, 차원을 변경하지 않고 리사이징만 진행
    X_train_resized = utils.resize_images(X_train_expanded)  # 리사이징 함수가 입력 데이터 형태를 처리할 수 있도록 수정 필요
    X_test_resized = utils.resize_images(X_test_expanded)  # 리사이징 함수가 입력 데이터 형태를 처리할 수 있도록 수정 필요

    #에포크별 학습 진행
    for epoch in range(config.epoch):

        # 데이터 타입 변환 및 TensorDataset에 로드
        X_train = torch.from_numpy(X_train_resized).float()
        X_test = torch.from_numpy(X_test_resized).float()
        # y_train = torch.from_numpy(y_train).long()
        # y_test = torch.from_numpy(y_test).long()

        # y_train이 numpy 배열인 경우에만 변환 수행
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train).long()
        # y_test에 대해서도 동일하게 적용
        if isinstance(y_test, np.ndarray):
            y_test = torch.from_numpy(y_test).long()

        print(y_train)

        # 기본 진행
        # X_train = torch.from_numpy(X_train).float() # 데이터 타입이 float이어야 할 경우
        # y_train = torch.from_numpy(y_train).long() # 레이블 일반적으로 long 타입 사용
        # X_test = torch.from_numpy(X_test).float() # 데이터 타입이 float이어야 할 경우
        # y_test = torch.from_numpy(y_test).long() # 레이블 일반적으로 long 타입 사용

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
        plt.savefig(f"./confusion_matrix/epoch{epoch}_{config.model}_fold{i + 1}_batch{config.batch_size}_lr{config.lr}_acc{eval_acc:.2f}.png")
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

        if patience == config.patience:
            print("early stopping at", epoch)
            break

    writer.close()

    del X_train, X_test, y_train, y_test
    gc.collect()

