import torch
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from torchvision.transforms import RandomHorizontalFlip, RandomRotation
from torch.utils import tensorboard

import util
import training
from model.ResNet_relu import ResNet, Bottleneck
import gc

DEBUG = True
k_fold = 5
seed = 42
util.fix_seed(seed)

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))
device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능 여부에 따라 device 정보 저장

# argparse
config = argparse.ArgumentParser()
config.add_argument("--batch_size", default=4, type=int)
config.add_argument("--lr", default=0.005, type=float)
config.add_argument("--gpus", default="0", type=str)
config.add_argument("--epoch", default=200, type=int)
config.add_argument("--earlystop_patience", default=10, type=int)
config.add_argument("--train_size", default=0.8, type=float)
config.add_argument("--val_size", default=0.2, type=float)
config.add_argument("--num_classes", default=3, type=int)
config.add_argument("--model", default="ResNet", type=str)

config = config.parse_args()

data_dir = f"working\\kfold{k_fold}_seed{seed}"

# 폴드 데이터셋이 준비되어있지 않으면 생성하기
if not os.path.exists(data_dir):
    util.split_KFold('Dataset_BUSI_with_GT', k_fold=k_fold, seed=seed)

if DEBUG:
    util.datadir_check(data_dir)

for fold in range(k_fold):

    fold_data_dir = os.path.join(data_dir, f"fold_{fold}")

    print("============================================")
    print(f"====== K Fold Validation step => {fold}/{k_fold} =======")
    print("============================================")

    # Define the minority classes in your dataset
    class_names = ['malignant', 'normal', 'benign']
    minority_classes = ['malignant', 'normal']

    # Define data transformations for train, validation, and test sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.Grayscale(num_output_channels=1),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(30, expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize([83.63/255], [9.16/255])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([83.63/255], [9.16/255])
        ])
    }

    # Create datasets for train, validation
    image_datasets = {
        x: ImageFolder(
            root=os.path.join(fold_data_dir, x),
            transform=data_transforms[x]
        )
        for x in ['train', 'validation']
    }

    # Create dataloaders for train, validation, and test
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=config.batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'validation']}

    # Calculate dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}

    # Get class labels
    class_names = image_datasets['train'].classes

    # Print dataset sizes and class labels
    print("Dataset Sizes:", dataset_sizes)
    print("Class Labels:", class_names)

    ######
    # 모델 가져오기
    ######

    if config.model == 'ResNet':
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=len(class_names))
    else:
        model = config.model(num_classes=len(class_names))

    model = model.to(device)

    trainer = training.Trainer(config, model, device)

    stopping_check = float('inf')
    patience = 0
    checkpoint_score = 0

    # tensorboard 로그 기록
    writer = tensorboard.SummaryWriter(f"./logs/{config.model}_fold{fold + 1}_batch{config.batch_size}_lr{config.lr}(relu,micro)")

    # 에포크별 학습 진행
    for epoch in range(config.epoch):

        train_result = trainer.train(epoch, image_datasets['train'])
        eval_result, c_mat = trainer.eval(epoch, image_datasets['validation'])

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

        # # 결과 출력
        # print(f"{epoch} train result:", train_result)
        # print(f"{epoch} val result:", eval_result)
        # 결과 출력
        print(f"{epoch} train result: loss: {train_result['loss']:.5f}, acc: {train_result['acc']:.5f}")
        print(f"{epoch} val result: loss: {eval_result['loss']:.5f}, acc: {eval_result['acc']:.5f}")

        # tensorboard for confusion matrix
        ax = util.plot_confusion_matrix(c_mat.cpu().numpy())
        cm = ax.get_figure()
        if not os.path.exists("./confusion_matrix"):
            os.makedirs("./confusion_matrix")

        eval_acc = eval_result["acc"]
        eval_loss = eval_result["loss"]
        plt.savefig(
            f"./confusion_matrix/epoch{epoch}_{config.model}_fold{fold + 1}_batch{config.batch_size}_lr{config.lr}_acc{eval_acc:.2f}.png")
        writer.add_figure("Confusion Matrix", cm, epoch)
        if stopping_check < eval_loss:
            patience += 1
        else:
            patience = 0

        stopping_check = eval_loss

        if not os.path.exists("./output"):
            os.makedirs("./output")
        if checkpoint_score < eval_acc:
            torch.save(trainer.model.state_dict(),
                       f"./output/epoch{epoch}_{config.model}_fold{fold + 1}_batch{config.batch_size}_lr{config.lr}_acc{eval_acc:.2f}.ckpt")
            torch.save(
                c_mat,
                f"./confusion_matrix/epoch{epoch}_{config.model}_fold{fold + 1}_batch{config.batch_size}_lr{config.lr}_acc{eval_acc:.2f}.pth")

        if patience == config.earlystop_patience:
            print("early stopping at", epoch)
            break

    writer.close()

    del image_datasets, dataloaders
    gc.collect()
    torch.cuda.empty_cache()  # GPU 메모리 해제

    break
