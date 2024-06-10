import torch
import os
import random
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd


def fix_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDDN seed고정을 통한 정확한 학습 재현
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def datadir_check(data_dir="working\\kfold5_seed42"):
    # Check if the base directory exists
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return

    # Loop through each fold directory
    for fold in os.listdir(data_dir):
        fold_dir = os.path.join(data_dir, fold)
        if os.path.isdir(fold_dir):
            print(f"\nChecking {fold_dir}...")

            # Check train and validation directories
            for phase in ['train', 'validation']:
                phase_dir = os.path.join(fold_dir, phase)
                if os.path.exists(phase_dir):
                    print(f"\t{phase} directory:")
                    for label in os.listdir(phase_dir):
                        label_dir = os.path.join(phase_dir, label)
                        if os.path.isdir(label_dir):
                            num_files = len(os.listdir(label_dir))
                            print(f"\t\t{label}: {num_files} files")
                else:
                    print(f"\t{phase_dir} does not exist.")


def split_KFold(data_dir="Dataset_BUSI_with_GT", k_fold=5, seed=42):
    print("Data preparing for k-fold cross-validation...")
    # Create a list to store file paths and labels
    file_paths = []
    labels = []

    # Loop through the subdirectories (benign, malignant, normal)
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                if image_file.endswith('.png') and not (image_file.endswith('_mask.png') or
                                                        image_file.endswith('_mask_1.png') or
                                                        image_file.endswith('_mask_2.png')):
                    image_path = os.path.join(label_dir, image_file)
                    labels.append(label)
                    file_paths.append(image_path)

    # Create a DataFrame to store the file paths and labels
    data = pd.DataFrame({'Image_Path': file_paths, 'Label': labels})

    # Define the number of folds for k-fold validation
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)

    # Create directories for k-fold validation
    base_dir = f'working\\kfold{k_fold}_seed{seed}'
    os.makedirs(base_dir, exist_ok=True)

    for fold, (train_index, val_index) in enumerate(skf.split(data, data['Label'])):
        fold_dir = os.path.join(base_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'validation'), exist_ok=True)

        train_data = data.iloc[train_index]
        val_data = data.iloc[val_index]

        for _, row in train_data.iterrows():
            image_path = row['Image_Path']
            label = row['Label']
            label_dir = os.path.join(fold_dir, 'train', label)
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy(image_path, label_dir)

        for _, row in val_data.iterrows():
            image_path = row['Image_Path']
            label = row['Label']
            label_dir = os.path.join(fold_dir, 'validation', label)
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy(image_path, label_dir)

    print("Data prepared for k-fold cross-validation.")


def plot_confusion_matrix(cf_matrix):
    classes = [
        "benign",
        "malignant",
        "normal",
    ]

    dpi_val = 68.84
    plt.figure(figsize=(1024 / dpi_val, 768 / dpi_val), dpi=dpi_val)
    sns.set_context(font_scale=1)
    cm_numpy = cf_matrix
    df_cm = pd.DataFrame(
        cm_numpy / np.sum(cm_numpy, axis=1)[:, np.newaxis],
        index=classes,
        columns=classes,
    )

    ax = sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 40}, cbar=True)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=25)  # x축 글자 크기 조정
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=25)  # y축 글자 크기 조정

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=40)  # 색상 막대 글자 크기 조정

    return ax


if __name__ == '__main__':
    split_KFold()
