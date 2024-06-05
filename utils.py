"""
Created by minseo on 2024-03-12.
Description: 
"""
import random
import os
import torch
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from PIL import Image


def load_and_pad_image(file_path, target_size=(224, 224)):
    image = Image.open(file_path).convert('L')  # 이미지를 그레이스케일로 변환
    old_size = image.size  # 원래 이미지 크기
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image.resize(new_size, Image.LANCZOS)

    # 새로운 이미지 생성 및 패딩
    new_image = Image.new("L", target_size)
    new_image.paste(image, ((target_size[0] - new_size[0]) // 2,
                            (target_size[1] - new_size[1]) // 2))
    return np.array(new_image)


def fix_seed(seed):
    '''
    모든 난수 생성기에 동일한 시드(seed)를 설정하여 실험의 재현성을 보장합니다.

    :param seed: 설정할 난수 시드 값입니다.
    :return: 없음
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def sort_key(name):
    '''
    파일 이름에서 숫자 부분을 추출하여 정렬 기준으로 사용합니다.
    이 함수는 클래스 폴더를 숫자 순서대로 정렬할 때 사용됩니다.

    :param name: 정렬 기준으로 사용할 파일 이름입니다.
    :return: 파일 이름에서 추출한 숫자입니다.
    '''
    match = re.match(r"(\d+)_output", name)
    return int(match.group(1)) if match else 0

def get_data_list(data_path):
    '''
    지정된 경로에서 데이터 파일의 이름, 경로, 레이블을 수집합니다.
    이 함수는 지정된 데이터 경로에서 모든 데이터 파일을 찾아 각 파일의 이름, 경로 및 레이블을 리스트로 반환합니다.

    :param data_path: 데이터 파일들이 위치한 디렉토리의 경로입니다.
    :return: 데이터 파일 이름, 데이터 파일 경로, 데이터 파일 레이블로 구성된 리스트를 반환합니다.
    '''
    y = []
    file_path_list = []
    class_list = sorted(os.listdir(data_path), key=sort_key)  # 숫자 오름차순으로 정렬
    for label, _class in enumerate(class_list):
        print(f"class {label} : {_class}", end=" ")
        class_path = os.path.join(data_path,_class)
        file_list = os.listdir(class_path)
        for file_name in file_list:
            file_path = os.path.join(data_path, _class, file_name)

            file_path_list.append(file_path)
            y.append(label)
        print(f"(find {len(file_list)} files)")
    print(f"In all class, find {len(file_path_list)} files")
    return [file_path_list, y]


def get_image_size_statistics(file_path_list):
    max_size = (0, 0)
    min_size = (float('inf'), float('inf'))
    max_width = 0
    max_height = 0

    for file_path in file_path_list:
        with Image.open(file_path) as img:
            width, height = img.size
            max_size = max(max_size, (width, height), key=lambda x: x[0] * x[1])
            min_size = min(min_size, (width, height), key=lambda x: x[0] * x[1])
            max_width = max(max_width, width)
            max_height = max(max_height, height)

    print(f"Max Image Size: {max_size}")
    print(f"Min Image Size: {min_size}")
    print(f"Max Width: {max_width}")
    print(f"Max Height: {max_height}")
    return (max_width, max_height)


def plot_confusion_matrix(cf_matrix):
    classes = [
        "benign",
        "malignant",
        "normal",
    ]

    dpi_val = 68.84
    plt.figure(figsize=(1024 / dpi_val, 768 / dpi_val), dpi=dpi_val)
    sn.set_context(font_scale=1)
    cm_numpy = cf_matrix
    df_cm = pd.DataFrame(
        cm_numpy / np.sum(cm_numpy, axis=1)[:, np.newaxis],
        index=classes,
        columns=classes,
    )

    return sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 10}, cbar=True)
