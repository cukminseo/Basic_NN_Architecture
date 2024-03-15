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


def resize_images(images, new_size=(224, 224)):
    resized_images = []
    for image in images:
        # 이미지가 토치 텐서인 경우, 넘파이 배열로 변환
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # 데이터 타입이 np.float32 또는 다른 형태인 경우, np.uint8로 변환
        if image.dtype != np.uint8:
            # 최대/최소 정규화 후 255를 곱하여 uint8로 변환
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # 이미지 데이터가 (1, 높이, 너비) 형태인 경우, (높이, 너비)로 변환
        if image.shape[0] == 1:
            image = image.squeeze(0)

        # Numpy 배열을 PIL 이미지로 변환
        image_pil = Image.fromarray(image)

        # 이미지 리사이징
        image_resized = image_pil.resize(new_size, Image.BILINEAR)

        # 리사이징된 이미지를 다시 Numpy 배열로 변환
        image_resized_np = np.array(image_resized)

        # 그레이스케일 이미지를 RGB로 변환
        if len(image_resized_np.shape) == 2 or image_resized_np.shape[0] == 1:
            image_resized_np = np.stack((image_resized_np,) * 3, axis=-1)

        # (높이, 너비, 채널 수)에서 (채널 수, 높이, 너비)로 차원 순서 변경
        image_resized_np = np.transpose(image_resized_np, (2, 0, 1))

        # 리스트에 추가
        resized_images.append(image_resized_np)

    # 리스트를 Numpy 배열로 변환하여 반환
    return np.stack(resized_images)


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
    data_name_list = []
    y = []
    data_path_list = []
    clss = sorted(os.listdir(data_path), key=sort_key)  # 숫자 오름차순으로 정렬
    for cls in clss:
        print(f"class {clss.index(cls)} : {cls}")
        labels = os.listdir(f"{data_path}/{cls}")
        for label in labels:
            for data_name in os.listdir(f"{data_path}/{cls}/{label}"):
                if data_name == "Unnamed":
                    continue
                data_name_list.append(data_name)
                data_path_list.append(f"{data_path}/{cls}/{label}/{data_name}")
                # 여기서 cls 문자열에서 숫자 부분만 추출하여 y에 저장합니다.
                match = re.match(r"(\d+)_output", cls)
                cls_number = int(match.group(1)) if match else 0
                # y에 실제 값 대입
                # y.append(cls_number)
                # y에 순서 값 대입
                y.append(clss.index(cls))
    print(f"find {len(data_name_list)} files")
    return [data_name_list, data_path_list, y]


def plot_confusion_matrix(cf_matrix):
    classes = [
        "0_output",
        "50_output",
        "60_output",
        "70_output",
        "75_output",
        "80_output",
        "85_output",
        "90_output",
        "95_output",
        "100_output",
        "105_output",
        "110_output",
        "115_output",
        "120_output",
        "125_output",
        "130_output",
        "135_output",
        "140_output",
        "145_output",
        "150_output",
        "160_output",
        "170_output",
        "180_output",
        "190_output",
        "200_output",
        "210_output",
        "220_output",
        "230_output",
        "240_output",
        "250_output",
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
