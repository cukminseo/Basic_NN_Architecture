import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError  # UnidentifiedImageError를 가져옴
import seaborn as sns
import pandas as pd

labels = ['benign', 'malignant', 'normal']
data_dir = 'Dataset_BUSI_with_GT'

##################################
# 각 데이터의 통계 확인하여 violin plot으로 시각화
##################################

# 통계를 저장할 딕셔너리 초기화
stats = {}

for label in labels:
    label_dir = os.path.join(data_dir, label)
    image_files = [file for file in os.listdir(label_dir) if file.endswith('.png')]

    widths = []
    heights = []

    for image_file in image_files:
        image_path = os.path.join(label_dir, image_file)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
        except UnidentifiedImageError:
            print(f"Cannot identify image file {image_path}")

    widths = np.array(widths)
    heights = np.array(heights)

    # 통계 계산
    stats[label] = {
        'widths': widths,
        'heights': heights
    }

    print(f"Class: {label}")
    print(f"\tAverage Width: {np.mean(widths):.2f}")
    print(f"\tAverage Height: {np.mean(heights):.2f}")
    print(f"\tMax Width: {np.max(widths)}")
    print(f"\tMax Height: {np.max(heights)}")
    print(f"\tMin Width: {np.min(widths)}")
    print(f"\tMin Height: {np.min(heights)}")
    print(f"\tWidth Quartiles: {np.percentile(widths, [25, 50, 75])}")
    print(f"\tHeight Quartiles: {np.percentile(heights, [25, 50, 75])}")
    print()

# 바이올린 플롯을 그리기 위해 데이터 프레임 생성
df_list = []

for label in labels:
    widths = stats[label]['widths']
    heights = stats[label]['heights']

    df_list.append(pd.DataFrame({
        'Label': [label] * len(widths) + [label] * len(heights),
        'Dimension': ['Width'] * len(widths) + ['Height'] * len(heights),
        'Size': np.concatenate([widths, heights])
    }))

df = pd.concat(df_list)

# 바이올린 플롯 그리기
plt.figure(figsize=(12, 6))
sns.violinplot(x='Label', y='Size', hue='Dimension', data=df, split=True)
plt.title('Width-Height Distribution')
plt.savefig('Width-Height violin_plot.jpg')


##################################
# 각 데이터 확인
##################################

# 이미지를 표시할 영역 생성
fig, axs = plt.subplots(3, 6, figsize=(18, 9))

# 각 이미지의 원하는 너비와 높이
resize_width = 500
resize_height = 500

for i, label in enumerate(labels):
    label_dir = os.path.join(data_dir, label)

    # 이미지 파일 목록을 가져옴
    image_files = [file for file in os.listdir(label_dir) if file.endswith('.png')]
    print(f"class {label} : {len(image_files)} files found!")
    # 이미지 파일 목록을 정렬
    image_files = sorted(image_files)

    # 첫 6개의 이미지 파일을 반복
    for j in range(6):
        if j < len(image_files):
            # 이미지를 로드하고 리사이즈
            image_path = os.path.join(label_dir, image_files[j])
            image = Image.open(image_path)
            image = image.resize((resize_width, resize_height), Image.LANCZOS)

            # 이미지 또는 마스크에 따라 레이블 결정
            if j % 2 == 0:
                image_label = f'{label} - Image {j // 2 + 1}'
            else:
                image_label = f'{label} - Image {j // 2 + 1} Mask'

            # 해당 레이블로 이미지를 표시
            axs[i, j].imshow(image)
            axs[i, j].set_title(image_label)
            axs[i, j].axis('off')

plt.tight_layout()
plt.savefig('check_data.jpg')



##################################
# 전체 클래스의 데이터에 대해 그레이스케일 이미지 평균 표준편차 확인
##################################
print()
print()
print()
# 각 이미지의 원하는 너비와 높이
resize_width = 500
resize_height = 500

# 평균과 표준편차를 저장할 리스트 초기화
all_mean_list = []
all_std_list = []

for i, label in enumerate(labels):
    label_dir = os.path.join(data_dir, label)

    # 이미지 파일 목록을 가져옴
    image_files = [file for file in os.listdir(label_dir) if file.endswith('.png') and not (
        file.endswith('_mask.png') or file.endswith('_mask_1.png') or file.endswith('_mask_2.png'))]
    print(f"class {label} : {len(image_files)} files found.")
    # 이미지 파일 목록을 정렬
    image_files = sorted(image_files)

    # 평균과 표준편차를 저장할 리스트 초기화
    mean_list = []
    std_list = []

    for image_file in image_files:
        image_path = os.path.join(label_dir, image_file)
        try:
            with Image.open(image_path) as img:
                img = img.resize((resize_width, resize_height), Image.LANCZOS)
                img = img.convert("L")  # 그레이스케일로 변환
                img_array = np.array(img)
                mean_list.append(np.mean(img_array))
                all_mean_list.append(np.mean(img_array))
                std_list.append(np.std(img_array))
                all_std_list.append(np.std(img_array))
        except UnidentifiedImageError:
            print(f"Cannot identify image file {image_path}")

    # 각 클래스에 대해 평균과 표준편차 계산
    mean_array = np.array(mean_list)
    std_array = np.array(std_list)

    print(f"Class: {label}")
    print(f"\tMean of pixel values: {np.mean(mean_array):.2f}")
    print(f"\tStandard deviation of pixel values: {np.std(std_array):.2f}")
    print()

# 전체 클래스에 대해 평균과 표준편차 계산
all_mean_array = np.array(all_mean_list)
all_std_array = np.array(all_std_list)
print(f"Mean of pixel values(All class): {np.mean(all_mean_array):.2f}")
print(f"Standard deviation of pixel values(All class): {np.std(all_std_array):.2f}")
print()

