import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEBUG_MODE = False

# 경로 설정
logs_folder = 'Y:\\전혈실험_결과\\ML6_just_log\\inceptionNet_logs'

# 텐서보드 이벤트 파일을 불러오기 위한 함수
def get_event_files(logs_folder):
    event_files = []
    for root, dirs, files in os.walk(logs_folder):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    return event_files

# 이벤트 파일에서 필요한 데이터 추출
def extract_metrics_from_event_files(event_files):
    metrics = {}

    for event_file in event_files:
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()

        # 필요한 데이터가 있는 키를 찾기
        tags = event_acc.Tags()['scalars']
        if DEBUG_MODE:
            print(f"Available tags in {event_file}:", tags)  # 디버깅을 위해 사용 가능한 태그 출력

        for tag in tags:
            events = event_acc.Scalars(tag)
            for event in events:
                step = event.step
                value = event.value
                if step not in metrics:
                    metrics[step] = {}
                metrics[step][tag] = value

    if DEBUG_MODE:
        # 디버깅을 위해 추출한 데이터 출력
        print("Extracted metrics:", metrics)
    return metrics

# 가장 높은 정확도를 가진 에포크와 그 지표들을 출력
def find_best_epoch(metrics):
    best_epoch = None
    max_acc = -float('inf')
    best_metrics = {}

    for epoch, data in metrics.items():
        if epoch > 20 and 'acc/val' in data and data['acc/val'] > max_acc:
            max_acc = data['acc/val']
            best_epoch = epoch
            best_metrics = data

    if best_epoch is not None:
        best_metrics['epoch'] = best_epoch

    return best_metrics if best_epoch is not None else None

# F1-score 계산 함수
def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# 폴더 내 모든 실험을 처리하여 결과를 수집하고, 각 실험별 베스트 에포크와 그 성능 지표를 출력
def process_all_experiments(logs_folder):
    all_best_metrics = []
    sub_folders = [os.path.join(logs_folder, d) for d in os.listdir(logs_folder) if os.path.isdir(os.path.join(logs_folder, d))]

    for sub_folder in sub_folders:
        print(f"Processing {sub_folder}...")
        event_files = get_event_files(sub_folder)
        if not event_files:
            print(f"No event files found in {sub_folder}.")
            continue

        metrics = extract_metrics_from_event_files(event_files)
        best_metrics = find_best_epoch(metrics)

        if best_metrics:
            precision = best_metrics.get('precision/val', 0)
            recall = best_metrics.get('recall/val', 0)
            f1_score = calculate_f1(precision, recall)
            best_metrics['f1-score/val'] = f1_score

            all_best_metrics.append(best_metrics)
            print(f"Best epoch in {sub_folder}: {best_metrics['epoch']}")
            print(f"Accuracy: {best_metrics['acc/val']:.4f}")
            print(f"F1 Score (val): {f1_score:.4f}")
            print(f"Loss (val): {best_metrics['loss/val']:.4f}")
            print(f"Precision (val): {precision:.4f}")
            print(f"Recall (val): {recall:.4f}\n")
        else:
            print(f"Could not find the best epoch metrics in {sub_folder}.\n")

    if all_best_metrics:
        metrics_keys = ['acc/val', 'f1-score/val', 'loss/val', 'precision/val', 'recall/val']
        summary = {key: [m[key] for m in all_best_metrics] for key in metrics_keys}

        means = {key: np.mean(values) for key, values in summary.items()}
        stddevs = {key: np.std(values) for key, values in summary.items()}

        print("\nSummary of all experiments:")
        for key in metrics_keys:
            print(f"{key} - Mean: {means[key]:.4f}, Standard Deviation: {stddevs[key]:.4f}")
    else:
        print("No valid metrics found in any experiment.")

# 실행 함수
def main():
    process_all_experiments(logs_folder)

if __name__ == "__main__":
    main()
