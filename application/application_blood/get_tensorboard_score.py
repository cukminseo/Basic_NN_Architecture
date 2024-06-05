import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEBUG_MODE = False

# 경로 설정
logs_folder = 'Y:\\전혈실험_결과\\ML6\\efficientNet_logs\\efficientNet_0_batch32_lr0.0005'

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
        if 'acc/val' in data and data['acc/val'] > max_acc:
            max_acc = data['acc/val']
            best_epoch = epoch
            best_metrics = data

    if best_epoch is not None:
        best_metrics['epoch'] = best_epoch

    return best_metrics if best_epoch is not None else None

# 실행 함수
def main():
    event_files = get_event_files(logs_folder)
    if not event_files:
        print("No event files found in the logs folder.")
        return

    metrics = extract_metrics_from_event_files(event_files)
    best_metrics = find_best_epoch(metrics)

    if best_metrics:
        print(f"Best epoch: {best_metrics['epoch']}")
        print(f"Accuracy: {best_metrics['acc/val']:.4f}")
        print(f"F1 Score (val): {best_metrics['f1-score/val']:.4f}")
        print(f"Loss (val): {best_metrics['loss/val']:.4f}")
        print(f"Precision (val): {best_metrics['precision/val']:.4f}")
        print(f"Recall (val): {best_metrics['recall/val']:.4f}")
    else:
        print("Could not find the best epoch metrics.")

if __name__ == "__main__":
    main()
