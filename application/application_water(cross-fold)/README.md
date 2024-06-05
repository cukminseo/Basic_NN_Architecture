# water_quality_prediction

* alexnet.py : alexnet 모델 구현한 코드
* run_experiment.py : 메인 코드, 이걸 통해서 실험 진행
* training.py : trainer 구현한 코드
* utils.py : 여러가지 코드에 있어서 필요한 tool들 구현한 코드

```bash
python run_experiment.py --batch_size 64 --lr 0.000055 --gpus [0, 1]
```
* --batch_size : batch size를 조절해주는 argument
* --lr : learning rate를 조절해주는 argument
* --gpus : gpu를 할당해주는 arguemnt ([0, 1] 같은 경우는 gpu 0, 1번을 모두 사용한다는 뜻 (병렬연산), 하나만 쓰고 싶으면 [0])
