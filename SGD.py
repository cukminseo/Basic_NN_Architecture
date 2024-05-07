import torch
import torch.nn as nn
import torch.optim as optim


class Optimizer(object):
    """
    신경망을 최적화하는 기능을 공하는 상위 클래스
    """

    def __init__(self, lr: float = 0.01):
        """
        학습률이 설정되어야 한다.
        """
        self.lr = lr

    def step(self) -> None:
        """
        Optimizer를 구현하는 구상 클래스는 이 메서드를 구현해야 한다.
        """
        pass


class SGD_book(Optimizer):
    """
    확률적 경사 하강법을 적용한 Optimizer
    """

    def __init__(self, lr: float = 0.01) -> None:
        '''Pass '''
        super().__init__(lr)

    def step(self):
        """
        각 파라미터에 학습률을 곱해 기울기방향으로 수정함
        """
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad


class SGD_book_same(Optimizer):
    """
    확률적 경사 하강법을 적용한 Optimizer
    """

    def __init__(self, lr: float = 0.01) -> None:
        '''Pass '''
        super().__init__(lr)

    def step(self):
        """
        각 파라미터에 학습률을 곱해 기울기방향으로 수정함
        """
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad


class SGD_MS(Optimizer):
    """
    확률적 경사 하강법을 적용한 Optimizer
    """

    def __init__(self, lr: float = 0.01) -> None:
        '''Pass '''
        super().__init__(lr)

    def step(self):
        """
        각 파라미터에 학습률을 곱해 기울기방향으로 수정함
        """
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param = param - self.lr * param_grad


# 간단한 신경망 정의
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# 데이터 준비
x = torch.randn(100, 2)
y = torch.randn(100, 1)

# 손실 함수 정의
criterion = nn.MSELoss()

# Optimizer 생성
optimizer_SGD_book = SGD_book(lr=0.01)
optimizer_SGD_book_same = SGD_book_same(lr=0.01)
optimizer_SGD_MS = SGD_MS(lr=0.01)

# 각 Optimizer를 사용하여 학습 진행
for epoch in range(10):
    # SGD_book 사용
    net = Net()
    optimizer = optimizer_SGD_book
    optimizer.zero_grad()

    output = net(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # SGD_book_same 사용
    net = Net()
    optimizer = optimizer_SGD_book_same
    optimizer.zero_grad()

    output = net(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # SGD_MS 사용
    net = Net()
    optimizer = optimizer_SGD_MS
    optimizer.zero_grad()

    output = net(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# 학습 결과 확인
print("SGD_book 학습 결과:", net(x))
print("SGD_book_same 학습 결과:", net(x))
print("SGD_MS 학습 결과:", net(x))