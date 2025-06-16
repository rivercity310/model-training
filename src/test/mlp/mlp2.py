import numpy as np

# 다중 클래스 분류 MLP


def softmax(x):
    """
    입력 받은 숫자 배열을 전체 합이 1인 확률 배열로 변환
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def to_one_hot(y, num_classes):
    """
    정수 라벨(0, 1, ...)을 원-핫 인코딩 벡터로 변환
    ex) y=1, num_classes=3 -> [0, 1, 0]
    """
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def relu(x):
    """ReLU 활성화 함수"""
    return np.maximum(0, x)


def relu_derivative(x):
    """
    ReLU 함수의 미분
    0보다 크면 1, 아니면 0
    """
    return np.where(x > 0, 1, 0)


class MultiClassMLP:
    def __init__(self, input_size, hidden_size, num_classes):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # 가중치와 편향 초기화
        # --- 해결 방안 2: He 가중치 초기화 적용 ---
        # ReLU 활성화 함수에 더 적합한 방식으로 가중치를 초기화합니다.
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, num_classes) * np.sqrt(2 / hidden_size)
        self.bias2 = np.zeros((1, num_classes))

    def forward(self, X):
        self.hidden_sum = np.dot(X, self.weights1) + self.bias1
        self.activated_hidden = relu(self.hidden_sum)
        self.output_sum = np.dot(self.activated_hidden, self.weights2) + self.bias2

        probabilities = softmax(self.output_sum)
        return probabilities

    def backward(self, X, y_one_hot, output_probabilities, learning_rate):
        num_samples = X.shape[0]

        # 출력층의 오차 델타 계산 (Softmax + Cross-Entropy Loss)의 미분
        # 위 식의 계산은 놀랍게도 (예측 확률 - 실제 정답)과 일치
        output_delta = output_probabilities - y_one_hot

        # 은닉층 오차 계산
        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * relu_derivative(self.activated_hidden)

        # 가중치와 편향의 그래디언트 계산
        dw2 = np.dot(self.activated_hidden.T, output_delta) / num_samples
        db2 = np.sum(output_delta, axis=0, keepdims=True) / num_samples
        dw1 = np.dot(X.T, hidden_delta) / num_samples
        db1 = np.sum(hidden_delta, axis=0, keepdims=True) / num_samples

        # --- 그래디언트 클리핑 적용 ---
        # 그래디언트 폭주 방지를 위해(너무 커지는 것) 임계값을 정해 잘라냅니다.
        grad_clip_threshold = 5.0
        dw2 = np.clip(dw2, -grad_clip_threshold, grad_clip_threshold)
        db2 = np.clip(db2, -grad_clip_threshold, grad_clip_threshold)
        dw1 = np.clip(dw1, -grad_clip_threshold, grad_clip_threshold)
        db1 = np.clip(db1, -grad_clip_threshold, grad_clip_threshold)

        # 가중치와 편향 업데이트
        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2
        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        y_one_hot = to_one_hot(y, num_classes=self.num_classes)

        for epoch in range(epochs):
            # 순전파
            probabilities = self.forward(X)

            # 역전파
            self.backward(X, y_one_hot, probabilities, learning_rate)

            if (epoch + 1) % 100 == 0:
                # Cross Entropy Loss 계산
                loss = -np.sum(y_one_hot * np.log(probabilities + 1e-9)) / len(y)
                print(f"{epoch + 1}번째 학습 후, 오차(Loss): {loss:.4f}")

    def predict(self, X):
        # 확률이 가장 높은 클래스의 인덱스 반환
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)


if __name__ == '__main__':
    # 3개의 클래스를 가진 가상 데이터 생성
    np.random.seed(0)
    num_samples_per_class = 50
    # 클래스 0, 1, 2에 해당하는 데이터 클러스터 생성
    X0 = np.random.randn(num_samples_per_class, 2) + np.array([0, -2])
    X1 = np.random.randn(num_samples_per_class, 2) + np.array([2, 2])
    X2 = np.random.randn(num_samples_per_class, 2) + np.array([-2, 2])

    X = np.vstack((X0, X1, X2))
    y = np.array([0] * num_samples_per_class + [1] * num_samples_per_class + [2] * num_samples_per_class).reshape(-1, 1)

    # 모델 생성 및 훈련
    mlp = MultiClassMLP(input_size=2, hidden_size=10, num_classes=3)

    print("\n=== NumPy 다중 클래스 분류 훈련 시작 ===")
    mlp.train(X, y, epochs=10000, learning_rate=0.1)
    print("=== 훈련 종료 ===\n")

    # 예측 및 정확도 계산
    predictions = mlp.predict(X)
    accuracy = np.mean(predictions == y.flatten()) * 100
    print(f"최종 훈련 정확도: {accuracy:.2f}%")
