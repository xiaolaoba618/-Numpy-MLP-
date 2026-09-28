import numpy as np

# %% 定义 MLP 分类器

class MLPClassifier:
    """
    使用 NumPy 从零实现的多层感知机（MLP）分类器。

    支持字符串和数字类别标签。训练时会自动将类别编码为数字，
    预测时再将数字类别还原为原始类别。

    用法：
        model = MLPClassifier(
            hidden_layers=[128, 64],
            learning_rate=0.1,
            epochs=100,
            lambda_L2=0.00001
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        accuracy = model.score(X_test, y_test)

    例如：
        y_train = ["Normal", "Failure", "Normal"]

        模型内部自动编码：
        "Failure" -> 0
        "Normal"  -> 1

        predict() 最终仍返回：
        ["Normal", "Failure", ...]

    参数：
        hidden_layers: 隐藏层结构，例如 [128, 64]。
        learning_rate: 梯度下降的学习率。
        epochs: 训练轮数。
        lambda_L2: L2 正则化参数。
        random_state: 随机种子。

    输入数据格式：
        X: (n_samples, n_features)
        y: (n_samples,)，可以是数字或字符串类别。

    输出：
        predict(): 返回原始类别。
        predict_proba(): 返回各类别的预测概率。
        score(): 返回分类准确率。
    """

    def __init__(self, hidden_layers=[128, 64], learning_rate=0.1, epochs=100, lambda_L2=0.00001, random_state=42):
        """
        创建一个 MLP 分类器。

        hidden_layers 决定网络的隐藏层数量和每层神经元数量。

        例如：
            [128, 64] → 输入 → 128 → 64 → 输出
            [256, 128, 64] → 输入 → 256 → 128 → 64 → 输出

        类别标签无需提前转换为数字，fit() 会自动进行编码。
        """
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_L2 = lambda_L2
        self.random_state = random_state
        self.weights, self.biases = [], []
        self.loss_history, self.accuracy_history = [], []
        self.input_size, self.output_size = None, None
        self.classes_ = None
        self.class_to_index = None

    # %% 定义激活函数

    @staticmethod
    def relu(x):
        """
        ReLU 激活函数。

        公式：
            ReLU(x) = max(0, x)

        输入：
            x: 任意形状的 NumPy 数组。

        输出：
            与 x 形状相同的数组。
        """
        return np.maximum(0, x)

    # %% 定义 SoftMax

    @staticmethod
    def softmax(x):
        """
        将输出层的数值转换为各类别的概率。

        每一列代表一个样本，所有类别概率之和为 1。

        输入：
            x: (n_classes, n_samples)

        输出：
            (n_classes, n_samples)
        """
        # 每一列分别减去该列最大值，防止 exp 数值溢出
        x_max = np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    # %% 交叉熵损失函数

    @staticmethod
    def cross_entropy(y_true, y_pred):
        """
        计算多分类交叉熵损失。

        输入：
            y_true: (n_classes, n_samples)，One-Hot 标签。
            y_pred: (n_classes, n_samples)，模型预测概率。

        输出：
            所有样本的平均交叉熵损失。
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred), axis=0)
        return np.mean(loss)

    # %% One-Hot 编码

    @staticmethod
    def one_hot(labels, output_size):
        """
        将数字类别标签转换为 One-Hot 编码。

        输入：
            labels: (n_samples,)，整数类别标签。
            output_size: 类别数量。

        输出：
            (output_size, n_samples)。
        """
        n = len(labels)
        result = np.zeros((output_size, n), dtype=np.uint8)

        for i in range(n):
            result[labels[i], i] = 1

        return result

    # %% 类别编码

    def _encode_labels(self, y):
        """
        将原始类别转换为模型内部使用的数字类别。

        例如：
            ["Normal", "Failure", "Normal"]
            →
            [1, 0, 1]

        类别映射保存在 class_to_index 中。
        """
        self.classes_ = np.unique(y)
        self.class_to_index = {
            label: i for i, label in enumerate(self.classes_)
        }

        return np.array(
            [self.class_to_index[label] for label in y],
            dtype=int
        )

    # %% 类别转换

    def _decode_labels(self, labels):
        """
        将模型内部的数字类别还原为原始类别。

        例如：
            [1, 0, 1]
            →
            ["Normal", "Failure", "Normal"]
        """
        return self.classes_[labels]

    # %% 初始化模型参数

    def _initialize_parameters(self):
        """
        根据网络结构初始化权重和偏置。

        隐藏层使用 He 初始化，适用于 ReLU 激活函数。
        输入层和输出层维度由 fit() 根据训练数据自动确定。
        """
        # 固定随机种子，保证每次运行得到相同的初始参数
        np.random.seed(self.random_state)
        layer_sizes = [self.input_size, *self.hidden_layers, self.output_size]
        self.weights, self.biases = [], []

        for i in range(len(layer_sizes) - 1):
            input_size, output_size = layer_sizes[i], layer_sizes[i + 1]
            W = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
            b = np.zeros((output_size, 1))
            self.weights.append(W)
            self.biases.append(b)

    # %% 前向传播

    def _forward(self, train):
        """
        执行一次前向传播。

        输入：
            train: (n_features, n_samples)

        返回：
            P: 最终预测概率。
            Z_list: 每层线性变换后的结果。
            H_list: 输入和各隐藏层激活后的结果。
        """
        H = train
        Z_list, H_list = [], [train]

        # ---------- 隐藏层 ----------

        for i in range(len(self.hidden_layers)):
            Z = self.weights[i] @ H + self.biases[i]
            H = self.relu(Z)
            Z_list.append(Z)
            H_list.append(H)

        # ---------- 输出层 ----------

        i = len(self.weights) - 1
        Z = self.weights[i] @ H + self.biases[i]
        P = self.softmax(Z)
        Z_list.append(Z)

        return P, Z_list, H_list

    # %% L2 正则化损失

    def _calculate_L2_loss(self):
        """
        计算所有权重的 L2 正则化惩罚项。

        返回：
            L2 正则化损失。
        """
        L2_Loss = (self.lambda_L2 / 2) * sum(
            np.sum(W ** 2) for W in self.weights
        )
        return L2_Loss

    # %% 反向传播

    def _backward(self, train, train_labels, P, Z_list, H_list):
        """
        根据 Loss 计算所有权重和偏置的梯度。

        输入：
            train: 输入数据。
            train_labels: One-Hot 标签。
            P: 前向传播得到的预测概率。
            Z_list: 各层 Z。
            H_list: 输入和各隐藏层 H。

        返回：
            dW_list: 每层权重的梯度。
            db_list: 每层偏置的梯度。
        """
        n, n_layers = train.shape[1], len(self.weights)
        dW_list, db_list = [None] * n_layers, [None] * n_layers

        # ---------- 输出层 ----------

        i = n_layers - 1
        dZ = (P - train_labels) / n
        dW_list[i] = dZ @ H_list[i].T + self.lambda_L2 * self.weights[i]
        db_list[i] = np.sum(dZ, axis=1, keepdims=True)
        dH = self.weights[i].T @ dZ

        # ---------- 隐藏层 ----------

        for i in range(n_layers - 2, -1, -1):
            # ReLU 导数
            dZ = dH * (Z_list[i] > 0)
            dW_list[i] = dZ @ H_list[i].T + self.lambda_L2 * self.weights[i]
            db_list[i] = np.sum(dZ, axis=1, keepdims=True)

            if i > 0:
                dH = self.weights[i].T @ dZ

        return dW_list, db_list

    # %% 梯度下降

    def _update_parameters(self, dW_list, db_list):
        """
        使用梯度下降更新网络参数。

        根据：
            W = W - learning_rate × dW
            b = b - learning_rate × db
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW_list[i]
            self.biases[i] -= self.learning_rate * db_list[i]

    # %% 开始训练

    # %% 开始训练

    # %% 开始训练

    def fit(self, X, y):
        """
        使用给定数据训练 MLP。

        输入：
            X: (n_samples, n_features)
               每一行代表一个样本。

            y: (n_samples,)
               每个样本对应的类别，可以是数字或字符串。

        标签处理：
            fit() 会自动识别所有类别，并建立数字编码。
            例如：
                "Failure" → 0
                "Normal"  → 1

        训练过程：
            类别编码
            → 数据预处理
            → 初始化参数
            → 前向传播
            → Loss
            → 反向传播
            → 梯度下降

        返回：
            self，使得可以继续调用其他模型方法。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X 必须是二维数组")
        if y.ndim != 1:
            raise ValueError("y 必须是一维数组")
        if X.shape[0] != len(y):
            raise ValueError("X 和 y 的样本数量必须一致")

        self.input_size = X.shape[1]

        # 自动将字符串/数字类别编码为 0, 1, 2, ...
        y = self._encode_labels(y)
        self.output_size = len(self.classes_)

        # sklearn 风格输入：(n_samples, n_features)
        # 内部计算：(n_features, n_samples)
        train = X.T
        train_labels = self.one_hot(y, self.output_size)

        self._initialize_parameters()
        self.loss_history, self.accuracy_history = [], []

        for epoch in range(self.epochs):

            # ---------- 前向传播 ----------

            P, Z_list, H_list = self._forward(train)

            # --------- L2 正则化惩罚项 ---------

            L2_Loss = self._calculate_L2_loss()

            # ---------- Loss ----------

            loss = self.cross_entropy(train_labels, P) + L2_Loss

            # ---------- 反向传播 ----------

            dW_list, db_list = self._backward(
                train, train_labels, P, Z_list, H_list
            )

            # ---------- 梯度下降 ----------

            self._update_parameters(dW_list, db_list)

            # ---------- 记录训练过程 ----------

            self.loss_history.append(loss)

            predictions = np.argmax(P, axis=0)
            true_labels = np.argmax(train_labels, axis=0)
            accuracy = np.mean(predictions == true_labels)

            self.accuracy_history.append(accuracy)

            # ---------- 每完成 10% 显示一次 ----------

            progress = epoch + 1
            interval = max(1, self.epochs // 10)

            if progress % interval == 0 or progress == self.epochs:
                print(
                    f"Epoch {progress}/{self.epochs}, "
                    f"Loss={loss:.6f}, "
                    f"Accuracy={accuracy:.4f}"
                )

        return self

    # %% 预测概率

    def predict_proba(self, X):
        """
        计算每个样本属于各类别的概率。

        输入：
            X: (n_samples, n_features)

        输出：
            (n_samples, n_classes)

        返回的概率列顺序与 classes_ 一致。

        例如：
            classes_ = ["Failure", "Normal"]

            [0.02, 0.98]
            表示：
                Failure: 2%
                Normal: 98%
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X 必须是二维数组")
        if X.shape[1] != self.input_size:
            raise ValueError(f"输入特征数量应为 {self.input_size}")

        P, _, _ = self._forward(X.T)
        return P.T

    # %% 预测类别

    def predict(self, X):
        """
        预测每个样本所属的类别。

        输入：
            X: (n_samples, n_features)

        输出：
            (n_samples,)

        返回原始类别，而不是模型内部使用的数字编码。

        例如：
            ["Normal", "Failure", "Normal"]
        """
        labels = np.argmax(self.predict_proba(X), axis=1)
        return self._decode_labels(labels)

    # %% 计算准确率

    def score(self, X, y):
        """
        计算模型的分类准确率。

        输入：
            X: (n_samples, n_features)
            y: (n_samples,)，可以是数字或字符串类别。

        输出：
            accuracy: 预测正确的样本比例。
        """
        return np.mean(self.predict(X) == np.asarray(y))