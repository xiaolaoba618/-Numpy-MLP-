# MyMLP

一个使用 NumPy 从零实现的多层感知机（MLP）分类器。

这是一个个人学习项目，主要用于理解和实践神经网络背后的数学原理。项目没有使用 PyTorch、TensorFlow 等深度学习框架，MLP 的核心计算均使用 NumPy 手动实现。

## 功能

- 使用 NumPy 从零实现 MLP
- 支持自定义隐藏层数量和结构
- ReLU 激活函数
- Softmax 输出
- 交叉熵损失
- L2 正则化
- 反向传播
- 梯度下降
- 自动处理字符串和数字类别
- 提供类似 scikit-learn 的 `fit()`、`predict()`、`predict_proba()`、`score()` 接口

## 安装

安装 NumPy：

```bash
pip install numpy
```

## 使用方法

```python
from xlbMLP import MLPClassifier

model = MLPClassifier(
    hidden_layers=[128, 64],
    learning_rate=0.01,
    epochs=100,
    lambda_L2=0.00001
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
accuracy = model.score(X_test, y_test)

print(predictions)
print(accuracy)
```

## 自定义网络结构

可以通过 hidden_layers 自由设置隐藏层数量和每层神经元数量：

```python
hidden_layers=[128, 64, 32]
```

表示：

输入层 → 128 → 64 → 32 → 输出层

例如：

```python
model = MLPClassifier(
    hidden_layers=[256, 128, 64],
    learning_rate=0.01,
    epochs=100
)
```

表示：

输入层 → 256 → 128 → 64 → 输出层

## 类别标签

模型可以直接处理字符串类别，不需要手动进行 Label Encoding：

```python
y = ["Normal", "Failure", "Normal", ...]
```

训练时模型会自动将类别转换为内部数字编码，预测时再还原为原始类别。

## 数学逻辑

整个 MLP 的训练过程遵循：

输入
 ↓
前向传播
 ↓
计算 Loss
 ↓
反向传播
 ↓
梯度下降
 ↓
更新参数

### 1. 前向传播

假设输入数据为：

$$
X \in \mathbb{R}^{d\times n}
$$

其中 $d$ 为特征数量，$n$ 为样本数量。

对于第 $l$ 层：

$$
Z^{(l)} = W^{(l)}H^{(l-1)}+b^{(l)}
$$

其中：

$$
H^{(0)}=X
$$

隐藏层使用 ReLU 激活函数：

$$
H^{(l)} = \mathrm{ReLU}(Z^{(l)})
$$

其中：

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

最后一层不使用 ReLU，而是使用 Softmax 得到各类别概率：

$$
P_i = \frac{e^{Z_i-\max(Z)}}{\sum_j e^{Z_j-\max(Z)}}
$$

减去 $\max(Z)$ 是为了提高数值计算的稳定性。

### 2. 交叉熵损失

假设真实标签经过 One-Hot 编码得到 $Y$，模型预测概率为 $P$。

多分类交叉熵为：

$$
L_{\mathrm{CE}} = -\frac{1}{n} \sum_{j=1}^{n} \sum_{i=1}^{K} Y_{ij}\log P_{ij}
$$

其中：

$n$：样本数量

$K$：类别数量

$Y_{ij}$：第 $j$ 个样本是否属于第 $i$ 类

$P_{ij}$：模型预测该样本属于第 $i$ 类的概率

### 3. L2 正则化

为了限制权重过大，在损失函数中加入 L2 正则化：

$$
L_{\mathrm{L2}} = \frac{\lambda}{2} \sum_l \left\|W^{(l)}\right\|_F^2
$$

因此总损失为：

$$
L = L_{\mathrm{CE}} + L_{\mathrm{L2}}
$$

对于权重，其梯度为：

$$
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L_{\mathrm{CE}}}{\partial W^{(l)}} + \lambda W^{(l)}
$$

### 4. 反向传播

Softmax 与交叉熵结合后，输出层的梯度可以直接写为：

$$
dZ^{(L)} = \frac{P-Y}{n}
$$

然后计算权重和偏置的梯度：

$$
dW^{(L)} = dZ^{(L)} \left(H^{(L-1)}\right)^T + \lambda W^{(L)}
$$

$$
db^{(L)} = \sum dZ^{(L)}
$$

梯度继续向前传播：

$$
dH^{(l)} = \left(W^{(l+1)}\right)^T dZ^{(l+1)}
$$

ReLU 的导数为：

$$
\mathrm{ReLU}'(x) = \begin{cases} 1, & x>0\\ 0, & x\leq0 \end{cases}
$$

因此：

$$
dZ^{(l)} = dH^{(l)} \odot \mathrm{ReLU}'(Z^{(l)})
$$

随后：

$$
dW^{(l)} = dZ^{(l)} \left(H^{(l-1)}\right)^T + \lambda W^{(l)}
$$

$$
db^{(l)} = \sum dZ^{(l)}
$$

其中 $\odot$ 表示元素对应相乘。

### 5. 梯度下降

使用梯度下降更新网络参数：

$$
W^{(l)} \leftarrow W^{(l)} - \eta dW^{(l)}
$$

$$
b^{(l)} \leftarrow b^{(l)} - \eta db^{(l)}
$$

其中 $\eta$ 为学习率。

### 6. 权重初始化

隐藏层使用 He 初始化：

$$
W^{(l)} \sim \mathcal{N}\left(0, \frac{2}{n_{\mathrm{in}}}\right)
$$

其中 $n_{\mathrm{in}}$ 为上一层神经元数量。

偏置初始化为：

$$
b^{(l)}=0
$$

He 初始化适用于使用 ReLU 激活函数的神经网络。

## 整体算法

整个模型可以概括为：

$$
\boxed{
\text{输入}
\rightarrow
\text{前向传播}
\rightarrow
\text{Loss}
\rightarrow
\text{反向传播}
\rightarrow
\text{梯度下降}
}
$$

重复上述过程若干个 Epoch，使损失函数逐渐降低。

## 项目结构

```text
MyMLP/
├── mlp.py
├── test.py
├── README.md
├── LICENSE
└── data/
    ├── train.csv
    └── test.csv
```

## 项目说明

这是一个个人学习和实践项目。

项目的主要目的是为了个人入门深度学习，了解背后数学原理，而不是构建一个用于生产环境的机器学习框架。

因此，本项目没有使用 PyTorch、TensorFlow 等高级深度学习框架。

## License

MIT 

