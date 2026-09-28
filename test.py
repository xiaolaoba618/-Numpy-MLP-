import pandas as pd
from xlbMLP import MLPClassifier

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# 标准化
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)

model = MLPClassifier(
    hidden_layers=[128, 64],
    learning_rate=0.01,
    epochs=400,
    lambda_L2=0.00001
)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

print("预测结果:", predictions.dtype)
print("真实结果:", y_test.dtype)
print("测试准确率:", model.score(X_test, y_test))