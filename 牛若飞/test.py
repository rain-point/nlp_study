from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
#生成训练数据
#X,y = make_classification(n_samples=150,n_features=10) #shape (150,10)
X,y = load_iris(return_X_y=True)
X = X[:100]
y = y[:100]
#数据拆分
#局部样本训练模型
#新样本数据模型表现不好
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#超参数
#lrs = [0.01,0.2,0.03,0.4]#调整学习率
lr = 0.1
epochs = 3000 #训练次数
#模型计算参数
def forward(theta,X,bias):
    Z = np.dot(theta,X.T) + bias
#将预测值转化为概率
    y_hat= 1/(1 + np.exp(-Z))
    return y_hat
#损失函数
def loss(y, y_hat):
    e = 1e-8
    return - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)
#计算梯度
def cal_gradient(y_hat,y,X):
    m = X.shape[-1]
    delta_theta = np.dot((y_hat - y),X)/m
    delta_bias = np.mean(y_hat - y)/m
    return delta_theta, delta_bias
# for lr in lrs:
#     #权重参数
theta = np.random.randn(1,4) #shape (1,4)
bias = 0
split_rates = [0.1, 0.3, 0.5, 0.7, 0.9]  
for sr in split_rates:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sr)
    #模型训练
    for i in range(epochs):
        #前向计算
        y_hat = forward(theta , X_train,bias)
        #计算损失
        loss_value = loss(y_train,y_hat)
        #计算梯度
        delta_theta, delta_bias = cal_gradient(y_hat,y_train,X_train)
        #更新梯度
        theta = theta - lr * delta_theta
        bias = bias - lr * delta_bias
        
        if i % 100 == 0:
            # 计算准确率
            acc = np.mean(np.round(y_hat) == y_train)  # [False,True,...,False] -> [0,1,...,0]
            print(f"epoch: {i}, loss: {np.mean(loss_value)}, acc: {acc}")
#模型推理
idx = np.random.randint(len(X_test))# 随机选择一个测试样本索引
x = X_test[idx]
y = y_test[idx]
predict = np.round(forward(theta,x,bias))
print(f"y: {y}, predict: {predict}")
