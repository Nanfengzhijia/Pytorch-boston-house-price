import numpy as np
from sklearn.datasets import load_boston
import torch
from torch import nn

# 数据加载
boston = load_boston()
x = boston['data']
y = boston['target']
print(x.shape)
print(y.shape)

# 将y转换形状
y = y.reshape(-1,1)

from sklearn.preprocessing import  MinMaxScaler

# 数据规范化
ss_input = MinMaxScaler()
x = ss_input.fit_transform(x)

# 将数据放到Torch中
x = torch.from_numpy(x).type(torch.FloatTensor)
y = torch.from_numpy(y).type(torch.FloatTensor)
# 数据集切分
from sklearn.model_selection import train_test_split
train_x, text_x, train_y, text_y = train_test_split(x, y, test_size=0.25)

# 构建网络
model = nn.Sequential(
    nn.Linear(13,16)
    nn.ReLU(),
    nn.Linear(16,1)
)
# 定义优化器和损失函数
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

# 训练
max_epoch = 300
iter_loss = []
for i in range(max_epoch):
# 前向传播
    y_pred = model(train_x)
# 计算损失函数
    loss = criterion(y_pred, train_y)
    iter_loss.append(loss.item())
# 清空上一轮的梯度
    optimizer.zero_grad()
# 反向传播
    loss.backward()
# 参数更新
    optimizer.step()

import matplotlib.pyplot as plt
# 绘制不同迭代过程的loss
x = np.arange(max_epoch)
y = np.array(iter_loss)
plt.plot(x,y)
plt.title('Loss Value in all interations')
plt.xlabel('Interation')
plt.ylabel('Mean loss Value')
plt.show

# 测试
output = model(text_x)
predict_list = output.detach().numpy()
print(predict_list)

# 真实值与预测值的散点图

x = np.arange(text_x.shape[0])
y1 = np.arange(predict_list)
y2 = np.arange(text_y)
line1 = plt.scatter(x,y1,c='red',Label='predict')
line2 = plt.scatter(x,y2,c='yellow',Label='real')
plt.lengend(loc = 'best')
plt.title('Prediction Vs Real')
plt.ylabel('House Price')
plt.show()
