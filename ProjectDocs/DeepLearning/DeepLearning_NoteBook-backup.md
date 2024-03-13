# 理论知识

## 相机几何

将**三维世界中的坐标点**（单位为米）映射到**二维图像平面**（单位为像素）的过程

### 针孔相机模型

![image-20240122104110785](./.assets/image-20240122104110785.png)

相似三角形

![](./.assets/image-20240122104130245.png)

现实世界空间点$P$，经过小孔$O$投影之后，落在物理成像平面$O'-x'-y'$上，成像点为$P'$

设$P$点坐标为
$$
P=[X, Y, Z]^T
$$
$P'$点坐标为
$$
P' =[X',Y',Z']^T
$$
物理成像平面到小孔的距离为$f$（焦距）

根据三角形相似关系
$$
\frac{Z}{f}=-\frac{X}{X'}=-\frac{Y}{Y'}
$$
负号表示成的像是倒立的，**我们可以等价的包图像平面对称到相机的前方**，把公式中的负号去掉

![](./.assets/image-20240122104921332.png)

式子更加简洁
$$
\frac{Z}{f}=\frac{X}{X'}=\frac{Y}{Y'}
$$
把$X',Y'$放入等式的左侧，整理得到
$$
X'=f\frac{X}{Z}\\
Y'=f\frac{Y}{Z}
$$
上述式子的单位都是统一的，都是米，$X',f,Z$等单位都是米，但是最终我们获得的是一个个像素，为了**描述传感器将感受到的光线转换为图像像素的过程**，我们设在物理成像平面上固定着一个**像素平面**，$o-u-v$，像素坐标是$[u,v]^T$

**像素坐标系：**通常的定义方式，原点$o'$位于图像的左上角，$u$轴向右与$x$轴平行，$v$轴向下与$y$轴平行，像素坐标系和成像平面之间，相差了**一个缩放+一个原点的平移**

![](./.assets/image-20240122105827261.png)

设像素坐标在$u$轴上缩放了$\alpha$倍，在$v$轴上缩放了$\beta$倍，同时原点平移了$[c_x,c_y]^T$,则**$P'$的坐标**与**像素坐标**$[u,v]$

之间的关系为
$$
u = \alpha X' + c_x\\
v = \beta Y' + c_y
$$
代入，把$\alpha x=f_x, \beta y = f_y$得到
$$
u = f_x \frac {X}{Z} + c_x\\
v = f_y \frac {Y}{Z} + c_y
$$
其中，$f$的单位为米，$\alpha,\beta$的单位为像素/米， 所以$f_x, f_y,c_x,c_y$的单位为像素，该式子写成矩阵更加简介，不过**左侧需要转换为齐次坐标，右侧是非齐次坐标**
$$
\begin{pmatrix}u\\v\\1\end{pmatrix}=\dfrac{1}{Z}\begin{pmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix}\begin{pmatrix}X\\Y\\Z\end{pmatrix}\stackrel{\text{def}}{=}\frac{1}{Z}\boldsymbol{K}\boldsymbol{P}
$$
习惯性把$Z$移到左侧
$$
Z\begin{pmatrix}u\\v\\1\end{pmatrix}=\begin{pmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix}\begin{pmatrix}X\\Y\\Z\end{pmatrix}\stackrel{\text{def}}{=}\boldsymbol{K}\boldsymbol{P}
$$
该式中，我们把中间的量的组成的矩阵称为**相机的内参数矩阵K**

在式子


$$
\begin{pmatrix}u\\v\\1\end{pmatrix}=\dfrac{1}{Z}\begin{pmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix}\begin{pmatrix}X\\Y\\Z\end{pmatrix}\stackrel{\text{def}}{=}\frac{1}{Z}\boldsymbol{K}\boldsymbol{P}
$$
$XYZ$使用的是$P$在相机坐标系下的坐标，但是实际上相机在运动，所以**$$P$$的相机坐标应该是它的世界坐标（记为$$P_w$$）根据相机的当前位姿变换到相机坐标系下的结果**

相机的位姿由它的旋转矩阵$$R$$和平移向量$$t$$来描述，则有
$$
ZP_{uv}=Z\begin{pmatrix}u\\v\\1\end{pmatrix}=KP=K(RP_w+t)=KTP_w
$$
注意后一个式子隐含了一次**齐次坐标到非齐次坐标的转换**，描述了$P$的世界坐标到像素坐标的投影关系。相机位姿$R,t$又称为**相机的外参数**

我们可以把一个世界坐标点先转换到相机坐标系，再除掉它最后一维的数值（即该点距离相机平面成像的深度），**相当于把最后一维进行归一化处理**，得到点$P$在相机归一化平面下的投影
$$
(RP_w + t) = [X, Y, Z]^T \rarr [\frac{X}{Z}, \frac{Y}{Z},1]^T
$$

![](./.assets/image-20240122115031646.png)

归一化坐标可以看成**相机前方$z=1$米处的点**，这个平面称为归一化平面，归一化平面左乘内参得到像素坐标，**所以我们可以把像素坐标$[u,v]^T$看成对归一化平面上的点的量化测量结果**

从这个模型中我们可以看出，如果对相机坐标同时乘以任意非零常数，**归一化坐标都是一样的**，也就是说**点的深度信息在投影过程中丢失了**，所以单目视觉无法获得像素点的深度值



### 畸变模型

为了获得更好的成像效果，在相机的前方加入了透镜，透镜的加入会对成像过程中光线的传播产生新的影响：

1. 透镜自身形状对光线传播的影响
2. 机械组装过程中，透镜和成像平面不可能完全重合

由透镜形状引起的畸变通常称为**径向畸变**包括**桶形畸变、枕形畸变**

机械组装中不能使透镜和成像平面严格平行，所以会引入**切向畸变**

![image-20240122124903192](./.assets/image-20240122124903192.png)

![](./.assets/image-20240122124930965.png)

考虑**归一化平面**上的任意一点$p$，它的坐标为$[x,y]^T$，也可以写成极坐标形式$[r,\theta]^T$，径向畸变可以看成坐标点沿着长度方向发生了变化（**也就是其距离原点的长度发生了变化**）。切向畸变可以看成坐标点沿着切线方向发生了变化（**也就是水平夹角发生了变化**）通常假设这些畸变呈现多项式关系即
$$
x_{distorted}=x(1+k_1r^2+k_2r^4+k_3r^6)\\
y_{distorted}=y(1+k_1r^2+k_2r^4+k_3r^6)
$$
其中$x_{distorted},y_{distorted}$是畸变后点的归一化坐标。另外对于**切向畸变**，可以使用另外两个参数$p_1,p_2$进行纠正
$$
x_{distorted}=x+2p_1xy+p_2(r^2+2x^2)\\
y_{distorted}=y+p_1(r^2+2y^2)+2p_2xy
$$
联合上述式子，对于相机坐标系中任意一点$P$，我们可以通过五个畸变系数，找到这个点在像素平面上的正确位置

1. 将三维空间点投影到归一化图像平面，设它的归一化坐标为$[x,y]^T$

2. 对归一化平面上的点计算径向畸变和切向畸变
    $$
    x_{distorted}=x(1+k_1r^2+k_2r^4+k_3r^6)+2p_1xy+p_2(r^2+2x^2)\\
    y_{distorted}=y(1+k_1r^2+k_2r^4+k_3r^6)+p_1(r^2+2y^2)+2p_2xy
    $$

3. 将畸变后的点通过内参数矩阵投影到像素平面，得到该点在像素上的正确位置
    $$
    u=f_xx_{disorted}+c_x\\
    v=f_yy_{disorted}+c_y
    $$

在上述纠正畸变的过程中，我们使用了五个畸变项，在实际应用中，可以灵活选择纠正模型，比如只选择$k_1,p_1,p_2$这三项

## 基础知识

### 感受野

在卷积神经网络中，决定某一层输出结果中一个元素所对应的输入层的区域大小，被称为感受野，**通俗的解释是，输出feature map上的一个单元对应输入层上的区域大小**

![](./.assets/image-20231212113540263.png)

**感受野计算公式（从后向前推算）**
$$
F(i) = (F(i + 1) - 1) × Stride + KernalSize
$$

$$
FeatureMap: F = 1\\
Pool1:F = (1 - 1) × 2 + 2 = 2\\
Conv1:F = (2 - 1) × 2 + 3 = 5
$$
也就是说明，一个像素对应着5×5的大小的区域

### 分组卷积

**分组卷积**好处是节省参数

传统的卷积需要的参数：假设kernal_size = k,  in_channel=Cin, out_channel = n
$$
参数 = k × k × C_{in} × n
$$
![](./.assets/image-20231212180242408.png)

分组卷积就是将输入的特征图，拆分成g个组，针对每一个组，进行卷积

假设kernal_size = k, in_channel = Cin out_channel = n, 分成g个组

单独看每个组，

输入的特征图channel = Cin / g  这一组卷积核的参数为 $k × k × \frac {C_{in}}{g} × \frac{n}{g}$,一共有g个组
$$
参数 =k × k × \frac {C_{in}}{g} × \frac{n}{g} × g
$$

![](./.assets/image-20231212180640510.png)

当为feature的每一个channel分配一个组，且卷积核的channel = 1，就是[DW卷积](###DepthWise卷积)

### DepthWise卷积

![image-20231212191037611](./.assets/image-20231212191037611.png)

普通卷积的计算量
$$
(D_k × D_k × M × N) × D_F × D_F
$$
DW  + PW 卷积的计算量
$$
(D_k × D_k × M) × D_F × D_F + (1 × 1 × M × N) ×  D_F × D_F
$$

### 通道注意力

#### SEnet

**Squeeze-and-Excitation Networks**

学习通道之间的相关性，自适应重新校准通道的特征响应

![](./.assets/20180423230918755.jpeg)

1. Squeeze：全局平均池化 C×H×W -> C×1×1
2. Excitation：两层全连接 + sigmod 限制在[0,1]
3. 特征重标定：Excitation得到的结果作为权重乘到特征图上的每个位置

#### CBAM

**Convolutional Block Attention Module**

![](./.assets/v2-ef1f5582aca80f0f2ccbb407c2b1f58e_r.jpg)

![](./.assets/v2-e2d538e5f082427047831eea3df70b94_r.jpg)

通道注意力&空间注意力

1. 经过最大池化和平均池化，通过共享全连接层，分别得到对应的特征，add起来
2. 通过最大池化和平均池化，通过卷积层，卷积得到空间特征图分布权重
3. 最后经过通道权重和空间权重，得到最终的特征图

#### SRM

**A Style-based Recalibration Module for Convolutional Neural Networks**

![在这里插入图片描述](./.assets/123.png)

1. 原始特征经过最大池化和平均池化得到C×D的权重
2. 经过全连接层得到C×1的权重
3. 然后再进行权重相乘求转换后的特征图

## 图像分类



### 图像处理

#### 全连接层

**全连接层前向计算**

全连接神经网络是指**任意两个相邻层之间的神经元全部互相连接**

![](./.assets/image-20231211143542288.png)

$X_1=\begin{bmatrix}\text{x}_1\\\text{x}_2\\\end{bmatrix}$是输入的特征向量（一般是卷积层最后的输出，展平后的一维向量），从第一层到第二层，神经元个数从2到3，也就是说从2个特征通过矩阵变换到了三个特征，矩阵$W_{12}X_1 = X_2$

$\mathrm{W}_{12}=\begin{bmatrix}w_{11}^{(1)}&\mathrm{w}_{21}^{(1)}\\\mathrm{W}_{12}^{(1)}&\mathrm{W}_{22}^{(1)}\\\mathrm{W}_{13}^{(1)}&\mathrm{W}_{23}^{(1)}\end{bmatrix}$，可以看到，每一行是一组权重，表示一个特征向量的加权之和，一共有3组这样的加权之和，所以转化成了3个特征，$\begin{bmatrix}w_{11}^{(1)}&\mathrm{w}_{21}^{(1)}\\\mathrm{W}_{12}^{(1)}&\mathrm{W}_{22}^{(1)}\\\mathrm{W}_{13}^{(1)}&\mathrm{W}_{23}^{(1)}\end{bmatrix}\begin{bmatrix}\text{x}_1\\\text{x}_2\\\end{bmatrix}=\begin{bmatrix}\text{a}_1\\\text{a}_2\\\text{a}_3\\\end{bmatrix}$，$\begin{bmatrix}\text{a}_1\\\text{a}_2\\\text{a}_3\\\end{bmatrix}$经过加上bias或者是激活函数，得到同样尺寸的向量，这里我偷个懒，不写新的符号了，还是$\begin{bmatrix}\text{a}_1\\\text{a}_2\\\text{a}_3\\\end{bmatrix}$，然后此时是3个特征，要变为2个特征，所以$\mathrm{W}_{23}=\begin{bmatrix}\mathrm{w}_{11}^{(2)}&\mathrm{w}_{21}^{(2)}&\mathrm{w}_{31}^{(2)}\\\mathrm{w}_{12}^{(2)}&\mathrm{w}_{22}^{(2)}&\mathrm{w}_{32}^{(2)}\\\end{bmatrix}$，可以看到每一行是一组权重，一共是2行，表示一个特征向量的加权和，一共有2组这样的加权和，所以转化为了2个特征，$\begin{bmatrix}\mathrm{w}_{11}^{(2)}&\mathrm{w}_{21}^{(2)}&\mathrm{w}_{31}^{(2)}\\\mathrm{w}_{12}^{(2)}&\mathrm{w}_{22}^{(2)}&\mathrm{w}_{32}^{(2)}\\\end{bmatrix}\begin{bmatrix}\text{a}_1\\\text{a}_2\\\text{a}_3\\\end{bmatrix}=\begin{bmatrix}\text{y}_1\\\text{y}_2\\\end{bmatrix}$，经过激活函数或者是bias，得到最终的输出特征，经过softmax得到最终的概率分布，敲定最终的输出结果，[参考](###交叉熵损失)

#### 卷积层

1. 卷积核的channel = 输入特征图的channel
2. 输出特征图的channel = 卷积核的个数
3. 卷积层**bias**，就是在卷积相乘之后的结果加上一个偏置
4. 卷积层之后加上**激活函数**，就是在卷积之后的结果带入激活函数中，得到输出值

![](./.assets/image-20231211140658484.png)



卷积：

1. 输入特征图的尺寸： $W_1 × H_1 × C_1$
2. 输出特征图的尺寸：$W_2 × H_2 × C_2$
3. 卷积核尺寸：$F × F × C_1$ ，一共有$K$个卷积核
4. 步长：$S$
5. 填充：$P$

$$
W_2 = \frac {W_1 + 2P - F}{S} + 1 \\
H_2 = \frac {H_1 + 2P - F}{S} + 1 \\
C_2 = K
$$



#### 池化层

1. 没有训练参数
2. 只改变特征图的W和H，不改变channel，因为池化是在每个channel上进行的
3. 一般设置size和stride相同

![image-20231211142444528](./.assets/image-20231211142444528.png)

池化：

1. 输入特征图的尺寸： $W_1 × H_1 × C_1$
2. 输出特征图的尺寸：$W_2 × H_2 × C_2$
3. 池化核尺寸：$F × F$ ，池化一般是在每个channel上进行的
4. 步长：$S$
5. 填充：$P$

$$
W_2 = \frac {W_1 + 2P - F}{S} + 1 \\
H_2 = \frac {H_1 + 2P - F}{S} + 1 \\
C_2 = C_1
$$





### Lenet

![image-20231211172341769](./.assets/image-20231211172341769.png)

1. 创建模型类

   ```python
   import torch.nn as nn
   import torch.nn.functional as F
   
   
   class LeNet(nn.Module):
       def __init__(self):
           super(LeNet, self).__init__()
           self.conv1 = nn.Conv2d(3, 16, 5) # in_channels = 3 out_channels = 16 kernal_size = 5
           self.pool1 = nn.MaxPool2d(2, 2)  # kernal_size = 2 stride = 2
           self.conv2 = nn.Conv2d(16, 32, 5)
           self.pool2 = nn.MaxPool2d(2, 2)
           self.fc1 = nn.Linear(32*5*5, 120)
           self.fc2 = nn.Linear(120, 84)
           self.fc3 = nn.Linear(84, 10)
   
       def forward(self, x):            # input(3, 32, 32) 
           x = F.relu(self.conv1(x))    # output(16, 28, 28)
           x = self.pool1(x)            # output(16, 14, 14)
           x = F.relu(self.conv2(x))    # output(32, 10, 10)
           x = self.pool2(x)            # output(32, 5, 5)
           x = x.view(-1, 32*5*5)       # output(32*5*5)  (第0维为-1，占位符这里是batch_size)
           x = F.relu(self.fc1(x))      # output(120)
           x = F.relu(self.fc2(x))      # output(84)
           x = self.fc3(x)              # output(10)
           return x
   
   
   # 测试
   import torch
   # [N, C, H, W]
   input1 = torch.rand([2, 3, 32, 32]) # 生成一个该尺寸的tensor
   # print(input1)
   model = LeNet()
   # print(model)
   output = model(input1)
   print(output)
   """
   tensor([[ 0.0324,  0.0964,  0.1025, -0.0259, -0.1406,  0.0447, -0.0958, -0.0039,
            -0.0606, -0.0549],
           [ 0.0319,  0.0999,  0.1022, -0.0245, -0.1447,  0.0388, -0.0978, -0.0013,
            -0.0618, -0.0487]], grad_fn=<AddmmBackward0>)
   """
   
   ```

2. train.py

   ```python
   import torch
   import torchvision
   import torch.nn as nn
   from model import LeNet
   import torch.optim as optim
   import torchvision.transforms as transforms
   
   
   def main():
       transform = transforms.Compose(
           [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
   
       # 50000张训练图片
       # 第一次使用时要将download设置为True才会自动去下载数据集
       # 下载到当前工作区上一级的datasets目录
       train_set = torchvision.datasets.CIFAR10(root='../datasets', train=True,
                                                download=True, transform=transform)
       train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                                  shuffle=True, num_workers=0)
   
       # 10000张验证图片
       # 第一次使用时要将download设置为True才会自动去下载数据集
       val_set = torchvision.datasets.CIFAR10(root='../datasets', train=False,
                                              download=True, transform=transform)
       val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                                shuffle=False, num_workers=0)
       val_data_iter = iter(val_loader)
       val_image, val_label = next(val_data_iter)
       # print(val_image.shape)
       # print(val_label)
       # 元组类型，括号圈起来的不能改变
       # classes = ('plane', 'car', 'bird', 'cat',
       #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
   
       net = LeNet()
   
       loss_function = nn.CrossEntropyLoss()
       optimizer = optim.Adam(net.parameters(), lr=0.001)
   
       # 将训练集迭代5轮
       for epoch in range(5):  # loop over the dataset multiple times
   
           running_loss = 0.0
           for step, data in enumerate(train_loader, start=0):
               # get the inputs; data is a list of [inputs, labels]
               inputs, labels = data
   
               # zero the parameter gradients
               # 清除历史梯度
               optimizer.zero_grad()
               # forward + backward + optimize
               outputs = net(inputs)
               loss = loss_function(outputs, labels)
               loss.backward()   # 反向传播
               optimizer.step()  # 参数更新
   
               # print statistics
               running_loss += loss.item()
               if step % 500 == 499:    # print every 500 mini-batches执行一次验证
                   with torch.no_grad():# 不计算损失梯度，因为是验证环节
                       outputs = net(val_image)  # [batch, 10]
                       predict_y = torch.max(outputs, dim=1)[1] # [batch] 值为最大概率的下标
                       # 到.sum计算的是tensor .item转化为数值 
                       # val_label.size(0)是获取label第0维的长度
                       accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
   
                       print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                             (epoch + 1, step + 1, running_loss / 500, accuracy))
                       running_loss = 0.0
   
       print('Finished Training')
       
       # 保存权重参数
       save_path = './Lenet.pth'
       torch.save(net.state_dict(), save_path)
   
   
   if __name__ == '__main__':
       main()
   
   ```

   

3. predict.py

   

### AlexNet

1. 首次使用GPU训练

2. 用ReLu函数代替传统的Sigmod和Tanh激活函数

3. 使用LRN局部响应归一化（LRN作用不大，舍弃）

4. 全连接层使用随机失活DropOut，减少过拟合现象

    ![](./.assets/image-20231212100754996.png)

    ![](./.assets/image-20231212104728195.png)

**网络结构**



![image-20231211173131954](./.assets/image-20231211173131954.png)

![img](./.assets/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAWmthaXNlbg==,size_20,color_FFFFFF,t_70,g_se,x_16.png)

1. 网络搭建

   ```python
   import torch.nn as nn
   import torch
   
   
   class AlexNet(nn.Module):
       # 默认1000分类 不初始化权重
       def __init__(self, num_classes=1000, init_weights=False):
           super(AlexNet, self).__init__()
           # nn.Sequential
           # 卷积池化层
           self.features = nn.Sequential(                              # input[3, 224, 224]
               nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # output[48, 55, 55]
               nn.ReLU(inplace=True),# 增加计算量，降低内存使用 inpace方法
               nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
               nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
               nn.ReLU(inplace=True),
               nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
               nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
               nn.ReLU(inplace=True),
               nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
               nn.ReLU(inplace=True),
               nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
               nn.ReLU(inplace=True),
               nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
           )
           # 全连接层
           self.classifier = nn.Sequential(
               nn.Dropout(p=0.5), # 随机失活
               nn.Linear(128 * 6 * 6, 2048),
               nn.ReLU(inplace=True),
               nn.Dropout(p=0.5),
               nn.Linear(2048, 2048),
               nn.ReLU(inplace=True),
               nn.Linear(2048, num_classes),
           )
           if init_weights:
               self._initialize_weights()
   
       def forward(self, x):
           x = self.features(x)
           # 从第一维度开始打平
           x = torch.flatten(x, start_dim=1)
           x = self.classifier(x)
           return x
   
       def _initialize_weights(self):
           # 遍历实例的module
           for m in self.modules():
               if isinstance(m, nn.Conv2d):
                   nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                   if m.bias is not None:
                       nn.init.constant_(m.bias, 0)
               elif isinstance(m, nn.Linear):
                   nn.init.normal_(m.weight, 0, 0.01)
                   nn.init.constant_(m.bias, 0)
   
   ```

   

2. train.py

   ```python
   import os
   import sys
   import json
   
   import torch
   import torch.nn as nn
   from torchvision import transforms, datasets, utils
   import matplotlib.pyplot as plt
   import numpy as np
   import torch.optim as optim
   from tqdm import tqdm
   
   from model import AlexNet
   
   
   def main():
       device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       print("using {} device.".format(device))
   
       data_transform = {
           "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
           "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
   
       data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path 先获取当前路径，返回到上上层
       image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path 得到数据集的路径
       assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
       train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                            transform=data_transform["train"])
       train_num = len(train_dataset)
   
       # 处理类别，生成索引和类别对应的json文件
       # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
       flower_list = train_dataset.class_to_idx
       cla_dict = dict((val, key) for key, val in flower_list.items()) # 键值对互换位置
       # write dict into json file
       json_str = json.dumps(cla_dict, indent=4)
       with open('class_indices.json', 'w') as json_file:
           json_file.write(json_str)
   
       batch_size = 32
       nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
       print('Using {} dataloader workers every process'.format(nw))
   
       train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)
   
       validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                               transform=data_transform["val"])
       val_num = len(validate_dataset)
       validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                     batch_size=4, shuffle=True,
                                                     num_workers=nw)
   
       print("using {} images for training, {} images for validation.".format(train_num,
                                                                              val_num))
       # 展示图片
       # test_data_iter = iter(validate_loader)
       # test_image, test_label = test_data_iter.next()
       
       # def imshow(img):
       #     img = img / 2 + 0.5  # unnormalize
       #     npimg = img.numpy()
       #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
       #     plt.show()
       
       # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
       # imshow(utils.make_grid(test_image))
   
       # 实例化一个网络
       net = AlexNet(num_classes=5, init_weights=True)
   
       net.to(device)
       loss_function = nn.CrossEntropyLoss()
       # Adam优化器
       optimizer = optim.Adam(net.parameters(), lr=0.0002)
   
       epochs = 20
       save_path = './AlexNet.pth'
       best_acc = 0.0
       train_steps = len(train_loader)
       for epoch in range(epochs):
           # train
           net.train() # 这样配合net.train 可以保证只在训练阶段DropOut 和 Batch Normalization
           running_loss = 0.0 
           train_bar = tqdm(train_loader, file=sys.stdout)
           for step, data in enumerate(train_bar):
               images, labels = data
               optimizer.zero_grad()
               # forward + backward + optimize
               outputs = net(images.to(device))
               loss = loss_function(outputs, labels.to(device))
               loss.backward()
               optimizer.step()
   
               # print statistics
               # 累加各个batch的损失
               running_loss += loss.item()
   
               # 输出每个batch的损失
               train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                        epochs,
                                                                        loss)
   
           # validate
           net.eval()
           acc = 0.0  # accumulate accurate number / epoch 每个epoch的准确率
           # 每个epoch遍历一整个验证集
           with torch.no_grad():
               val_bar = tqdm(validate_loader, file=sys.stdout)
               for val_data in val_bar:
                   val_images, val_labels = val_data
                   outputs = net(val_images.to(device))
                   predict_y = torch.max(outputs, dim=1)[1]
                   acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
   
           val_accurate = acc / val_num
           # running loss  / train_steps(训练集batch的数量)
           print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                 (epoch + 1, running_loss / train_steps, val_accurate))
   
           if val_accurate > best_acc:
               best_acc = val_accurate
               torch.save(net.state_dict(), save_path)
   
       print('Finished Training')
   
   
   if __name__ == '__main__':
       main()
   
   ```

   

### VGG

1. 通过堆叠多个3*3的卷积核代替大尺度卷积核来**减少所需要的参数**

   ![](./.assets/image-20231212113325439.png)

   [感受野](###感受野)

   参数 = 卷积核长 * 卷积核宽  * 卷积核channel * 卷积核个数

   7*7参数 = 7 * 7 * C * C = 49C^2

   3个 3*3参数 = 3 * 3 *C *C +  3 * 3 *C *C +  3 * 3 *C *C = 27C^2
   
   

**网络结构**

conv的size = 3 stride = 1 padding = 1  （长宽不变）

maxpool的size = 2 stride = 2  （长宽变为一半）

  

![一文读懂VGG网络](./.assets/v2-dfe4eaaa4450e2b58b38c5fe82f918c0_1440w.png)



![](./.assets/image-20231212113346476.png)



1. 网络搭建

   ```python
   import torch.nn as nn
   import torch
   
   # official pretrain weights
   model_urls = {
       'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
       'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
       'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
       'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
   }
   
   
   class VGG(nn.Module):
       # 传入的feature是借助cfg生成的特征提取features的sequential
       def __init__(self, features, num_classes=1000, init_weights=False):
           super(VGG, self).__init__()
           self.features = features
           self.classifier = nn.Sequential(
               nn.Linear(512*7*7, 4096),
               nn.ReLU(True),
               nn.Dropout(p=0.5),
               nn.Linear(4096, 4096),
               nn.ReLU(True),
               nn.Dropout(p=0.5),
               nn.Linear(4096, num_classes)
           )
           if init_weights:
               self._initialize_weights()
       # x是输入的图像数据
       def forward(self, x):
           # N x 3 x 224 x 224
           x = self.features(x)
           # N x 512 x 7 x 7
           x = torch.flatten(x, start_dim=1)
           # N x 512*7*7
           x = self.classifier(x)
           return x
   
       def _initialize_weights(self):
           for m in self.modules():
               if isinstance(m, nn.Conv2d):
                   # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                   nn.init.xavier_uniform_(m.weight)
                   if m.bias is not None:
                       nn.init.constant_(m.bias, 0)# 偏置默认初始化为0
               elif isinstance(m, nn.Linear):
                   nn.init.xavier_uniform_(m.weight)
                   # nn.init.normal_(m.weight, 0, 0.01)
                   nn.init.constant_(m.bias, 0)
   
   
   # 传入配置列表构建网络
   def make_features(cfg: list):
       layers = []
       # 初始化第一层的输入
       in_channels = 3
       for v in cfg:
           if v == "M":
               # 创建最大池化下采样层（因为池化核都一样大小）
               layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
           else:
               # 创建卷积层
               # 输出通道对应当前卷积核的个数（卷积核的长宽和padding都一样）
               conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
               layers += [conv2d, nn.ReLU(True)]
               # 输出变为下一层
               in_channels = v
       # 返回nn.sequentioal 非关键字参数传入
       return nn.Sequential(*layers)
   
   # 配置文件
   # 数值是卷积层卷积核个数
   # ‘M’是最大池化层
   cfgs = {
       'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
   }
   
   # 实例化vgg
   def vgg(model_name="vgg16", **kwargs):
       assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
       cfg = cfgs[model_name]
   
       model = VGG(make_features(cfg), **kwargs) # **kwargs可变长度的字典变量，因为输入有多个，可以传参一部分
       return model
   
   
   # vgg_model = vgg(model_name="vgg16")
   ```

   

2. train.py

   ```python
   import os
   import sys
   import json
   
   import torch
   import torch.nn as nn
   from torchvision import transforms, datasets
   import torch.optim as optim
   from tqdm import tqdm
   
   from model import vgg
   
   
   def main():
       device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       print("using {} device.".format(device))
   
       data_transform = {
           "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
           "val": transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
   
       data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
       image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
       assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
       train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                            transform=data_transform["train"])
       train_num = len(train_dataset)
   
       # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
       flower_list = train_dataset.class_to_idx
       cla_dict = dict((val, key) for key, val in flower_list.items())
       # write dict into json file
       json_str = json.dumps(cla_dict, indent=4)
       with open('class_indices.json', 'w') as json_file:
           json_file.write(json_str)
   
       batch_size = 32
       nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
       print('Using {} dataloader workers every process'.format(nw))
   
       train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)
   
       validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                               transform=data_transform["val"])
       val_num = len(validate_dataset)
       validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                     batch_size=batch_size, shuffle=False,
                                                     num_workers=nw)
       print("using {} images for training, {} images for validation.".format(train_num,
                                                                              val_num))
   
       # test_data_iter = iter(validate_loader)
       # test_image, test_label = test_data_iter.next()
   
       # 构建模型
       model_name = "vgg16"
       net = vgg(model_name=model_name, num_classes=5, init_weights=True)
   
       net.to(device)
       loss_function = nn.CrossEntropyLoss()
       optimizer = optim.Adam(net.parameters(), lr=0.0001)
   
       epochs = 30
       best_acc = 0.0
       save_path = './{}Net.pth'.format(model_name)
       train_steps = len(train_loader)
       for epoch in range(epochs):
           # train
           net.train()
           running_loss = 0.0
           train_bar = tqdm(train_loader, file=sys.stdout)
           for step, data in enumerate(train_bar):
               images, labels = data
               optimizer.zero_grad()
               outputs = net(images.to(device))
               loss = loss_function(outputs, labels.to(device))
               loss.backward()
               optimizer.step()
   
               # print statistics
               running_loss += loss.item()
   
               train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                        epochs,
                                                                        loss)
   
           # validate
           net.eval()
           acc = 0.0  # accumulate accurate number / epoch
           with torch.no_grad():
               val_bar = tqdm(validate_loader, file=sys.stdout)
               for val_data in val_bar:
                   val_images, val_labels = val_data
                   outputs = net(val_images.to(device))
                   predict_y = torch.max(outputs, dim=1)[1]
                   acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
   
           val_accurate = acc / val_num
           print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                 (epoch + 1, running_loss / train_steps, val_accurate))
   
           if val_accurate > best_acc:
               best_acc = val_accurate
               torch.save(net.state_dict(), save_path)
   
       print('Finished Training')
   
   
   if __name__ == '__main__':
       main()
   
   ```

### GoogleNet

1. 引入Inception结构，融合不同尺度的特征信息
2. 使用1×1的卷积核进行降维以及映射处理
3. 添加两个辅助分类器帮助训练
4. 丢弃全连接层，使用平均池化层，大大减少模型的参数



![image-20231212122011296](./.assets/image-20231212122011296.png)

**网络结构**



![](./.assets/format,png#pic_center.png)



**Inception结构**

![](./.assets/image-20231212122212142.png)

串行改成并行，从上一层过来，经过不同的卷积核，然后按照channel进行拼接，**高度和宽度必须一致**

![](./.assets/image-20231212122254294.png)

使用1*1的卷积核**进行降维**，channel维度降维，参数量减少了

![](./.assets/image-20231212122419888.png)

**辅助分类器**

![](./.assets/image-20231212122721951.png)

平均池化核是 5*5 stride是3，在inception（4a）和inception（4d）处进行辅助分类



1. 模型搭建

   ```python
   import torch.nn as nn
   import torch
   import torch.nn.functional as F
   
   # 主网络
   class GoogLeNet(nn.Module):
       # 类别个数、是否使用辅助分类器，是否初始化权重
       def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
           super(GoogLeNet, self).__init__()
           self.aux_logits = aux_logits
   
           self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
           self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True) #最大池化下采样，计算出小数，就向上取整
   
           self.conv2 = BasicConv2d(64, 64, kernel_size=1)
           self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
           self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
   
           self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
           self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
           self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
   
           self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
           self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
           self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
           self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
           self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
           self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
   
           self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
           self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
   
           # 最后创建辅助分类器
           if self.aux_logits:
               self.aux1 = InceptionAux(512, num_classes)
               self.aux2 = InceptionAux(528, num_classes)
   
           # 平均池化下采样层
           self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
           self.dropout = nn.Dropout(0.4)
           self.fc = nn.Linear(1024, num_classes)
   
           # 是否初始化权重
           if init_weights:
               self._initialize_weights()
   
       def forward(self, x):
           # N x 3 x 224 x 224
           x = self.conv1(x)
           # N x 64 x 112 x 112
           x = self.maxpool1(x)
           # N x 64 x 56 x 56
           x = self.conv2(x)
           # N x 64 x 56 x 56
           x = self.conv3(x)
           # N x 192 x 56 x 56
           x = self.maxpool2(x)
   
           # N x 192 x 28 x 28
           x = self.inception3a(x)
           # N x 256 x 28 x 28
           x = self.inception3b(x)
           # N x 480 x 28 x 28
           x = self.maxpool3(x)
           # N x 480 x 14 x 14
           x = self.inception4a(x)
           # N x 512 x 14 x 14
           if self.training and self.aux_logits:    # eval model lose this layer
               aux1 = self.aux1(x)
   
           x = self.inception4b(x)
           # N x 512 x 14 x 14
           x = self.inception4c(x)
           # N x 512 x 14 x 14
           x = self.inception4d(x)
           # N x 528 x 14 x 14
           if self.training and self.aux_logits:    # eval model lose this layer
               aux2 = self.aux2(x)
   
           x = self.inception4e(x)
           # N x 832 x 14 x 14
           x = self.maxpool4(x)
           # N x 832 x 7 x 7
           x = self.inception5a(x)
           # N x 832 x 7 x 7
           x = self.inception5b(x)
           # N x 1024 x 7 x 7
   
           x = self.avgpool(x)
           # N x 1024 x 1 x 1
           x = torch.flatten(x, 1)
           # N x 1024
           x = self.dropout(x)
           x = self.fc(x)
           # N x 1000 (num_classes)
           if self.training and self.aux_logits:   # eval model lose this layer
               return x, aux2, aux1   # 返回辅助分类器和主分支结果
           return x                   # 返回主分支结果
   
       def _initialize_weights(self):
           for m in self.modules():
               if isinstance(m, nn.Conv2d):
                   nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                   if m.bias is not None:
                       nn.init.constant_(m.bias, 0)
               elif isinstance(m, nn.Linear):
                   nn.init.normal_(m.weight, 0, 0.01)
                   nn.init.constant_(m.bias, 0)
   
   # 小模块 inception结构
   class Inception(nn.Module):
       def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
           super(Inception, self).__init__()
   
           # 分支1
           self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
   
           # 分支2
           self.branch2 = nn.Sequential(
               BasicConv2d(in_channels, ch3x3red, kernel_size=1),
               BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
           )
           # 分支3
           self.branch3 = nn.Sequential(
               BasicConv2d(in_channels, ch5x5red, kernel_size=1),
               # 在官方的实现中，其实是3x3的kernel并不是5x5，这里我也懒得改了，具体可以参考下面的issue
               # Please see https://github.com/pytorch/vision/issues/906 for details.
               BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
           )
           # 分支4
           self.branch4 = nn.Sequential(
               nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
               BasicConv2d(in_channels, pool_proj, kernel_size=1)
           )
   
       def forward(self, x):
           branch1 = self.branch1(x)
           branch2 = self.branch2(x)
           branch3 = self.branch3(x)
           branch4 = self.branch4(x)
   
           outputs = [branch1, branch2, branch3, branch4]
           # 合并的维度，在深度channel维度上进行合并
           return torch.cat(outputs, 1)
   
   # 辅助分类器
   class InceptionAux(nn.Module):
       def __init__(self, in_channels, num_classes):
           super(InceptionAux, self).__init__()
           self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
           self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]
   
           self.fc1 = nn.Linear(2048, 1024)
           self.fc2 = nn.Linear(1024, num_classes)
   
       def forward(self, x):
           # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
           x = self.averagePool(x)
           # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
           # 1 * 1的卷积核降维，哪怕深度不一样，也可以进行降维到同样的深度
           x = self.conv(x)
           # N x 128 x 4 x 4
           x = torch.flatten(x, 1)
           # 原论文是70%概率随机失活
           x = F.dropout(x, 0.5, training=self.training)
           # N x 2048
           x = F.relu(self.fc1(x), inplace=True)
           x = F.dropout(x, 0.5, training=self.training)
           # N x 1024
           x = self.fc2(x)
           # N x num_classes
           return x
   
   # 小模块 卷积 + relu
   class BasicConv2d(nn.Module):
       def __init__(self, in_channels, out_channels, **kwargs):
           super(BasicConv2d, self).__init__()
           self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
           self.relu = nn.ReLU(inplace=True)
   
       def forward(self, x):
           x = self.conv(x)
           x = self.relu(x)
           return x
   
   ```

2. train.py

   ```python
   import os
   import sys
   import json
   
   import torch
   import torch.nn as nn
   from torchvision import transforms, datasets
   import torch.optim as optim
   from tqdm import tqdm
   
   from model import GoogLeNet
   
   
   def main():
       device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       print("using {} device.".format(device))
   
       data_transform = {
           "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
           "val": transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
   
       data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
       image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
       assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
       train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                            transform=data_transform["train"])
       train_num = len(train_dataset)
   
       # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
       flower_list = train_dataset.class_to_idx
       cla_dict = dict((val, key) for key, val in flower_list.items())
       # write dict into json file
       json_str = json.dumps(cla_dict, indent=4)
       with open('class_indices.json', 'w') as json_file:
           json_file.write(json_str)
   
       batch_size = 32
       nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
       print('Using {} dataloader workers every process'.format(nw))
   
       train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)
   
       validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                               transform=data_transform["val"])
       val_num = len(validate_dataset)
       validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                     batch_size=batch_size, shuffle=False,
                                                     num_workers=nw)
   
       print("using {} images for training, {} images for validation.".format(train_num,
                                                                              val_num))
   
       # test_data_iter = iter(validate_loader)
       # test_image, test_label = test_data_iter.next()
   
       # 定义网络
       net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
       # 如果要使用官方的预训练权重，注意是将权重载入官方的模型，不是我们自己实现的模型
       # 官方的模型中使用了bn层以及改了一些参数，不能混用
       # import torchvision
       # net = torchvision.models.googlenet(num_classes=5)
       # model_dict = net.state_dict()
       # # 预训练权重下载地址: https://download.pytorch.org/models/googlenet-1378be20.pth
       # pretrain_model = torch.load("googlenet.pth")
       # del_list = ["aux1.fc2.weight", "aux1.fc2.bias",
       #             "aux2.fc2.weight", "aux2.fc2.bias",
       #             "fc.weight", "fc.bias"]
       # pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
       # model_dict.update(pretrain_dict)
       # net.load_state_dict(model_dict)
       net.to(device)
       loss_function = nn.CrossEntropyLoss()
       optimizer = optim.Adam(net.parameters(), lr=0.0003)
   
       epochs = 30
       best_acc = 0.0
       save_path = './googleNet.pth'
       train_steps = len(train_loader)
       for epoch in range(epochs):
           # train
           net.train()
           running_loss = 0.0
           train_bar = tqdm(train_loader, file=sys.stdout)
           for step, data in enumerate(train_bar):
               images, labels = data
               optimizer.zero_grad()
               logits, aux_logits2, aux_logits1 = net(images.to(device))
               # 三个损失函数
               loss0 = loss_function(logits, labels.to(device))
               loss1 = loss_function(aux_logits1, labels.to(device))
               loss2 = loss_function(aux_logits2, labels.to(device))
               # 加权损失
               loss = loss0 + loss1 * 0.3 + loss2 * 0.3
               loss.backward()
               optimizer.step()
   
               # print statistics
               running_loss += loss.item()
   
               train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                        epochs,
                                                                        loss)
   
           # validate
           net.eval()
           acc = 0.0  # accumulate accurate number / epoch
           with torch.no_grad():
               val_bar = tqdm(validate_loader, file=sys.stdout)
               for val_data in val_bar:
                   val_images, val_labels = val_data
                   # 只有一个输出、不需要管辅助分类器
                   outputs = net(val_images.to(device))  # eval model only have last output layer
                   predict_y = torch.max(outputs, dim=1)[1]
                   acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
   
           val_accurate = acc / val_num
           print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                 (epoch + 1, running_loss / train_steps, val_accurate))
   
           if val_accurate > best_acc:
               best_acc = val_accurate
               torch.save(net.state_dict(), save_path)
   
       print('Finished Training')
   
   
   if __name__ == '__main__':
       main()
   
   ```

   

### ResNet

![](./.assets/image-20231212130858057.png)

1. 超级深的网络结构（突破1000层）

2. 提出residual模块

3. 使用Batch Normalization加速训练（丢弃DropOut方法）

   ![img](./.assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70.jpeg)

![image-20231212132108217](./.assets/image-20231212132108217.png)



**Residual模块**

![](./.assets/image-20231212131539154.png)

**主分支和侧分支shortcut的高 宽 channel必须相同**，也就是shape完全相同，**相加是两个矩阵在相同维度上数值相加，而不是concat**



**18 34 的残差结构**

![](./.assets/image-20231212165233584.png)

![](./.assets/image-20231212132523239.png)

**深层的 残差结构**

![](./.assets/image-20231212170458291.png)

![](./.assets/image-20231212170507747.png)

**虚线表示的residual模块实际上就是保证残差连接的shape具有一致性**，optionB方法，将输入部分的高宽和深度与主线的高宽和深度一直，事实上，只有不同的残差模块之间才会有这样的虚线连接形式，相同的残差模块之间不会用虚线连接shortcut，因为不需要，shape相同



**Batch Normalization**

[转到本篇](###BatchNormlization)[博文](https://blog.csdn.net/qq_37541097/article/details/104434557?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170236993316800227428091%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=170236993316800227428091&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-104434557-null-null.nonecase&utm_term=Batch&spm=1018.2226.3001.4450)

BN就是在训练过程中，针对每一个Batch在所有Feature Map的相同的channel上进行归一化操作

![img](./.assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70-1702370173866-9.png)

使用BN时需要**注意**的点：

1. 训练时要将traning参数设置为True，在验证时将trainning参数设置为False。在pytorch中可通过创建模型的model.train()和model.eval()方法控制
2. batch size尽可能设置大点，设置小后表现可能很糟糕，设置的越大求的均值和方差越接近整个训练集的均值和方差。
3. 建议将bn层放在卷积层（Conv）和激活层（例如Relu）之间，且卷积层不要使用偏置bias，因为没有用。





1. 模型搭建

   ```python
   import torch.nn as nn
   import torch
   
   
   # 18层34层残差结构
   # 既要有虚线连接的残差结构
   # 也要有实线连接的残差结构
   class BasicBlock(nn.Module):
       # expansion是一个标志，层次比较resnet有的残差结构卷积核的个数不相同，这里定义了一个倍数，在这里用不到
       expansion = 1
       # 两种残差结构 conv1和downsample stride不同，要传入一个stride
       # dowmsample是虚线shortcut和实线shortcut
       def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
           super(BasicBlock, self).__init__()
           # padding为1可以同时满足stride=1和stride=2的卷积，偏置不取
           self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                  kernel_size=3, stride=stride, padding=1, bias=False)
           # bn只是在每个channel上归一化，没改变channel
           self.bn1 = nn.BatchNorm2d(out_channel)
           self.relu = nn.ReLU()
           # 这个stride不是传入的stride，都是1
           self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                  kernel_size=3, stride=1, padding=1, bias=False)
           self.bn2 = nn.BatchNorm2d(out_channel)
           # shortcut
           self.downsample = downsample
   
       def forward(self, x):
           # 保存x
           identity = x
           if self.downsample is not None:
               # shortcut输出
               identity = self.downsample(x)
   
           out = self.conv1(x)
           out = self.bn1(out)
           out = self.relu(out)
   
           out = self.conv2(out)
           out = self.bn2(out)
   
           # 残差连接相加
           out += identity
           out = self.relu(out)
   
           return out
   # 50 101 152
   # 多层的残差结构
   # 卷积核的个数4倍的关系
   class Bottleneck(nn.Module):
       """
       注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
       但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
       这么做的好处是能够在top1上提升大概0.5%的准确率。
       可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
       """
       # 卷积核变为4倍的数量
       expansion = 4
       # 传入的stride默认是1
       # 传入的out_channel是卷积核对应的第一层和第二层卷积核的个数，并不是第三层的卷积核个数
       def __init__(self, in_channel, out_channel, stride=1, downsample=None):
           super(Bottleneck, self).__init__()
   
           self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                  kernel_size=1, stride=1, bias=False)  # squeeze channels
           self.bn1 = nn.BatchNorm2d(out_channel)
           # -----------------------------------------
           # 注意这里stride = stride是传入的
           self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                  kernel_size=3, stride=stride, bias=False, padding=1)
           self.bn2 = nn.BatchNorm2d(out_channel)
           # -----------------------------------------
           self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                                  kernel_size=1, stride=1, bias=False)  # unsqueeze channels
           self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
           self.relu = nn.ReLU(inplace=True)
           self.downsample = downsample
   
       def forward(self, x):
           identity = x
           if self.downsample is not None:
               identity = self.downsample(x)
   
           out = self.conv1(x)
           out = self.bn1(out)
           out = self.relu(out)
   
           out = self.conv2(out)
           out = self.bn2(out)
           out = self.relu(out)
   
           out = self.conv3(out)
           out = self.bn3(out)
   
           out += identity
           out = self.relu(out)
   
           return out
   
   
   class ResNet(nn.Module):
   
       def __init__(self,
                    block,             # 传入的残差结构名字Basicblock 或者是 Bottleneck
                    blocks_num,        # 使用的残差结构的数目，传入的是列表参数
                    num_classes=1000,
                    include_top=True):
           super(ResNet, self).__init__()
           self.include_top = include_top
           self.in_channel = 64
           # 第一层
           self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                                  padding=3, bias=False)
           self.bn1 = nn.BatchNorm2d(self.in_channel)
           self.relu = nn.ReLU(inplace=True)
           self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
           self.layer1 = self._make_layer(block, 64, blocks_num[0])
           self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
           self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
           self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
           if self.include_top:
               self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
               self.fc = nn.Linear(512 * block.expansion, num_classes)
   
           for m in self.modules():
               if isinstance(m, nn.Conv2d):
                   nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
   
       def _make_layer(self, block, channel, block_num, stride=1):
           downsample = None
           if stride != 1 or self.in_channel != channel * block.expansion:
               downsample = nn.Sequential(
                   nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                   nn.BatchNorm2d(channel * block.expansion))
   
           layers = []
           layers.append(block(self.in_channel,
                               channel,
                               downsample=downsample,
                               stride=stride))
           self.in_channel = channel * block.expansion
   
           for _ in range(1, block_num):
               layers.append(block(self.in_channel, channel))
   
           return nn.Sequential(*layers)
   
       def forward(self, x):
           x = self.conv1(x)
           x = self.bn1(x)
           x = self.relu(x)
           x = self.maxpool(x)
   
           x = self.layer1(x)
           x = self.layer2(x)
           x = self.layer3(x)
           x = self.layer4(x)
   
           if self.include_top:
               x = self.avgpool(x)
               x = torch.flatten(x, 1)
               x = self.fc(x)
   
           return x
   
   
   def resnet34(num_classes=1000, include_top=True):
       # https://download.pytorch.org/models/resnet34-333f7ec4.pth
       return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
   
   
   
   def resnet101(num_classes=1000, include_top=True):
       # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
       return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
   
   
   
   ```

2. train.py

   ```python
   import os
   import sys
   import json
   
   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torchvision import transforms, datasets
   from tqdm import tqdm
   
   from restnet_my import resnet34   #改成我自己写的
   
   
   def main():
       device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       print("using {} device.".format(device))
       
       # 参数normal变了
       data_transform = {
           "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
           "val": transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
   
       data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
       image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
       assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
       train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                            transform=data_transform["train"])
       train_num = len(train_dataset)
   
       # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
       flower_list = train_dataset.class_to_idx
       cla_dict = dict((val, key) for key, val in flower_list.items())
       # write dict into json file
       json_str = json.dumps(cla_dict, indent=4)
       with open('class_indices.json', 'w') as json_file:
           json_file.write(json_str)
   
       batch_size = 16
       nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
       print('Using {} dataloader workers every process'.format(nw))
   
       train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)
   
       validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                               transform=data_transform["val"])
       val_num = len(validate_dataset)
       validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                     batch_size=batch_size, shuffle=False,
                                                     num_workers=nw)
   
       print("using {} images for training, {} images for validation.".format(train_num,
                                                                              val_num))
       
       net = resnet34()
       # load pretrain weights
       # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
       model_weight_path = "./resnet34-pre.pth"
       assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
       net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
       # for param in net.parameters():
       #     param.requires_grad = False
   
       # change fc layer structure
       # 加载参数，然后修改最后一层进行迁移学习
       in_channel = net.fc.in_features
       net.fc = nn.Linear(in_channel, 5)
       net.to(device)
   
       # define loss function
       loss_function = nn.CrossEntropyLoss()
   
       # construct an optimizer
       params = [p for p in net.parameters() if p.requires_grad]
       optimizer = optim.Adam(params, lr=0.0001)
   
       epochs = 3
       best_acc = 0.0
       save_path = './resNet34.pth'
       train_steps = len(train_loader)
       for epoch in range(epochs):
           # train
           net.train()
           running_loss = 0.0
           train_bar = tqdm(train_loader, file=sys.stdout)
           for step, data in enumerate(train_bar):
               images, labels = data
               optimizer.zero_grad()
               logits = net(images.to(device))
               loss = loss_function(logits, labels.to(device))
               loss.backward()
               optimizer.step()
   
               # print statistics
               running_loss += loss.item()
   
               train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                        epochs,
                                                                        loss)
   
           # validate
           net.eval()
           acc = 0.0  # accumulate accurate number / epoch
           with torch.no_grad():
               val_bar = tqdm(validate_loader, file=sys.stdout)
               for val_data in val_bar:
                   val_images, val_labels = val_data
                   outputs = net(val_images.to(device))
                   # loss = loss_function(outputs, test_labels)
                   predict_y = torch.max(outputs, dim=1)[1]
                   acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
   
                   val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                              epochs)
   
           val_accurate = acc / val_num
           print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                 (epoch + 1, running_loss / train_steps, val_accurate))
   
           if val_accurate > best_acc:
               best_acc = val_accurate
               torch.save(net.state_dict(), save_path)
   
       print('Finished Training')
   
   
   if __name__ == '__main__':
       main()
   
   ```



### ResNeXt

更新了block

![](./.assets/image-20231212175318862.png)

[分组卷积](###分组卷积)

下面这三种情况是等价的

![image-20231212181525464](./.assets/image-20231212181525464.png)



1. 模型搭建

   ```python
   import torch.nn as nn
   import torch
   
   # ResNeXt网络  组卷积
   # 18层34层残差结构
   # 既要有虚线连接的残差结构
   # 也要有实线连接的残差结构
   class BasicBlock(nn.Module):
       # expansion是一个标志，层次比较resnet有的残差结构卷积核的个数不相同，这里定义了一个倍数，在这里用不到
       expansion = 1
       # 两种残差结构 conv1和downsample stride不同，要传入一个stride
       # dowmsample是虚线shortcut和实线shortcut
       def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
           super(BasicBlock, self).__init__()
           # padding为1可以同时满足stride=1和stride=2的卷积，偏置不取
           self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                  kernel_size=3, stride=stride, padding=1, bias=False)
           # bn只是在每个channel上归一化，没改变channel
           self.bn1 = nn.BatchNorm2d(out_channel)
           self.relu = nn.ReLU()
           # 这个stride不是传入的stride，都是1
           self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                  kernel_size=3, stride=1, padding=1, bias=False)
           self.bn2 = nn.BatchNorm2d(out_channel)
           # shortcut
           self.downsample = downsample
   
       def forward(self, x):
           # 保存x
           identity = x
           if self.downsample is not None:
               # shortcut输出
               identity = self.downsample(x)
   
           out = self.conv1(x)
           out = self.bn1(out)
           out = self.relu(out)
   
           out = self.conv2(out)
           out = self.bn2(out)
   
           # 残差连接相加
           out += identity
           out = self.relu(out)
   
           return out
   
   # 多层的残差结构
   # 卷积核的个数4倍的关系
   class Bottleneck(nn.Module):
       """
       注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
       但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
       这么做的好处是能够在top1上提升大概0.5%的准确率。
       可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
       """
       # 卷积核变为4倍的数量
       expansion = 4
       # 传入的stride默认是1
       # 
       def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                    groups=1, width_per_group=64): 
           super(Bottleneck, self).__init__()
           # 假设输入的特征图channel = 128  
           # in_channel = 128  groups = 32 width_per_group = 4
           # 计算得出width = 128
   
           # 如果groups = 1, width_per_group = 64, 即是普通的卷积，这时候width=128，即每一个残差模块，第一层的卷积核个数
           width = int(out_channel * (width_per_group / 64.)) * groups
   
           self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                                  kernel_size=1, stride=1, bias=False)  # squeeze channels
           self.bn1 = nn.BatchNorm2d(width)
           # -----------------------------------------
           self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                                  kernel_size=3, stride=stride, bias=False, padding=1)
           self.bn2 = nn.BatchNorm2d(width)
           # -----------------------------------------
           # 注意第三个卷积层变成了outchannel * expansion
           self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                                  kernel_size=1, stride=1, bias=False)  # unsqueeze channels
           self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
           self.relu = nn.ReLU(inplace=True)
           self.downsample = downsample
   
       def forward(self, x):
           identity = x
           if self.downsample is not None:
               identity = self.downsample(x)
   
           out = self.conv1(x)
           out = self.bn1(out)
           out = self.relu(out)
   
           out = self.conv2(out)
           out = self.bn2(out)
           out = self.relu(out)
   
           out = self.conv3(out)
           out = self.bn3(out)
   
           out += identity
           out = self.relu(out)
   
           return out
   
   
   class ResNet(nn.Module):
   
       def __init__(self,
                    block,
                    blocks_num,
                    num_classes=1000,
                    include_top=True,
                    groups=1,
                    width_per_group=64):
           super(ResNet, self).__init__()
           self.include_top = include_top
           self.in_channel = 64
   
           self.groups = groups
           self.width_per_group = width_per_group
   
           self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                                  padding=3, bias=False)
           self.bn1 = nn.BatchNorm2d(self.in_channel)
           self.relu = nn.ReLU(inplace=True)
           self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
           self.layer1 = self._make_layer(block, 64, blocks_num[0])
           self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
           self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
           self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
           if self.include_top:
               self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
               self.fc = nn.Linear(512 * block.expansion, num_classes)
   
           for m in self.modules():
               if isinstance(m, nn.Conv2d):
                   nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
   
       def _make_layer(self, block, channel, block_num, stride=1):
           downsample = None
           if stride != 1 or self.in_channel != channel * block.expansion:
               downsample = nn.Sequential(
                   nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                   nn.BatchNorm2d(channel * block.expansion))
   
           layers = []
           layers.append(block(self.in_channel,
                               channel,
                               downsample=downsample,
                               stride=stride,
                               groups=self.groups,
                               width_per_group=self.width_per_group))
           self.in_channel = channel * block.expansion
   
           for _ in range(1, block_num):
               layers.append(block(self.in_channel,
                                   channel,
                                   groups=self.groups,
                                   width_per_group=self.width_per_group))
   
           return nn.Sequential(*layers)
   
       def forward(self, x):
           x = self.conv1(x)
           x = self.bn1(x)
           x = self.relu(x)
           x = self.maxpool(x)
   
           x = self.layer1(x)
           x = self.layer2(x)
           x = self.layer3(x)
           x = self.layer4(x)
   
           if self.include_top:
               x = self.avgpool(x)
               x = torch.flatten(x, 1)
               x = self.fc(x)
   
           return x
   
   
   def resnet34(num_classes=1000, include_top=True):
       # https://download.pytorch.org/models/resnet34-333f7ec4.pth
       return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
   
   
   def resnet50(num_classes=1000, include_top=True):
       # https://download.pytorch.org/models/resnet50-19c8e357.pth
       return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
   
   
   def resnet101(num_classes=1000, include_top=True):
       # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
       return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
   
   
   def resnext50_32x4d(num_classes=1000, include_top=True):
       # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
       groups = 32
       width_per_group = 4
       return ResNet(Bottleneck, [3, 4, 6, 3],
                     num_classes=num_classes,
                     include_top=include_top,
                     groups=groups,
                     width_per_group=width_per_group)
   
   
   def resnext101_32x8d(num_classes=1000, include_top=True):
       # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
       groups = 32
       width_per_group = 8
       return ResNet(Bottleneck, [3, 4, 23, 3],
                     num_classes=num_classes,
                     include_top=include_top,
                     groups=groups,
                     width_per_group=width_per_group)
   
   ```

   

### MobileNetv1、v2

1. 提出DepthWise Convolution（大大减少运算量和参数量）
2. 增加超参数$\alpha卷积核个数比值 、\beta图片的输入分辨率$



**DW卷积**

DW卷积实际上是分组卷积的特殊情况，用单通道的卷积核取卷积一个feature map的一个通道，最终输入通道数和输出通道数相同  [跳转](###DepthWise卷积)

![](./.assets/image-20231212184808985.png)



![](./.assets/image-20231212184828283.png)



**DW与PW的堆叠代替传统的卷积核**

![](./.assets/image-20231212191804092.png)





**Mobile v2**

1. Inverted Residual（倒残差结构）
2. Linear Bottleneck



**倒残差结构**

升维再降维、ReLU6激活函数

![](./.assets/image-20231212192249162.png)



![](./.assets/image-20231212192139850.png)



当**stride = 1且输入的特征矩阵与输出的特征矩阵shape相同时，才会有shortcut连接**

![](./.assets/image-20231212192352329.png)

![](./.assets/image-20231212192817066.png)

t是扩展因子，表示特征图的channel经过第一个卷积层扩展多少倍

c是输出特征矩阵的channel数

n是bottleneck的重复次数

s是步距，**仅为当前bottleneck第一层的dw卷积的stride，其余的stride都是1**

k是分类的类别个数

![image-20231212192926462](./.assets/image-20231212192926462.png)



1. 模型搭建

   ```python
   from torch import nn
   import torch
   
   # 将卷积核个数设置为round_nearest的整数倍 
   def _make_divisible(ch, divisor=8, min_ch=None):
       """
       This function is taken from the original tf repo.
       It ensures that all layers have a channel number that is divisible by 8
       It can be seen here:
       https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
       """
       if min_ch is None:
           min_ch = divisor
       # 将输入通道数，ch调整为离8最近的整数倍（类似于四舍五入）
       new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
       # Make sure that round down does not go down by more than 10%.
       if new_ch < 0.9 * ch:
           new_ch += divisor
       return new_ch
   
   
   class ConvBNReLU(nn.Sequential):
       def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
           padding = (kernel_size - 1) // 2
           super(ConvBNReLU, self).__init__(
               nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
               nn.BatchNorm2d(out_channel),
               nn.ReLU6(inplace=True)
           )
   
   # 倒残差结构
   class InvertedResidual(nn.Module):
       # expand_ratio 扩展因子t
       def __init__(self, in_channel, out_channel, stride, expand_ratio):
           super(InvertedResidual, self).__init__()
           # 第一层卷积核的个数 tk
           hidden_channel = in_channel * expand_ratio
           # 是否有捷径分支,满足条件use_short = true
           self.use_shortcut = stride == 1 and in_channel == out_channel
   
           layers = []
           if expand_ratio != 1: # 如果等于1，就不需要1×1的扩展卷积层了
               # 1x1 pointwise conv
               layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
           # extend能一次性批量传入一堆层
           layers.extend([
               # 3x3 depthwise conv
               # dw卷积
               ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
               # 1x1 pointwise conv(linear)
               nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
               nn.BatchNorm2d(out_channel),
           ])
   
           self.conv = nn.Sequential(*layers)
   
       def forward(self, x):
           if self.use_shortcut:
               return x + self.conv(x)
           else:
               return self.conv(x)
   
   
   class MobileNetV2(nn.Module):
       def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
           super(MobileNetV2, self).__init__()
           block = InvertedResidual
           # 超参数alpha
           input_channel = _make_divisible(32 * alpha, round_nearest)
           last_channel = _make_divisible(1280 * alpha, round_nearest)
   
           inverted_residual_setting = [
               # t, c, n, s
               [1, 16, 1, 1],
               [6, 24, 2, 2],
               [6, 32, 3, 2],
               [6, 64, 4, 2],
               [6, 96, 3, 1],
               [6, 160, 3, 2],
               [6, 320, 1, 1],
           ]
   
           features = []
           # conv1 layer
           features.append(ConvBNReLU(3, input_channel, stride=2))
           # building inverted residual residual blockes
           for t, c, n, s in inverted_residual_setting:
               output_channel = _make_divisible(c * alpha, round_nearest)
               for i in range(n):
                   stride = s if i == 0 else 1
                   features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                   input_channel = output_channel
           # building last several layers
           features.append(ConvBNReLU(input_channel, last_channel, 1))
           # combine feature layers
           self.features = nn.Sequential(*features)
   
           # building classifier
           self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
           self.classifier = nn.Sequential(
               nn.Dropout(0.2),
               nn.Linear(last_channel, num_classes)
           )
   
           # weight initialization
           for m in self.modules():
               if isinstance(m, nn.Conv2d):
                   nn.init.kaiming_normal_(m.weight, mode='fan_out')
                   if m.bias is not None:
                       nn.init.zeros_(m.bias)
               elif isinstance(m, nn.BatchNorm2d):
                   nn.init.ones_(m.weight)
                   nn.init.zeros_(m.bias)
               elif isinstance(m, nn.Linear):
                   nn.init.normal_(m.weight, 0, 0.01)
                   nn.init.zeros_(m.bias)
   
       def forward(self, x):
           x = self.features(x)
           x = self.avgpool(x)
           x = torch.flatten(x, 1)
           x = self.classifier(x)
           return x
   
   ```

2. train.py

   ```python
   import os
   import sys
   import json
   
   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torchvision import transforms, datasets
   from tqdm import tqdm
   
   from model_v2 import MobileNetV2
   
   
   def main():
       device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       print("using {} device.".format(device))
   
       batch_size = 16
       epochs = 5
   
       data_transform = {
           "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
           "val": transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
   
       data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
       image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
       assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
       train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                            transform=data_transform["train"])
       train_num = len(train_dataset)
   
       # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
       flower_list = train_dataset.class_to_idx
       cla_dict = dict((val, key) for key, val in flower_list.items())
       # write dict into json file
       json_str = json.dumps(cla_dict, indent=4)
       with open('class_indices.json', 'w') as json_file:
           json_file.write(json_str)
   
       nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
       print('Using {} dataloader workers every process'.format(nw))
   
       train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)
   
       validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                               transform=data_transform["val"])
       val_num = len(validate_dataset)
       validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                     batch_size=batch_size, shuffle=False,
                                                     num_workers=nw)
   
       print("using {} images for training, {} images for validation.".format(train_num,
                                                                              val_num))
   
       # create model
       net = MobileNetV2(num_classes=5)
   
       # load pretrain weights
       # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
       model_weight_path = "./mobilenet_v2.pth"
       assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
       pre_weights = torch.load(model_weight_path, map_location='cpu')
   
       # delete classifier weights
       # 载入除最后一层权重之外的所有权重
       # 只有当神经网络模型中的参数 k 的元素数量（.numel()）与预训练模型中的参数 v 的元素数量相同时，才将这个键值对包含在 pre_dict 中。
       pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
       # 通过 net.load_state_dict(pre_dict, strict=False) 将 pre_dict 中的权重加载到神经网络模型 net 中
       # missing_keys 是一个列表，包含在加载过程中，模型中存在但是在 pre_weights 中不存在的键。
       # unexpected_keys 是一个列表，包含在加载过程中，pre_weights 中存在但是在模型中不存在的键。
       missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
   
       # freeze features weights
       for param in net.features.parameters():
           param.requires_grad = False
   
       net.to(device)
   
       # define loss function
       loss_function = nn.CrossEntropyLoss()
   
       # construct an optimizer
       params = [p for p in net.parameters() if p.requires_grad]
       optimizer = optim.Adam(params, lr=0.0001)
   
       best_acc = 0.0
       save_path = './MobileNetV2.pth'
       train_steps = len(train_loader)
       for epoch in range(epochs):
           # train
           net.train()
           running_loss = 0.0
           train_bar = tqdm(train_loader, file=sys.stdout)
           for step, data in enumerate(train_bar):
               images, labels = data
               optimizer.zero_grad()
               logits = net(images.to(device))
               loss = loss_function(logits, labels.to(device))
               loss.backward()
               optimizer.step()
   
               # print statistics
               running_loss += loss.item()
   
               train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                        epochs,
                                                                        loss)
   
           # validate
           net.eval()
           acc = 0.0  # accumulate accurate number / epoch
           with torch.no_grad():
               val_bar = tqdm(validate_loader, file=sys.stdout)
               for val_data in val_bar:
                   val_images, val_labels = val_data
                   outputs = net(val_images.to(device))
                   # loss = loss_function(outputs, test_labels)
                   predict_y = torch.max(outputs, dim=1)[1]
                   acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
   
                   val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                              epochs)
           val_accurate = acc / val_num
           print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                 (epoch + 1, running_loss / train_steps, val_accurate))
   
           if val_accurate > best_acc:
               best_acc = val_accurate
               torch.save(net.state_dict(), save_path)
   
       print('Finished Training')
   
   
   if __name__ == '__main__':
       main()
   
   ```

   

### MobileNetv3

1. 加入SE模块（注意力机制）
2. 更新了激活函数

![](./.assets/image-20231212202636250.png)

**SE模块**

对不同的特征图施加一个注意力（权重），让关键的特征图权重更大一些，不太关键的权重小一些

 ![image-20231212203448363](./.assets/image-20231212203448363.png)

**重新设计耗时层的结构**

1. 减少第一个卷积层卷积核的个数

2. 精简了最后一层的结构

   ![image-20231212203809791](./.assets/image-20231212203809791.png)

**重新设计激活函数**

![](./.assets/image-20231212203929790.png)

![](./.assets/image-20231212204107777.png)

**网络结构**

![](./.assets/image-20231212204241364.png)

## 目标检测

### 基础知识

**分类网络必须掌握**

类别：

1. one stage：SSD Yolo
   1. 基于anchors直接进行分类以及边界框调整
2. two stage ： Faster RCNN
   1. 通过专门的模块去生成候选框（RPN）
   2. 基于之前生成的候选框，进一步进行分类和边界框进行调整

### R-CNN 

![](./.assets/image-20231213112614582.png)

深度学习应用于目标检测的开山之作（2014）

步骤：

1. 一张图像生成1K~2K个候选区域（使用Selective Search方法）
2. 对于每个候选区域使用深度网络提取特征
3. 特征送入每一类的SVM分类器，判别是否属于该类别
4. 使用回归器精细修正候选框的位置



![](./.assets/image-20231213115916500.png)



**候选区域的生成**

![](./.assets/image-20231213114044546.png)

利用Selective Search算法通过图像分割的方法得到一些原始区域、然后使用一些合并策略将这些区域合并，得到一个层次化的结构，而这些结构就包含着可能需要的物体

**对于每个候选区域，使用深度网络提特征**

将**2000**候选区域缩放到227*227，丢入AlexNet网络获取4096维特征，得到**2000 * 4096**维矩阵

![](./.assets/image-20231213114118255.png)

**特征送入每一类的SVM分类器（二分类），判定类别**

将2000 * 4096维特征与20个SVM（20个类）组成的权值矩阵 4096 * 20相乘，获得2000 * 20 维矩阵表示每个建议框是某个目标类别的得分。分别对上述2000*20维每一列即每一类进行[**非极大值抑制**](###NMS)去除重叠的建议框，得到该列即该类中得分最高的建议框

![](./.assets/image-20231213114742918.png)

![](./.assets/image-20231213114828364.png)

**使用回归器去修正候选框的位置**

对NMS处理后剩余的建议框进行进一步筛选，接着分别用20个回归器对上述20个类别中剩余的建议框进行回归操作，最终得到每个类别修正后的得分最高的bbx

![](./.assets/image-20231213115737002.png)

黄色框是提议框、绿色窗口为GT框，红色窗口为回归之和的预测框



**问题**

1. 测试速度慢
2. 训练速度慢且繁琐
3. 训练所需的空间大

### Fast R-CNN 

![](./.assets/image-20231213120222004.png)

**步骤：**

1. 一张图像生成1K~2K个候选区域（使用Selective Search方法）
2. 将**整幅图像**输入到网络得到对应的**特征图**，将SS算法生成的候选框**投影到特征图**得到相应的**特征矩阵**
3. 将每个特征通过ROI Pooling层缩放到7*7大小的**特征图**，接着将特征图展平经过一系列的全连接层得到预测结果（Reegion of Interest）

![](./.assets/image-20231213124646914.png)

**一次性计算整张图像特征**

训练过程只采样SS算法选出的部分候选框

**不限制输入图像的尺寸**

![](./.assets/image-20231213120622898.png)

**ROI Pooling**

无论输入图像尺寸多大，**都按7*7拆分**，然后进行下采样，最大池化操作

![](./.assets/image-20231213120910810.png)

**分类器**

输出N + 1个类别的概率,(N为检测目标的种类 1为背景)共N + 1个节点（经过softmax处理）

![](./.assets/image-20231213121141330.png)

**边界框回归器**

输出对应N+1个类别的候选边界框的回归参数($d_x,d_y,d_w,d_h$)，共（N+1）× 4个节点

![](./.assets/image-20231213121328733.png)

如何调整？将预测的候选边界框回归到最终的预测框

$P_x P_yP_wP_h$是初始的anchor参数

$\hat{G}_{x}\hat{G}_{y}\hat{G}_{w}\hat{G}_{h}$是为微调之后的anchor参数

$G_xG_yG_wG_h$是gt anchor参数
$$
\begin{aligned}
&\hat{G}_{x} =P_{w}d_{x}(P)+P_{x}  \\
&\hat{G}_{y} =P_{h}d_{y}(P)+P_{y}  \\
&\hat{G}_{w}=P_{w}\exp(d_{w}(P)) \\
&\hat{G}_{h}=P_{h}\exp(d_{h}(P))
\end{aligned}
$$
$P_x,P_y,P_w,P_h$分别是候选框中心的$x,y$坐标，以及宽和高

$\hat{G}_x,\hat{G}_y,\hat{G}_w,\hat{G}_h$分别是最终预测的目标边界框的中心的$x，y$坐标以及宽和高

**损失函数**

![](./.assets/image-20231213122036679.png)

![](./.assets/image-20231213123907117.png)

u = 0时，取0，即预测背景没有边界框损失

u >=1时，取1，即预测其他类别，是有边界框损失的

`p`：分类器预测的softmax概率分布

`u`：对应目标的真实类别标签

`t^u`：对应边界框回归器预测的对应类别`u`的的回归参数$(t_x^u,t_y^u,t_w^u,t_h^u)$

`v`：对应真实目标边界框的回归参数$v_x,v_y,v_w,v_h$
$$
v_x = \frac{G_x - P_x}{P_x}\ \ \ \
v_y = \frac{G_y- P_y}{P_y}\\
v_w = ln\frac{G_w}{P_w}\ \ \ \ 
v_h = ln\frac{G_h}{P_h}
$$


分类损失是[交叉熵损失](###交叉熵损失)

**实际上交叉熵损失就是真实类别下对应的softmax概率的-log**
$$
loss(x,class) = -log(P_{class})\\
P_{class} =\frac{e^{x[class]}}{\sum_je^{xj}}
$$
边界框回归损失是[smoothL1损失](####Smooth L1 Loss)

**实际上就是把预测的回归参数和真实的回归参数相减，仍到smoothL1函数中，再相加**
$$
\begin{aligned}L_{loc}(t^u,v)&=\sum_{i\in\{x,y,w,h\}}smooth_{L_1}(t_i^u-v_i)\\smooth_{L_1}(x)&=\begin{cases}0.5x^2&\mathrm{if~|x|<1}\\|x|-0.5&\mathrm{otherwise}&\end{cases}\end{aligned}
$$




### Faster R-CNN

RPN  + Fast R-CNN 打通了原来SS网络+特征提取，变为了端对端的实现

![](./.assets/image-20231213124938025.png)

**步骤：**

1. 将图像输入网络中得到相应的**特征图**
2. 使用RPN结构生成候选框，将RPN生成的候选框投影到特征图上获得相应的**特征矩阵**
3. 将每个特征矩阵通过ROI Pooling 缩放到7*7大小的特征图，接着将特征图展平，通过一系列全连接层得到预测结果

![](./.assets/image-20231213141158378.png)

**RPN网络**

对于特征图上每个3*3的滑动窗口，计算出**滑动窗口中心点**，对应**原始图像的中心点**，并计算出**k个anchor boxes**

![](./.assets/image-20231213130441695.png)

**实际上就是用特征图的窗口中心点的坐标*特征图和图长宽的缩放比就得到了原图的窗口的中心点坐标**

一个滑动窗口，按照经验，在Faster R-CNN会在对应原图位置生成3×3=9种anchor

三种尺度（面积）：{$128^2,256^2,512^2$}

三种比例：{$1:1,1:2,2:1$}

![](./.assets/image-20231213130319473.png)

256-d取决于所使用的bacbone最后一层的输出特征图的channel数，ZF网络就是256d，VGG16网络就是512d，在feature上，用一个3*3的卷积核，stride=1，padding=1，channel=256，卷积核个数为256，**得到一个与输入特征图同尺寸的特征矩阵**

在这个特征矩阵上，用**2k**个1*1的卷积核，stride = 1，padding=0，channel = 256，**每一个格子得到2k个得分**

在这个特征矩阵上，用**4k**个1*1的卷积核，stride = 1，padding=0，channel = 256，**每一个格子得到4k个坐标回归参数**

![](./.assets/image-20231213131725440.png)

**每一个点预测2k个得分，4k个回归参数**

cls：`[背景的概率][前景的概率]`

reg：`[dx][dy][dw][dh]`



对于一张`1000*600*3`的图像，大约有 `60*4*9（20k）`个anchor，忽略跨越边界的anchor，剩下约 `6k`个anchor。对于RPN生成的候选框之间存在大量的重叠，基于候选框的cls得分，采用非极大值抑制，IoU设置为0.7，这样每张图片只剩下`2k`个候选框



**训练数据的采样**

采样的正样本和负样本的比例为1：1，如果正样本不够，拿负样本来补齐

**正样本**：anchor与gt IoU超过0.7，如果找不到（都小于0.7），那么找与gt相交的IoU最大的那个anchor也是正样本

**负样本**：anchor与所有gt的IoU小于0.3的anchor



**RPN损失函数**

![](./.assets/image-20231213132750467.png)

$P_i$表示第$i$个anchor预测的真实标签的概率

$P_i^*$当为正样本时为1，当为负样本时为0 类似于艾弗森括号的作用

$t_i$表示第$i$个anchor的边界框回归参数

$t_i^*$表示第$i$个anchor对于的gt box边界框回归参数

$N_{cls}$表示一个mini-batch中所有样本数量为256

$N_{reg}$表示anchor的位置个数(不是anchor的数量)约为2400，实际上就是提取的特征图的长宽乘积



**对于分类的交叉熵损失**

有两种观点

1. 多分类的交叉熵损失

   ![](./.assets/image-20231213133945865.png)

   可以看到每一个anchor，一定有一个类别标签，一共有9个anchor也就有9个类别，1表示前景，0表示背景
   $$
   L_{cls} = -log(p_i)
   $$

   $$
   = -log(0.9) - log(0.2) - ... - log(0.1) - log(0.2)
   $$

2. **二分类的交叉熵损失**

   这时候，实际上预测的不是2k个score二是k个score，因为对于每个anchor我通过sigmod得到一个数，趋近于0是背景，趋近于1是前景

   ![](./.assets/image-20231213134526491.png)
   $$
   L_{cls}=-[p_i^*\log(p_i)+(1-p_i^*)\log(1-p_i)]
   $$

   $$
   = -log(0.9) -log(1 - 0.2) - log(0.1) - log(0.8)
   $$

$P_i$表示第$i$个anchor预测的真实标签的概率

$P_i^*$当为正样本时为1，当为负样本时为0 类似于艾弗森括号的作用

**边界框回归损失**

$P_i^*$当为正样本时为1，当为负样本时为0 类似于艾弗森括号的作用

$t_i$表示第$i$个anchor的边界框回归参数

$t_i^*$表示第$i$个anchor对于的gt box边界框回归参数

**$t_i$是直接预测出来的，$t_i^*$是直接计算出来的**
$$
\begin{aligned}t_x&=(x-x_a)/w_a,t_y=(y-y_a)/h_a,\\t_w&=\log(w/w_a),t_w=\log(h/h_a),\end{aligned}
$$
$x、y、w、h$是最后校准完成后的边界框参数，$x_a、y_a、w_a、h_a$是**初次得到的9个anchor边界框参数**（但是网络实际上是预测的回归参数）
$$
\begin{aligned}t_x^*&=(x^*-x_a)/w_a,t_y^*=(y^*-y_a)/h_a\\t_w^*&=\log(w^*/w_a),t_h^*=\log(h^*/h_a)\end{aligned}
$$
$x^*、y^*、w^*、h^*$是gt边界框参数
$$
t_i = [t_x,t_y,t_w,t_h]\ t_i^*=[t_x^*,t_y^*,t_w^*,t_h^*]\\
$$

$$
\begin{aligned}L_{reg}(t_i,t_i^*)&=\sum_{i}smooth_{L_1}(t_i-t_i^*)\\

smooth_{L_1}(x)&=\begin{cases}0.5x^2&\mathrm{if~|x|<1}\\|x|-0.5&\mathrm{otherwise}&\end{cases}\end{aligned}
$$



**联合训练方法**

**同时训练RPN网络和Fast R-CNN**

### FPN

特征金字塔

![](./.assets/image-20231217191443870.png)

几种常见的结构：

1. 特征图像金字塔结构
2. 传统的前向传播到最后一个特征图进行检测，在FasterR-CNN中用到
3. 取不同尺度的特征图进行预测，在SSD中用到
4. FPN，不同特征图进行特征融合

**特征融合**

![](./.assets/image-20231217191714060.png)

主干网络均是2x下采样，对每一层特征图，先经过一个1×1的卷积核调整到相同的channel256，保证相加时的shape保持一致，另外可以通过2x上采样[最近邻插值](###最近邻插值)，保持高宽一致，这样高层次的特征图和低层次的特征图可以直接相加

![](./.assets/image-20231217192128215.png)

**注意：P6只用于RPN，不在Fast-RCNN中使用，针对不同的预测特征层，RPN和Fast-RCNN的权重共享**

![](./.assets/image-20231217192521304.png)

P2~P6选取不同的anchor，不同的特征层上共用同一个RPN网络，因此共享了RPN网络的参数

**通过RPN得到的一系列Proposal如何分配到预测特征层上**

 ![](./.assets/image-20231217192746061.png)

 `k`：计算得到的预测特征层号2 3 4 5

`k0`：4

`wh`：proposal，在原图像上的宽和高

### RetinaNet



### SSD

![image-20231213232154003](./.assets/image-20231213232154003.png)

一个anchor需要预测c类+4个回归偏移量，一共有k个anchor，所以需要（c+4）*k个卷积核

输入的map 是mn，所以有 c+4 kmn个输出

### yolov1

### yolov2

### yolov3

### yolov5

### DETR

Detection Transformer

 **训练阶段**

![](./.assets/image-20240111174936913.png)

超参数生成100个bbx，利用匈牙利算法找出与gtbox等数量的预测框，利用找出来的框进行损失计算

**测试阶段**

预测出100个框，筛选出置信度最大的框  >0.7

![image-20240111175053644](./.assets/image-20240111175053644.png)

训练框架

object query是动态更新的，并行计算100个object query，**框架比较简单**，端到端实现

![image-20240111175230954](./.assets/image-20240111175230954.png)









![image-20240111175614439](./.assets/image-20240111175614439.png)

850是transformer的token的个数，意思上是在原来一个像素点对应位置所有channel作为数值（每个位置），token

![](./.assets/image-20240111180002369.png)

并且传入一个positinal encodeing ，维度与token的维度是一致的，因为直接相加



![image-20240111180123569](./.assets/image-20240111180123569.png)

 ![image-20240111180738256](./.assets/image-20240111180738256.png)

- `src`:输入的image feature
- `pos`:postional embeding

1. image feature 和 postional embeding进行add操作，然后生成 Q和K
2. 经过self-attention生成src2
3. src2经过dropout和src进行相加，送入norm
4. 送入norm后再送入FNN层
5. 再经过dropout和上一层src相加
6. 最后norm输出src

![image-20240111181151304](./.assets/image-20240111181151304.png)

- tgt是queries （100， 256）
- memory是encoder提取的特征，（850，bs，256）
- pos：（850，bs，256）
- query pos：（100， 256）就是object queries

1. quries和object queries相加得到Q和K，V就等于queries
2. qkv进行selfattention操作得到tgt2
3. tgt2经过drop out再与tgt进行相加，在norm得到tgt
4. object queries + tgt 生成Q （100， 256）
5. memory+pos embeding 生成 K（850, bs, 256）
6. memory是V(850, bs, 256)
7. 进行multi head attention
8. 最后输出tgt
9. train（6， bs， 100， 256） 六个decoder计算结果
10. val（bs， 100， 256）

![image-20240111182321287](./.assets/image-20240111182321287.png)

输出92个类别和91 + 1

bbx中心坐标以及高宽坐标，100个queries



![image-20240111182922679](./.assets/image-20240111182922679.png)

**损失函数**

1. 从100个预测框中，找出和真实标注框所匹配的预测框（匈牙利算法）

   空白处数值无限大，构建一个这样的矩阵

   ![](./.assets/image-20240111183759588.png)

2. 调用函数得到最佳匹配

   100行2列，从100个中筛选出两个

   第1列和14匹配，第0列和33行匹配，最优的匹配结果

   ![](./.assets/image-20240111183908958.png)

   代价矩阵的计算 

![image-20240111184102142](./.assets/image-20240111184102142.png)

![image-20240111184213720](./.assets/image-20240111184213720.png)

类别输出：[2, 100,  92] bs 100个框 92个类别

![](./.assets/image-20240111184408892.png)

取出tgt_ids，第一个targets 82 79

第二个，targets 1 1 34 1

然后，**从200 * 92**的矩阵中取出有意义的部分，即target1含有 82 79 target2含有1 1 34 1

用匈牙利算法进行求解



bbx cost



### 评价指标 



`TP(True Positive)`：**IoU > 0.5**的检测框数量（同一Ground Truth只计算一次）

`FP(False Positive)`: **IoU <= 0.5**的检测框的数量（或者是检测到同一个GT的多余的检测框的数量）

`FN(False Negative)`：没有检测到GT的数量（**漏检**的GT的数目）



`Precision`：
$$
查准率=\frac{TP}{TP+FP}
$$
![](./.assets/image-20231212210103322.png)

此时查准率为1，但是效果并不好，因为有漏检

`Recall`：
$$
查全率 = \frac{TP}{TP + FN}
$$
![](./.assets/image-20231212210042446.png)

此时查全率为1，但是效果并不好，因为查的不准



`AP`：P-R曲线下面积

`P-R曲线`：Precision-Recall曲线

`mAP`：**针对每个类别**求AP，多个类别AP求平均值，即为mAP

![](./.assets/image-20231213105933141.png)

Confidence阈值不断调低，计算P和R，目标边界框都是经过非极大值抑制处理过后的边界框

TP = 1 FP = 0 FN = 6 P = 1.0 R = 0.14

TP = 2 FP = 0 FN = 5 P = 1.0 R = 0.28

TP = 3 FP = 0 FN = 4 P = 1.0 R = 0.42

TP = 4 FP = 0 FN = 3 P = 1.0 R = 0.57

**TP = 4 FP = 1 FN = 3**  P =0.80 R = 0.57

 TP  = 4 FP = 2 FN = 3 P = 0.66 R = 0.57

TP  = 5 FP = 2 FN = 2 P = 0.71 R = 0.71



![](./.assets/image-20231213110726408.png)

ReCall相同时，只保留最大的Precision 

$(0.14-0)\times1.0+(0.28-0.14)\times1.0+(0.42-0.28)\times10+(0.57-0.42)\times1.0+(0.71-0.57)\times0.71=0.6694$

**(当前recall -  上一个recall) × （包含当前行以下的所有Precision的最大值）**



![image-20231213111302671](./.assets/image-20231213111302671.png)

COCO AP：当IOU取0.5 0.55 ~ 0.95 时的mAP，再取均值

AP Across Scales：不同尺寸的mAP

AR ：max值是每张图片最大的预测目标框

AR Across Scales：不同尺寸的mAR

## 大模型

### Agent

自动化AI工具

#### Agent架构解读与应用分析

- Agent是什么？

    - 代理/智能体

    - 理解命令 拆分命令 依靠LLM执行 
- Agent为什么可以这样做？

    - 感知能力 把能用到的信息转化为提示/得到很多记忆
    - 记忆能力 执行的每一步都会记忆 存下来
    - 思考能力 总结感知到的东西
    - 动作 调用API 搜索引擎 询问其他的智能体
- 与LLM的关系

    - LLM是一切
    - 实际的执行者
    - GPT4 API昂贵 落地难 多智能体更贵
- 多智能体

    - 多个角色 每个角色有自己的任务
    - 不同的智能体之间可以进行交互  预先设置好/自由发挥
- 框架

    - AutoGpt/metaGpt
    - key
    - 角色/交互/API调用
- GPTs分析

#### GPTs快速打造专属Agent

#### Agent打造专属客服

#### 微软AutoGenStudio智能体实战

#### MetaGpt框架

#### MetaGpt实战

#### RAG检索框架分析与应用

#### 斯坦福AI小镇架构与项目解读

#### langchain工具

#### MOE多专家系统

#### LLM与LORA微调策略

#### LLM下游任务训练自己的模型

#### GPT-LLM模型优化

# Python语法

## Python基础语法

1. 编码：默认UTF-8编码

2. 标识符

    1. 第一个字符为**字母或下划线**
    2. 标识符由字母数字下划线组成
    3. 大小写敏感

3. 保留字

    ```python
    import keyword
    print(keyword.kwlist)
    ['False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
    ```

4. 注释

    1. 单行注释 `#`

    2. 多行注释

        ```python
        “”“
        多行注释
        ”“”
        
        '''
        多行注释
        '''
        ```

5. 多行语句

    ```python
    total = item_one + \
            item_two + \
            item_three
    ```

    对于`[],{},()`，不需要使用`\`

    ```python
    total = ['item_one', 'item_two', 'item_three',
            'item_four', 'item_five']
    ```

5. 数字类型
    1. int（长整型），python没有long
    2. bool（布尔）
    3. float（浮点） 1.23， 3E-2
    4. complex（复数）1+2j
6. 字符串
    1. `'`和`"`完全相同
    2. `\`用来转义，使用`r`可以防止转义， r"this is a line with \n" ，`\n`会原样输出
    3. `+`连接字符串，`*`运算符重复
    4. 两种索引方式，从左往右`0`开始，从右往左`-1`开始
    5. 字符串的截取：`变量[头下标 : 尾下标 ：步长]`
    6. `"""`和`'''`可以指定多行字符串

7. 空行

    **函数之间或者类的方法之间**用空行分割，表示新的代码开始

8. print输出

    默认换行，如果实现不换行，在变量末尾加上`end=""`

    ```python
    x="a" 
    y="b" 
    # 换行输出 
    print( x ) 
    print( y )  
    print('---------') 
    # 不换行输出 
    print( x, end=" " ) 
    print( y, end=" " ) 
    ```

9. import与from...import

    1. 将整个模块导入

        ```python
        import somemodule
        ```

    2. 从某个模块中导入某个函数

        ```python
        form somemodule import somefunction
        ```

    3. 从某个模块导入多个函数

        ```python
        from somemodule import firstfunc， secondfunc， thirdfunc
        ```

    4. 将某个模块全部函数导入

        ```python
        from somemodule import *
        ```

        

## Python基本数据类型

1. 多个变量赋值

    ```python
    a = b = c = 1
    a, b, c = 1, 2, "runoob"
    ```

2. 标注数据类型

    1. Number

    2. String

    3. bool

    4. List（列表）

    5. Tuple（元组）

    6. Set（集合）

    7. Dictonary（字典）

        不可变数据：Number String Tuple

        可变数据：List Dictonary Set

3. 数值运算

    1. 5 + 4
    2. 4.3 - 2
    3. 3 * 7
    4. 2 / 4 （得到浮点数）
    5. 2 // 4 （得到整数）
    6. 17 % 3 （取余）
    7. 2 ** 5 （乘方）

4. List列表  `[]`

    ```python
    list = [ 'abcd', 786 , 2.23, 'runoob', 70.2 ]
    tinylist = [123, 'runoob']
    
    print (list)            # 输出完整列表
    print (list[0])         # 输出列表第一个元素
    print (list[1:3])       # 从第二个开始输出到第三个元素
    print (list[2:])        # 输出从第三个元素开始的所有元素
    print (tinylist * 2)    # 输出两次列表
    print (list + tinylist) # 连接列表
    
    ['abcd', 786, 2.23, 'runoob', 70.2]
    abcd
    [786, 2.23]
    [2.23, 'runoob', 70.2]
    [123, 'runoob', 123, 'runoob']
    ['abcd', 786, 2.23, 'runoob', 70.2, 123, 'runoob']
    ```



5. Tuple元组 `()`

    **元组和列表一致，但是元组的元素不能修改**

    **可以把字符串看成一种特殊的元组**

    ```python
    #!/usr/bin/python3
    
    tuple = ( 'abcd', 786 , 2.23, 'runoob', 70.2  )
    tinytuple = (123, 'runoob')
    
    print (tuple)             # 输出完整元组
    print (tuple[0])          # 输出元组的第一个元素
    print (tuple[1:3])        # 输出从第二个元素开始到第三个元素
    print (tuple[2:])         # 输出从第三个元素开始的所有元素
    print (tinytuple * 2)     # 输出两次元组
    print (tuple + tinytuple) # 连接元组
    
    ('abcd', 786, 2.23, 'runoob', 70.2)
    abcd
    (786, 2.23)
    (2.23, 'runoob', 70.2)
    (123, 'runoob', 123, 'runoob')
    ('abcd', 786, 2.23, 'runoob', 70.2, 123, 'runoob')
    
    ```

6. Set（集合） `{}`

    Python 中的集合（Set）是一种**无序、可变**的数据类型，用于存储**唯一的元素**。集合中的**元素不会重复**，并且可以进行**交集、并集、差集**等常见的集合操作。

    **注意：**创建一个空集合必须用 **set()** 而不是 **{ }**，因为 **{ }** 是用来创建一个空字典。

    ```python
    #!/usr/bin/python3
    
    sites = {'Google', 'Taobao', 'Runoob', 'Facebook', 'Zhihu', 'Baidu'}
    
    print(sites)   # 输出集合，重复的元素被自动去掉
    
    # 成员测试
    if 'Runoob' in sites :
        print('Runoob 在集合中')
    else :
        print('Runoob 不在集合中')
    
    
    # set可以进行集合运算
    a = set('abracadabra')  
    b = set('alacazam')
    
    print(a)
    
    print(a - b)     # a 和 b 的差集
    
    print(a | b)     # a 和 b 的并集
    
    print(a & b)     # a 和 b 的交集
    
    print(a ^ b)     # a 和 b 中不同时存在的元素
    
    {'Zhihu', 'Baidu', 'Taobao', 'Runoob', 'Google', 'Facebook'}
    Runoob 在集合中
    {'b', 'c', 'a', 'r', 'd'}  已经被拆分了
    {'r', 'b', 'd'}
    {'b', 'c', 'a', 'z', 'm', 'r', 'l', 'd'}
    {'c', 'a'}
    {'z', 'b', 'm', 'r', 'l', 'd'}
    
    ```

    



7. Dictonary（字典）

    列表是有序的对象集合，字典是**无序的对象集合**。两者之间的区别在于：字典当中的元素是**通过键来存取**的，而不是通过偏移存取。

    字典是一种映射类型，字典用 **{ }** 标识，它是一个无序的 **键(key) : 值(value)** 的集合。键(key)必须使用不可变类型。

    在同一个字典中，键(key)必须是唯一的。

    ```python
    #!/usr/bin/python3
    
    dict = {}
    dict['one'] = "1 - 菜鸟教程"
    dict[2]     = "2 - 菜鸟工具"
    
    tinydict = {'name': 'runoob','code':1, 'site': 'www.runoob.com'}
    
    
    print (dict['one'])       # 输出键为 'one' 的值
    print (dict[2])           # 输出键为 2 的值
    print (tinydict)          # 输出完整的字典
    print (tinydict.keys())   # 输出所有键
    print (tinydict.values()) # 输出所有值
    
    
    1 - 菜鸟教程
    2 - 菜鸟工具
    {'name': 'runoob', 'code': 1, 'site': 'www.runoob.com'}
    dict_keys(['name', 'code', 'site'])
    dict_values(['runoob', 1, 'www.runoob.com'])
    
    
    ```

    

# Pytorch 

## 训练代码框架

### 训练代码框架1-AlexNet

![v2-bd2fc9e1651ad59fbe658a582582991c_1440w](./.assets/v2-bd2fc9e1651ad59fbe658a582582991c_1440w.png)

```python
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import matplotlib.pyplot as plt
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ================================= 下载图片、预处理图片、数据加载器 =================================

transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616])
                                ])

train_set = torchvision.datasets.CIFAR10('../input/cifar10', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10('../input/cifar10', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ================================= 查看图片 =================================

# data_iter = iter(train_loader)
# sample_images, sample_labels = data_iter.next()
#
# plt.figure(figsize=(7, 7))
#
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     img = sample_images[i] / 2 + 0.5
#     plt.title(classes[sample_labels[i]])
#     plt.imshow(np.transpose(img, (1, 2, 0)))
# plt.show()


# ================================= 定义模型 =================================

class Alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2),
                                 nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2),
                                 nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2),
                                 nn.Flatten(), nn.Linear(256 * 5 * 5, 4096), nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, 4096), nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, 10))

    def forward(self, X):
        return self.net(X)


# ================================= 训练模型 =================================
model = Alexnet().to(device)

epochs = 20
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

eval_losses = []
eval_acces = []

for epoch in range(epochs):

    if (epoch + 1) % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.1

    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        predict = model(imgs)
        loss = criterion(predict, labels)
        print('epoch {}   loss: {}'.format(epoch, loss))

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    eval_loss = 0
    eval_acc = 0
    model.eval()
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        predict = model(imgs)
        loss = criterion(predict, labels)

        # record loss
        eval_loss += loss.item()

        # record accurate rate
        result = torch.argmax(predict, axis=1)
        acc_num = (result == labels).sum().item()
        acc_rate = acc_num / imgs.shape[0]
        eval_acc += acc_rate

    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))

    print('epoch: {}'.format(epoch))
    print('loss： {}'.format(eval_loss / len(test_loader)))
    print('accurate rate: {}'.format(eval_acc / len(test_loader)))
    print('\n')

plt.title('evaluation loss')
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.show()

```

### 训练代码框架2-Resnet50

![v2-ff2ef8fb41e880e7e7c786faf1442ac5_1440w](./.assets/v2-ff2ef8fb41e880e7e7c786faf1442ac5_1440w-1701684236618-6.png)

```python
import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    epochs = 30
    batch_size = 16
    save_steps = 10
    num_workers = 4
    lr = 0.001
    lr_step = 15

    # =============================== model ===============================

    resnet50 = nn.Sequential(models.resnet50(),
                             nn.ReLU(),
                             nn.Dropout(0.5),
                             nn.Linear(in_features=1000, out_features=5, bias=True))
    resnet50.to(device)

    # =============================== data ===============================

    data_root = "/kaggle/input/flowers/flower_photos"
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_root)

    # =============================== Transform ===============================
    img_size = 224
    train_transform = transforms.Compose([transforms.RandomResizedCrop(img_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                        transforms.CenterCrop(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # =============================== DataSet ===============================

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=train_transform)

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=val_transform)

    # =============================== DataLoader ===============================

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=num_workers)

    print("len(train_loader) = {}".format(len(train_loader)))
    print("len(val_loader) = {}".format(len(val_loader)))

    # ====================== loss_function、optimizer、scheduler =========================
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet50.parameters(), lr=lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=0.1)
    train_and_evaluate(resnet50, train_loader, val_loader, criteria, optimizer, scheduler, epochs, save_steps)


# flower 数据集下载地址：https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
def read_split_data(root, val_rate=0.2):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    print(class_indices)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []

    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)]
        images.sort()
        image_class = class_indices[cla]

        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        # 划分训练集 和 验证集
        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label


class MyDataSet(Dataset):

    def __init__(self, images_path, images_class, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return torch.as_tensor(img), torch.as_tensor(label)


class RunningAverage():

    def __init__(self):
        self.steps = 0
        self.loss_sum = 0
        self.acc_sum = 0

    def update(self, loss, acc):
        self.loss_sum += loss
        self.acc_sum += acc
        self.steps += 1

    def __call__(self):
        return self.loss_sum / float(self.steps), self.acc_sum / float(self.steps)


def save_checkpoint(state):
    save_dir = "pretrained_tm_weights"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    filepath = os.path.join(save_dir, 'best.pth')
    torch.save(state, filepath)
    
    
def train_and_evaluate(model, train_dataloader, val_dataloader, criteria, optimizer, scheduler, epochs, save_steps):
    best_val_acc = 0.0
    for epoch in range(epochs):

        print("Epoch {}/{}".format(epoch + 1, epochs))

        # ---------- train ------------

        model.train()
        metric_avg = RunningAverage()

        for i, (train_batch, labels_batch) in enumerate(train_dataloader):

            train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
            output_batch = model(train_batch)
            loss = criteria(output_batch, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % save_steps == 0:
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                predict_labels = np.argmax(output_batch, axis=1)
                acc = np.sum(predict_labels == labels_batch) / float(labels_batch.size)

                metric_avg.update(loss.item(), acc)

        scheduler.step()
        train_loss, train_acc = metric_avg()
        print("- Train metrics: loss={:.2f}, acc={:.2f}".format(train_loss, train_acc))

        # ---------- validate ------------
        model.eval()
        metric_avg = RunningAverage()

        for val_batch, labels_batch in val_dataloader:
            val_batch, labels_batch = val_batch.to(device), labels_batch.to(device)

            output_batch = model(val_batch)
            loss = criteria(output_batch, labels_batch)

            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            predict_labels = np.argmax(output_batch, axis=1)
            acc = np.sum(predict_labels == labels_batch) / float(labels_batch.size)

            metric_avg.update(loss.item(), acc)

        val_loss, val_acc = metric_avg()
        print("- Validate metrics: loss={:.2f}, acc={:.2f}".format(val_loss, val_acc))

        # ---------- Save weights ------------

        is_best = val_acc >= best_val_acc
        if is_best:
            print("- Found new best accuracy")
            best_val_acc = val_acc

            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optim_dict': optimizer.state_dict()})


if __name__ == '__main__':
    main()
```



## 经典网络框架

### Backbone

#### Resnet

#### Swin transformer

[跳转](####Swin Transformer)

## torch小函数（持续更新）

### torch.max

`torch.max(tensor, dim=a)`

返回第`a`维度下，最大数值以及下标

```python
import torch
a = torch.tensor([[1, 2],[1, 2],[3, 2]])
print(torch.max(a, dim=1))
"""
torch.return_types.max(
values=tensor([2, 2, 3]),
indices=tensor([1, 1, 0]))
"""
```

返回第`a`维度下，下标

```python
import torch
a = torch.tensor([[1,2],[1,2],[3,2]])
print(torch.max(a, dim=1)[1])
"""
tensor([1, 1, 0])
"""
```

### tensor.size

`tensor.size(a)`

返回tensor第`a`维度的长度

```python
import torch
a = torch.tensor([[1,2],[1,2],[3,2]])
b = torch.tensor([1, 2, 1])

print(a.size(0)) # 3
print(a.size(1)) # 2
print(b.size(0)) # 3

```

### torch.flatten

`torch.flatten(x, start_dim=a)`

从第`a`维开始，打平张量`x`

```python
import torch
a = torch.tensor([[[1, 2, 3],[2, 3, 3]],[[2, 2, 3],[2, 3, 2]]])
print(a.shape) # [2, 2, 3]
a = torch.flatten(a, start_dim=1)
print(a)
"""
tensor([[1, 2, 3, 2, 3, 3],
        [2, 2, 3, 2, 3, 2]])
"""
print(a.shape) #[2,  6]
```



## 数据处理

### torchvison.transforms模块

**官方文档**

https://pytorch.org/vision/stable/transforms.html#others

推荐使用v2版本的api

![](./.assets/image.png)

**数据处理和数据增强的方法**，这些模块实际上都是一个**类**，在使用之间需要先**实例化一个对象**，然后调用这个类的方法，进行变换

#### 图像尺寸变换与裁剪

1. transforms.Resize

   https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html?highlight=transforms+resize#torchvision.transforms.Resize

   ```python
   torchvison.transform.Resize(size,
                              interpolation=InterpoLationMode.BILINEAR,
                              max_size=None);
   ```

   

   - 作用：将图像按照指定的插值方式，resize到指定尺寸
   - 参数：
      - `size`：输出图像的尺寸。可以是元组（h, w）可以是单个整数，
         - 如果size是元组，输出大小将匹配h, w的大小
         - 如果size是整数，则图像将**较小的边resize到此数字**，并**保持宽高比**

      - `interpolation`：选用如下插值方法将图像resize到输出尺寸
         - `PIL.Image.NEAREST`最近邻插值
         - `PIL.Image.BILINEAR`双线性插值（默认）
         - `PIL.Image.BICUBIC`双三次插值

      - `max_size`：输出图像较长边的最大值。**仅当size为单个整数时才支持次功能**。如果图像的较长边在根据size缩放后大于max_size，则size将会被覆盖，使较长边等于max_size，此时短边会小于size

举例

PILImage对象size属性返回的是**w, h**，而resize的参数顺序是**h, w**。

   ```python
   from PIL import Image
   import torchvision.transforms as transforms
   import matplotlib.pyplot as plt
   
   original_img = Image.open('image.jpg')  # https://p.ipic.vip/7pvisy.jpg
   print(original_img.size)   # (3280, 1818)
   
   img_1 = transforms.Resize(1500, max_size=None)(original_img)
   print(img_1.size)   # (2706, 1500)
   
   img_2 = transforms.Resize((1500, 1500))(original_img)
   print(img_2.size)   # (1500, 1500)
   
   img_3 = transforms.Resize(1500, max_size=1600)(original_img)
   print(img_3.size)   # (1600, 886)
   
   
   plt.subplot(141)
   plt.axis("off")
   plt.imshow(original_img)
   
   plt.subplot(142)
   plt.axis("off")
   plt.imshow(img_1)
   
   plt.subplot(143)
   plt.axis("off")
   plt.imshow(img_2)
   
   plt.subplot(144)
   plt.axis("off")
   plt.imshow(img_3)
   
   plt.show()
   ```

   ![](./.assets/image-20231204162800579.png)



2. transforms.CenterCrop

   https://pytorch.org/vision/stable/generated/torchvision.transforms.CenterCrop.html?highlight=transforms+centercrop#torchvision.transforms.CenterCrop

   ```python
   torchvision.transforms.CenterCrop(size)
   ```

   - 作用：从**图片中心**裁剪出尺寸为size的图片
   - 参数：
     - `size`：所需裁剪的图片尺寸，即输出图像的尺寸
   - 注意：
     - 若切正方形，`transforms.CenterCrop(100)`和`transforms.CenterCrop((100, 100))`两种写法，效果一样
     - 如果设置的输出尺寸大于原图像尺寸，则会在四周补padding，padding颜色为黑色，像素值为0

   举例

   ```python
   from PIL import Image
   import torchvision.transforms as transforms
   import matplotlib.pyplot as plt
   
   original_img = Image.open('image.jpg')  # https://p.ipic.vip/7pvisy.jpg
   print(original_img.size)   # (3280, 1818)
   
   img_1 = transforms.CenterCrop(1500)(original_img)
   img_2 = transforms.CenterCrop((1500, 1500))(original_img)
   img_3 = transforms.CenterCrop((3000, 3000))(original_img)
   
   plt.subplot(141)
   plt.axis("off")
   plt.imshow(original_img)
   
   plt.subplot(142)
   plt.axis("off")
   plt.imshow(img_1)
   
   plt.subplot(143)
   plt.axis("off")
   plt.imshow(img_2)
   
   plt.subplot(144)
   plt.axis("off")
   plt.imshow(img_3)
   
   plt.show()
   ```

   ![](./.assets/image-20231204163613517.png)

3. transforms.RandomCrop

   https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomCrop.html?highlight=transforms+randomcrop#torchvision.transforms.RandomCrop

   ```python
   torchvision.transforms.RandomCrop(size,
                                    padding=None,
                                    pad_if_need=False,
                                    fill=0,
                                    padding_mode='constant')
   ```

   - 作用：
     - 从图片中随机裁剪出尺寸为size的图片
     - 如果设置了参数`padding`，先添加padding，再从padding后的图像中随机裁剪出大小为size的图片
   - 参数：
     - `size`：所需要裁剪的图片尺寸的大小，即输出图像尺寸
     - `padding`：设置填充大小
       - 当`padding`的值形式为a时，上下左右均匀填充a个像素
       - 当`padding`的值形式为（a, b）时，左右填充a个像素，上下填充b个像素
       - 当`padding`的值形式为（a, b, c, d）时，左上右下分别填充a, b, c, d个像素
     - `pad_if_need`：当原图像的尺寸小于设置的输出图像尺寸（有参数size决定），是否填充，默认为False
     - `padding_mode`：若`pad_if_need`设置为True，此参数起作用，默认值为constant
       - `"constant"`：像素值由参数`fill`指定（默认为黑色，rgb像素值为（0，0，0））
       - `"edge"`：padding的像素值为图像边缘的像素值
       - `"reflect"`：镜像填充，最后一个像素不镜像([1,2,3,4] --> [**3,2**,1,**2,3**,4,**3,2**])，从2 3镜像
       - `"symmetric"`：镜像填充，最后一个像素也镜像([1,2,3,4] --> [**2,1,1,2**,3,4,4,3])，从1 2 ，3 4镜像
     - `fill`：指定像素的填充值，当padding_mode为constant起作用，默认填充为黑色（0，0，0）
   - 注意：
     - 同时指定参数`padding_mode`和参数`fill`时，若`padding_mode`的值不为`"constant"`,则参数`fill`不起作用
     - 若指定的输出图像尺寸size大于输入图像尺寸，并且指定参数`pad_if_need=False`则会报错

   举例

   ```python
   from PIL import Image
   import torchvision.transforms as transforms
   import matplotlib.pyplot as plt
   
   original_img = Image.open('image.jpg')  # https://p.ipic.vip/7pvisy.jpg
   print(original_img.size)   # (3280, 1818)
   
   img_1 = transforms.RandomCrop(1500, padding=500)(original_img)
   img_2 = transforms.RandomCrop(3000, pad_if_needed=True, fill=(255, 0, 0))(original_img)
   img_3 = transforms.RandomCrop(3000, pad_if_needed=True, padding_mode="symmetric")(original_img)
   
   plt.subplot(141)
   plt.axis("off")
   plt.imshow(original_img)
   
   plt.subplot(142)
   plt.axis("off")
   plt.imshow(img_1)
   
   plt.subplot(143)
   plt.axis("off")
   plt.imshow(img_2)
   
   plt.subplot(144)
   plt.axis("off")
   plt.imshow(img_3)
   
   plt.show()
   ```

   ![](./.assets/image-20231204165646986.png)

4. transforms.RandomResizedCrop

   https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomResizedCrop.html?highlight=transforms+randomresizedcrop#torchvision.transforms.RandomResizedCrop

   ```python
   torchvision.transforms.RandomResizedCrop(size, 
                                            scale=(0.08, 1.0), 
                                            ratio=(0.75, 1.3333333333333333), 
                                            interpolation=InterpolationMode.BILINEAR)
   ```

   - 作用：
     - Step1：将图像进行随机剪裁，裁剪出的图像需满足（先裁剪）：
       - 裁剪后图像的**面积**占原图像面积的比例在指定范围内
       - 裁剪后图像的**高宽比**在指定范围内
     - Step2：将Step1得到的图像通过指定的方式进行缩放（resize到所需要的尺寸）
   - 参数：
     - `size`：输出图像的尺寸
       - 如果size是单值，resize到(size*size)，如果是（100，200）则resize到对应尺寸
     - `scale`：随机缩放面积的比例，默认随机选取（0.08,1)之间的一个数
     - `ratio`：随机长宽比，默认随机选取（0.75, 1.3333）之间的一个数。超过这个比例范围会明显失真
     - `interpolation`：选用如下插值方法将图像resize到输出尺寸
       - `PIL.Image.NEAREST`最近邻插值
       - `PIL.Image.BILINEAR`双线性插值（默认）
       - `PIL.Image.BICUBIC`双三次插值
   
   举例
   
   ```python
   from PIL import Image
   import torchvision.transforms as transforms
   import matplotlib.pyplot as plt
   
   original_img = Image.open('image.jpg')  # https://p.ipic.vip/7pvisy.jpg
   print(original_img.size)   # (3280, 1818)
   
   img = transforms.RandomResizedCrop(1500)(original_img)
   
   plt.subplot(121)
   plt.imshow(original_img)
   
   plt.subplot(122)
   plt.imshow(img)
   
   plt.show()
   ```
   
   ![](./.assets/image-20231204170241978.png)

#### 水平翻转与垂直翻转

1. 随机水平翻转

   https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html?highlight=torchvision+transforms+randomhorizontalflip#torchvision.transforms.RandomHorizontalFlip

   ```python
   torchvision.transforms.RandomHorizontalFlip(p=0.5)
   ```

   - 参数：
     - `p`：概率值，默认为0.5。图像会按照指定的概率随机做**水平翻转**

   举例

   ```python
   from PIL import Image
   import torchvision.transforms as transforms
   import matplotlib.pyplot as plt
   
   original_image = Image.open("image.jpg")
   img = transforms.RandomHorizontalFlip(p=0.9)(original_image)
   
   plt.subplot(121)
   plt.imshow(original_image)
   plt.axis("off")
   
   plt.subplot(122)
   plt.imshow(img)
   plt.axis("off")
   
   plt.show()
   ```

   ![](./.assets/image-20231204171420418.png)

2. 随机垂直翻转

   https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomVerticalFlip.html?highlight=randomverticalflip#torchvision.transforms.RandomVerticalFlip

   ```python
   torchvision.transforms.RandomVerticalFlip(p=0.5)
   ```

   参数：

   - `p`：概率值，默认为0.5。图像会按照指定的概率随机做**垂直翻转**

   举例

   ```python
   from PIL import Image
   import torchvision.transforms as transforms
   import matplotlib.pyplot as plt
   
   original_image = Image.open("image.jpg")
   img = transforms.RandomVerticalFlip(p=0.9)(original_image)
   
   plt.subplot(121)
   plt.imshow(original_image)
   plt.axis("off")
   
   plt.subplot(122)
   plt.imshow(img)
   plt.axis("off")
   
   plt.show()
   ```

   ![](./.assets/image-20231204171540862.png)

#### 图像颜色处理

https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html?highlight=colorjitter#torchvision.transforms.ColorJitter

```python
color_jitter = transforms.ColorJitter(brightness=0,
                                     contrast=0,
                                     saturation=0,
                                     hue=0)
```

参数：

- `brightness`：亮度调整系数。调整范围为[1 - brightness,  1 + brightness]，默认值为0
- `contrast`：对比度调整系数。调整范围为[1 - contrast, 1 + contrast]，默认值为0
- `satauration`：饱和度调整系数。调整范围为[1 - saturation ,  1 + saturation]
- `hue`：色调调整系数。调整范围为[- hue, hue]，默认值为0



#### transforms.ToTensor()

https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html?highlight=torchvision+transforms+totensor#torchvision.transforms.ToTensor

```python
torchvision.transforms.ToTensor()
```

`torchvision.transforms.ToTensor`会做三件事：

1. 将数据格式从numpy.ndarray或PIL.Image转换为torch.tensor，数据类型为torch.FloatTensor
2. 把像素值从0~255变换到 0~1之间，处理方式，像素值除以255
3. 将shape由（H，W，C）变换为（C，H， W）

![](./.assets/image-20231204172712106.png)

举例

```python
import numpy as np
import torchvision.transforms as transforms

original_img = np.array([[[123, 245, 32],
                          [64, 21, 235]],

                          [[23, 65, 235],
                           [25, 123, 121]]], dtype=np.uint8)
print("原图像:")
print(original_img.shape)  # H, W, C 
print(original_img.dtype)

print("*"*30)

img_tensor = transforms.ToTensor()(original_img)
print("ToTensor处理后:")
print(img_tensor)
print(img_tensor.shape)   # C, H, W
print(img_tensor.dtype)
```

![](./.assets/image-20231204173054150.png)

#### transfoms.Normalize()

https://pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html?highlight=torchvision+transforms+normalize#torchvision.transforms.Normalize

```python
torchvision.transformees.Normalize(mean, std, inplace=False)
```

- 参数
  - `mean`：（list）每个通道的均值
  - `std`：（list）每个通道的标准差
  - `inpace`：是否在原张量上进行标准化操作
- 作用：按照每个通道，使用对应通道的均值(mean)和标准差（std）对图像进行归一化
- 公式：$x = \frac {x-mean}{std}$ $x$为像素值
- 注意：
  - `transforms.Normalize()`一般都紧跟在`transforms.Totensor()`之后使用
  - `transforms.ToTensor(), transforms.Normalize()`是数据处理的最后2步
  - 经过`transforms.Normalize()`处理后得到的数据，一般是可以直接输入到模型中使用的

举例

```python
import numpy as np
import torchvision.transforms as transforms

original_img = np.array([[[123, 245, 32],
                          [64, 21, 235]],

                          [[23, 65, 235],
                           [25, 123, 121]]], dtype=np.uint8)


img_tensor = transforms.ToTensor()(original_img)
print("ToTensor 处理后 : ")
print(img_tensor)

print("*"*30)

img_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
print("Normalize 处理后 : ")
print(img_normalized)
```

![](./.assets/image-20231204174042225.png)

问：

​			 我们经常会看到代码中使用 `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`， 那么 `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` 这组数据是怎么来的呢？

答：

​	 		是根据 lmageNet 数据集中的图像进行计算得到的。

 ImageNet 是一个超大型数据集，在其上计算得出的均值和标准差，满足绝大部分图像的像素值分布。所以，我们一般都会使用这组 均值和标准差。 如果你自己的数据集很特别，你想计算自己数据集的均值和方差，然后进行使用，也是可以的，就是比较耗时耗资源。





#### transforms.Compose()

https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html?highlight=torchvision+transforms+compose#torchvision.transforms.Compose

**作用：**

`torchvision.transforms.Compose()`的作用是将多个图像转换操作组合在一起，他接受一个transforms**列表**作为参数，该列表包含要组合的转换操作，使用方式

```python
from torchvision.transforms import transforms
from PIL import Image

# 实例化一个对象
my_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
original_img = Image.open("./image.jpg")
img = my_transform(original_img)
```

上面的例子中，让图像依次经过RandomResizedCrop --> RandomHorizontalFlip -->ToTensor --> Normalize处理

**重点：**

我们在生成 `transdoems.Compose()`对象my_transforms之后，直接调用`my_transforms(origimal_img)`，即可处理图像，而一般类的使用，是需要用对象调用方法来实现功能的，比如obj.add(a, b)。而我们可以直接调用`my_transform`对象，不需要调用方法，直接将参数（数据）传给对象，就能实现图像处理功能，是因为在transforms.Compose的内部使用了`__call__`方法

**内部实现：**

```python
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
```



**拓展延伸**

1.  为什么我们能直接调用 模型对象，并传入参数： `model(input)` ，就直接实现 forward 方法中的功能呢 ？为什么不需要调用 forward 方法呢 ？

    在`nn.Module` 类中，实现了__call__方法，然后，在 __call__ 方法中调用的 forward方法

   ```Python
   def __call__(self, *input, **kwargs):
       return self.forward(*input, **kwargs)
   ```

​			所以，我们在生使用 `model(input)` 的时候，调用的是 `nn.Module` 类中的 __call__(input) 方法，然后在 __call__(input) 方法中，又调用的我们**自己写的 forward 方法**



2. `torch.nn.Sequential()` 和 `torchvision.transforms.Compose()`的内部实现 以及使用方式是很类似的，都是将输入的**一连串操作一个一个的迭代出来**，并且**按照顺序进行使用**。

   ```python
   class Sequential(Module):
       def __init__(self, *args):
           super(Sequential, self).__init__()
           if len(args) == 1 and isinstance(args[0], OrderedDict):
               for key, module in args[0].items():
                   self.add_module(key, module)
           else:
               for idx, module in enumerate(args):
                   self.add_module(str(idx), module)
       # 一个个迭代模型，然后按顺序输出出来
       def forward(self, input):
           for module in self:
               input = module(input)
           return input
   ```

### 重写transforms模块

重写 transforms 的目的，接受多个参数，并**对图像 和 标注做同步处理**

为什么要重写，因为在检测任务中，我们对图像平移翻转裁剪，对应的bbx位置也发生了变化，所以需要同时对bbx也进行transform，但是pytorch没有提供对标签transform的api所以需要我们自己重写

- 分类任务

  ![image-20231011183414956](./.assets/y5x7pr.png)

- 目标检测任务

  ![image-20231011201237953](./.assets/0san18.png)

## 数据读取

### Dataset&Dataloader

![image-20231011224256792](./.assets/87oy4n.png)

在Datasets类中定义三个方法：

1. 初始化方法：定义属性，例如图像的地址或标注的地址
2. getitem方法：根据索引取图像，进行处理，返回y
3. len方法：数据集的长度

目的：Dataset对 image和target做预处理，需要满足**图像可以直接输入到模型、标签需要和图像保持一致性（标签也需要resize...之类)**

![image-20231011224436072](./.assets/dt78jw.png)

Dataloader执行过程：

1. 从datasets中迭代出batch_size个索引
2. 从迭代出的索引，调用getitem方法，并返回处理后的数据y



### Dataloader

#### 采样方式Sampler

所有Sanmpler都是继承`torch.utils.data.sampler.Sampler`这个抽象类

1. 顺序采样SequentialSampler

   作用：接收一个Dataset对象，输出数据包中样本量的**顺序索引**

   - 内部代码

     ```python
     class SequentialSampler(Sampler):
         def __init__(self, data_source):
             self.data_source = data_source
         def __iter__(self):
             return iter(range(len(self.data_source)))
         def __len__(self):
             return len(self.data_source)
     ```

     `__init__`：接收参数Dataset对象

     `__iter__`：返回一个可迭代对象（返回的是索引值），因为SequentialSampler是顺序采样，所以返回的索引是顺序数值序列

     `__len__`：返回Dataset中数据的个数

   - 使用举例

     ```python
     import torch.utils.data.sampler as sampler
     
     data = list([17, 22, 3, 41, 8])
     seq_sampler = sampler.SequentialSampler(data_source=data)
     
     for index in seq_sampler:
         print("index: {}, data: {}".format(index, data[index]))
     ```

2. 随机采样RandomSampler

   作用：接收一个Dataset对象，输出数据包中样本量的随机索引（可指定是否重复）

   - 内部代码

     ```python
     class RandomSampler(Sampler):
         def __init__(self, data_source, replacement=False, num_samples=None):
             self.data_source = data_source
             self.replacement = replacement
             self._num_samples = num_samples
             
         def num_samples(self):
             if self._num_samples is None:
                 return len(self.data_source)
             return self._num_samples
         
         def __len__(self):
             return self.num_samples
                    
         def __iter__(self):
             n = len(self.data_source)
             if self.replacement:
                 # 生成的随机数是可能重复的
                 return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
             # 生成的随机数是不重复的
             return iter(torch.randperm(n).tolist())
     ```

     `__init__`：

     ​	`data_source(Dataset)`：采样的Dataset对象

        `replacement(bool)`：如果为True，则抽取的样本是有放回的。默认为False

        `num_samples(int)`：抽取样本的数量，默认是len(dataset)，当replacement为true时，应该被实例化

     `__iter__`：返回一个可迭代对象（返回的是索引），**因为RandmSampler是随机采样，所以返回的索引是随机的数值序列（当replacement=False时，生成的排列是无重复的）**

     `__len__`：返回dataset中的样本量

   - 使用举例

     ```python
     import torch.utils.data.sampler as sampler
     
     data = list([17, 22, 3, 41, 8])
     seq_sampler = sampler.SequentialSampler(data_source=data)
     
     for index in seq_sampler:
         print("index: {}, data: {}".format(index, data[index]))
     ```

3. 批采样BatchSampler

   作用：包装另一个采样器以生成一个小批量索引

   - 内部代码

     ```python
     class BatchSampler(Sampler):
         def __init__(self, sampler, batch_size, drop_last):、
             self.sampler = sampler
             self.batch_size = batch_size
             self.drop_last = drop_last
             
         def __iter__(self):
             batch = []
             for idx in self.sampler:
                 batch.append(idx)
                 # 如果采样个数和batch_size相等则本次采样完成
                 if len(batch) == self.batch_size:
                     yield batch
                     batch = []
             # for 结束后在不需要剔除不足batch_size的采样个数时返回当前batch        
             if len(batch) > 0 and not self.drop_last:
                 yield batch
                 
         def __len__(self):
             # 在不进行剔除时，数据的长度就是采样器索引的长度
             if self.drop_last:
                 return len(self.sampler) // self.batch_size
             else:
                 return (len(self.sampler) + self.batch_size - 1) // self.batch_size
     ```

     参数：

     `sampler`：其他采样器的实例

     `batch_size`：批量大小

     `drop_last`：为True时，如果最后一个batch得到的数据小于batch_size，则丢弃最后一个batch数据

   - 使用举例

     ```python
     import torch.utils.data.sampler as sampler
     data = list([17, 22, 3, 41, 8])
     
     seq_sampler = sampler.SequentialSampler(data_source=data)
     batch_sampler = sampler.BatchSampler(seq_sampler, 2, False )
     
     for index in batch_sampler:
         print(index)
     # [0, 1]
     # [2, 3]
     # [4]
     ```

     



### Dataloader参数

![image-20231011203134620](./.assets/q5a6cd.png)

1. pin_memory

   ![image-20231011204043526](./.assets/1safk1.png)

   **Pinned memory (锁页内存)** ：指将一部分内存**锁定在物理内存**中，防止这部分内存被交换到磁盘上或被其他程序使用，可以提高访问速度和可靠性。避免了数据从主机内存到 GPU 显存的复制过程中的页面交换问题。这种固定在物理内存中的内存称为 Pinned memory。

   **Pageable memory (可分页内存)** ： 是一种将物理内存划分成固定的页面，并在需要时将页面交换到磁盘上以释放内存空间的技术。

   - 其中内存被划分为固定大小的块，称为页（page）
   - 在使用 Pageable memory 传输数据时，数据存储在 虚拟内存 中，然后通过页面文件的方式管理数据的加载和卸载。

   **虚拟内存**：将计算机可用的 **物理内存** 和 **磁盘空间** 组合使用，以提供更大的可用内存空间。它使用一个虚拟地址空间来代替物理地址空间，使得每个进程都认为自己有独立的内存。当进程需要访问一个虚拟地址时，操作系统会将它映射到物理地址空间中的一个位置，这个过程称为地址转换。如果物理内存不足，**操作系统会将物理内存中一些不常用的数据暂存到磁盘上**。

2. collate_fn

   - 分类任务

     ![image-20231011204944012](./.assets/pv9c66.png)

   - 目标检测任务

     ![image-20231011205431500](./.assets/cwwgjd.png)

3. num_works

   **开几个进程执行这件事情，每一个work执行一个batch**

   当 num_workers 参数设置得太高时，可能会导致系统资源不足。

   为避免这种情况，通常建议**将 num_workers 参数设置为等于或小于 CPU 核心数**，以有效平衡数据加载效率和系统资源占用率。

   ```python
   batch_size = 16
   nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])   # number of workers
   train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=nw,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  collate_fn=train_dataset.collate_fn)
   ```


### torch.datasets模块

https://pytorch.org/vision/stable/datasets.html#image-detection-or-segmentation



#### 常用数据集下载

**下载mnist数据集**

 MNIST  全称：mixed national institute of standards and technology database

https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html?highlight=torchvision+datasets+mnist#torchvision.datasets.MNIST

```python
train_dataset = torchvision.datasets.MNIST(root,    
                                           					train=True,               
                                           					transform=transform,    
                                           					download=True)
```

参数：

- `root`：指定数据集下在哪里
- `train`：如果是true，下载训练集train.pt，如果是false，下载测试集test.pt，默认是true
- `transform`：一系列作用在PIL图片上的转换操作，返回一个数据处理后的版本，可以使用之前学过的ttransform模块
- `download`：是否下载到root指定的位置，如果已经下载，将不再下载

 使用举例 ：

- 因为是单通道，所以 `transforms.Normalize` 的均值和标准差 仅指定了一个值
- 记得把数据集的下载地址换掉，换成你想要它下载到的位置

```python
import torch
import torchvision
from torchvision.transforms import transforms
# import matplotlib.pyplot as plt

batch_size = 5

my_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5],   # mean=[0.485, 0.456, 0.406]
                                                        std=[0.5])])  # std=[0.229, 0.224, 0.225]
                                                 
train_dataset = torchvision.datasets.MNIST(root="./",    
                                                  train=True,               
                                                  transform=my_transform,    
                                                  download=True)

val_dataset = torchvision.datasets.MNIST(root="./",    
                                         train=False,               
                                         transform=my_transform,    
                                         download=True) 

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

print(len(train_dataset))  # 60000
print(len(train_loader))   # 12000

iterator = iter(train_loader)  
image, label = next(iterator)
print(image.shape)   # [5, 1, 28, 28] [batch_size, channel, h, w]
print(label)         # [4, 6, 1, 7, 1]

# for i in range(batch_size):
#     plt.subplot(1, batch_size, i+1)
#     plt.title(label[i].item())
#     plt.axis("off")
#     plt.imshow(image[i].permute(1, 2, 0))

# plt.show()
```



## 获取模型

### 模型下载

1. 下载**不带预训练参数**的模型（以resnet50）

   ```python
   from torchvision import models
   
   net = models.resnet50(weights=None)
   # 等价写法
   net = models.resnet50()
   ```

   支持下载的模型https://pytorch.org/vision/stable/models.html

   ![image (1)](./.assets/image (1).png)

2. 下载**带预训练参数**的模型

   ```python
   from torchvision import models
   
   # 加载V1版本权重参数
   model_v1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
   # 等价写法
   model_v1 = models.resnet50(weights="IMAGENET1K_V1")
   
   
   # 加载V2版本权重参数
   model_v2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
   # 等价写法
   model_v2 = models.resnet50(weights="IMAGENET1K_V2")
   
   
   # 加载默认版本权重参数 （一般默认使用最新版本的参数文件）
   model_v2 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
   # 等价写法
   model_v2 = models.resnet50(weights="DEFAULT")
   ```

   参数

   - `weight`：用于指定需要下载的权重版本
   - `progress`：如果为True，会在屏幕上显示下载的模型进度条，默认为true

   可选择的版本https://pytorch.org/vision/stable/models.html



### 新老版本写法差异

[点击跳转](###下载预训练模型torchvision.models)

## 网络搭建

### nn.Sequential、nn.ModuleList、nn.ModuleDict 

1. 简介

   - nn.Sequential、nn.ModuleList、nn.ModuleDict 类都继承自 Module 类

   - nn.Sequential、nn.ModuleList、nn.ModuleDict 语法，类似如下：

     ```python
     net = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
     
     net = nn.ModuleList([nn.Linear(32, 6)4, nn.ReLU()])
     
     net = nn.ModuleDict({'linear': nn.Linear(32, 64), 'act': nn.ReLU()})
     ```

2. 区别

   - `nn.ModuleList` 仅仅是一个储存各种模块的列表，这些模块之间**没有联系也没有顺序**（所以不用保证相邻层的输入输出维度匹配），而且**没有实现 forward 功能**需要自己实现
   - 和`nn.ModuleList` 一样， `nn.ModuleDict` 实例仅仅是存放了一些模块的字典，并**没有定义 forward 函数**需要自己定义
   - 而 `nn.Sequential` 内的模块**需要按照顺序排列**，要保证**相邻层的输入输出大小相匹配**；`nn.sequential` 内部 forward 功能已经实现，直接调用的，不需要再写 forward

   ```python
   import torch
   import torch.nn as nn
   
   net1 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
   net2 = nn.ModuleList([nn.Linear(32, 64), nn.ReLU()])
   net3 = nn.ModuleDict({'linear': nn.Linear(32, 64), 'act': nn.ReLU()})
   
   # print(net1)
   # print(net2)
   # print(net3)
   
   x = torch.randn(8, 3, 32)
   print(net1(x).shape)
   # print(net2(x).shape)  # 会报错，提示缺少forward
   # print(net3(x).shape)   # 会报错，提示缺少forward
   ```

   为 nn.ModuleList 写 forward 函数

   ```python
   import torch
   import torch.nn as nn
   
   
   class My_Model(nn.Module):
       def __init__(self):
           super(My_Model, self).__init__()
           self.layers = nn.ModuleList([nn.Linear(32, 64),nn.ReLU()])
   
       def forward(self, x):
           for layer in self.layers:
               x = layer(x)
           return x
   
   net = My_Model()
   
   x = torch.randn(8, 3, 32)
   out = net(x)
   print(out.shape)
   ```

   为 nn.ModuleDict 写 forward 函数

   ```python
   import torch
   import torch.nn as nn
   
   
   class My_Model(nn.Module):
       def __init__(self):
           super(My_Model, self).__init__()
           self.layers = nn.ModuleDict({'linear': nn.Linear(32, 64), 'act': nn.ReLU()})
   
       def forward(self, x): 
           # 取出每一层进行迭代
           for layer in self.layers.values():
               x = layer(x)
           return x
   
   net = My_Model()
   x = torch.randn(8, 3, 32)
   out = net(x)
   print(out.shape)
   ```

   将 nn.ModuleList 转换成 nn.Sequential

   ```python
   import torch
   import torch.nn as nn
   
   module_list = nn.ModuleList([nn.Linear(32, 64), nn.ReLU()])
   net = nn.Sequential(*module_list)
   x = torch.randn(8, 3, 32)
   print(net(x).shape)
   ```

   将 nn.ModuleDict 转换成 nn.Sequential

   ```python
   import torch
   import torch.nn as nn
   
   module_dict = nn.ModuleDict({'linear': nn.Linear(32, 64), 'act': nn.ReLU()})
   net = nn.Sequential(*module_dict.values())
   x = torch.randn(8, 3, 32)
   print(net(x).shape)
   ```

   

3. nn.ModuleDict和nn.ModuleList的区别

   - ModuleDict 可以给每个层**定义名字**，ModuleList 不会

   - ModuleList 可以**通过索引**读取，并且**使用 append 添加元素**

     ```python
     import torch.nn as nn
     
     net = nn.ModuleList([nn.Linear(32, 64), nn.ReLU()])
     net.append(nn.Linear(64, 10))
     print(net)
     ```

   - ModuleDict 可以通过 key 读取，并且可以像 字典一样添加元素

     ```python
     import torch.nn as nn
     
     net = nn.ModuleDict({'linear1': nn.Linear(32, 64), 'act': nn.ReLU()})
     net['linear2'] = nn.Linear(64, 128)
     print(net)
     ```

4. nn.ModuleDict和nn.ModuleList与Python list dict的区别

   加入到 ModuleList 、ModuleDict 里面的所有模块的参数会**被自动添加到整个网络中**。

   ```python
   import torch.nn as nn
   
   net = nn.ModuleDict({'linear': nn.Linear(32, 64), 'act': nn.ReLU()})
   
   for name, param in net.named_parameters():
       print(name, param.size()) 
   ```

   

   ```python
   import torch.nn as nn
   
   net = nn.ModuleList([nn.Linear(32, 64), nn.ReLU()])
   
   for name, param in net.named_parameters():
       print(name, param.size())
   ```

​		

### register_buffer、register_parameter

1. register_parameter

   register_parameter() 是 torch.nn.Module 类中的一个方法

   - 作用

     - 用于定义**可学习**的参数
     - 定义的参数可以保存到网络对象的参数中，可以用`net.parameters()` 或 `net.named_parameters()` 查看
     - 定义的参数可用`net.state_dict()` 转换到字典中，进而 保存到网络文件 / 网络参数文件中

   - 函数说明

     ```python
     register_parameter(name,param)
     ```

     - `name`：参数名称

     - `param`：参数张量，必须是`torch.nn.Parameter()` 对象 或 None ，否则报错如下

   - 使用举例
     
     ```python
     import torch
     import torch.nn as nn
     
     
     class MyModule(nn.Module):
         def __init__(self):
             super(MyModule, self).__init__()
             self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False)
             self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=1, bias=False)
     
             self.register_parameter('weight', torch.nn.Parameter(torch.ones(10, 10)))
             self.register_parameter('bias', torch.nn.Parameter(torch.zeros(10)))
     
     
         def forward(self, x):
             x = self.conv1(x)
             x = self.conv2(x)
             x = x * self.weight + self.bias
             return x
     
     
     net = MyModule()
     
     for name, param in net.named_parameters():
         print(name, param.shape)
     
     print('\n', '*'*40, '\n')
     
     for key, val in net.state_dict().items():
         print(key, val.shape) 
     ```
     
     
     

2. register_buffer

   register_buffer()是 torch.nn.Module() 类中的一个方法

   - 作用

     - 用于定义**不可学习**的参数
     - 定义的参数**不会被保存到网络对象的参数**中，使用 `net.parameters()` 或 `net.named_parameters()` 查看不到
     - 定义的参数可用 `net.state_dict()` 转换到字典中，进而 保存到网络文件 / 网络参数文件中

     `register_buffer()` 用于在网络实例中 注册缓冲区，存储在缓冲区中的数据，**类似于参数**（但不是参数），它与参数的区别为

     - 参数：可以被优化器更新 （requires_grad=False / True）
     - buffer 中的数据 ： **不会被优化器更新**

   - 函数说明

     ```python
     register_buffer(name，tensor)
     ```

     - `name`：参数名称
     - `tensor`：张量

   - 使用举例

     ```python
     import torch
     import torch.nn as nn
     
     
     class MyModule(nn.Module):
         def __init__(self):
             super(MyModule, self).__init__()
             self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False)
             self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=1, bias=False)
     
             self.register_buffer('weight', torch.ones(10, 10))
             self.register_buffer('bias', torch.zeros(10))
     
     
         def forward(self, x):
             x = self.conv1(x)
             x = self.conv2(x)
             x = x * self.weight + self.bias
             return x
     
     
     net = MyModule()
     
     for name, param in net.named_parameters():
         print(name, param.shape)
     
     print('\n', '*'*40, '\n')
     
     for key, val in net.state_dict().items():
         print(key, val.shape)
     ```




### BatchNormlization

batchnormlization实际上就是在一个batch中对每一个channel进行归一化操作

然后按照类似于动量法来更新最后的归一化后的值（可学习的参数）

1. 作用

   ![image-20231009130131007](./.assets/7hste3.png)

2. 计算方式

   ![image-20231009150829589](./.assets/pn83n9.png)

3. 查看BN层中参数的数量

   BN层中有 **weight** 和 **bias** 两个参数（也就是上面计算方式 中的 gamma 和 beta）

   如下代码中，BN层输入的 channel数为 16，每个 channel 有 2个可学习参数，一共有32个可学习参数

   ![](./.assets/ad5n8a.png)

4. 验证阶段&测试阶段的 Batch Normalization

![image-20231009151646182](./.assets/rj3j5n.png)

![image-20231009152959148](./.assets/xcvreh.png)

```python
import numpy as np
import torch.nn as nn
import torch


def batch_norm(feature, statistic_mean, statistic_var):
    feature_shape = feature.shape
    # 针对每一个channel
    for i in range(feature_shape[1]):
        channel = feature[:, i, :, :]
        mean = channel.mean()   # 均值
        std_1 = channel.std()   # 总体标准差
        std_t2 = channel.std(ddof=1)  # 样本标准差
        # 对channel中的数据进行归一化
        feature[:, i, :, :] = (channel - mean) / np.sqrt(std_1 ** 2 + 1e-5)
        # 更新统计均值 和 方差(动量法)
        statistic_mean[i] = statistic_mean[i] * 0.9 + mean * 0.1
        statistic_var[i] = statistic_var[i] * 0.9 + (std_t2 ** 2) * 0.1

    print(feature)
    print('statistic_mean : ', statistic_mean)
    print('statistic_var : ', statistic_var)


# [bs, ch, h, w]
feature_array = np.random.randn(2, 2, 2, 2)
feature_tensor = torch.tensor(feature_array.copy(), dtype=torch.float32)

# 初始化统计均值和方差
statistic_mean = [0.0, 0.0]
statistic_var = [1.0, 1.0]

# 手动计算 batch normalization 结果，打印统计均值和方差
batch_norm(feature_array, statistic_mean, statistic_var)

# 调用 torch.nn.BatchNorm2d
bn = nn.BatchNorm2d(2, eps=1e-5)
output = bn(feature_tensor)

print(output)
print('bn.running_mean : ', bn.running_mean)
print('bn.running_var : ', bn.running_var)

```

5. BN层使用节点

   ![image-20231009154817274](./.assets/0ihuwm.png)

   这种顺序之所以能够效果良好，是因为批量归一化能够使得输入分布均匀，而 ReLU 又能够将分布中的负值清除，从而达到更好的效果。

6. BN层对卷积层的影响

![image-20231009154914345](./.assets/wpbu3x.png)

### BatchNorm LayerNorm GroupNorm

BatchNorm、LayerNorm 和 GroupNorm 都是深度学习中常用的归一化方式。

它们通过将输入归一化到均值为 0 和方差为 1 的分布中，来**防止梯度消失和爆炸**，并**提高模型的泛化能力**。

**LayerNorm**

**Transformer block** 中会使用到 LayerNorm ， 一般输入尺寸形为 ：（batch_size, token_num, dim），会**在最后一个维度做 归一化**： nn.LayerNorm(dim)

![img](./.assets/mr3de4.jpg)

对比手动计算的 LN层输出结果 和 调用 `nn.LayerNorm()` 的输出结果

```python
import torch
import torch.nn as nn
import numpy as np

feature_array = np.array([[[[1, 0],  [0, 2]],
                           [[3, 4],  [1, 2]],
                           [[2, 3],  [4, 2]]],

                          [[[1, 2],  [-1, 0]],
                            [[1, 2], [3, 5]],
                            [[1, 4], [1, 5]]]], dtype=np.float32)


feature_array = feature_array.reshape((2, 3, -1)).transpose(0, 2, 1) # [2, 4, 3]
feature_tensor = torch.tensor(feature_array.copy(), dtype=torch.float32)

ln_out = nn.LayerNorm(normalized_shape=3)(feature_tensor)
print(ln_out)

b, token_num, dim = feature_array.shape
feature_array = feature_array.reshape((-1, dim))
for i in range(b*token_num):
    mean = feature_array[i, :].mean()
    var = feature_array[i, :].var()
    print(mean)
    print(var)

    feature_array[i, :] = (feature_array[i, :] - mean) / np.sqrt(var + 1e-5)
print(feature_array.reshape(b, token_num, dim))
```

**Group Norm**

![](./.assets/kua15n.jpg)

batch size 过大或过小都不适合使用 BN，而是使用 GN。

（1）当 batch size 过大时，BN 会将所有数据归一化到近乎相同的均值和方差。这可能会导致模型在训练时变得非常不稳定，并且很难收敛。

（2）当 batch size 过小时，BN 可能无法有效地学习数据的统计信息

![img](./.assets/4k68tf.jpg)

比如，Deformable DETR 中，就用到了 GroupNorm

![img](./.assets/9m1h76.jpg)

```python
import torch
import torch.nn as nn
import numpy as np

feature_array = np.array([[[[1, 0],  [0, 2]],
                           [[3, 4],  [1, 2]],
                           [[-2, 9], [7, 5]],
                           [[2, 3],  [4, 2]]],

                          [[[1, 2],  [-1, 0]],
                            [[1, 2], [3, 5]],
                            [[4, 7], [-6, 4]],
                            [[1, 4], [1, 5]]]], dtype=np.float32)

feature_tensor = torch.tensor(feature_array.copy(), dtype=torch.float32)
gn_out = nn.GroupNorm(num_groups=2, num_channels=4)(feature_tensor)
print(gn_out)

feature_array = feature_array.reshape((2, 2, 2, 2, 2)).reshape((4, 2, 2, 2))

for i in range(feature_array.shape[0]):
    channel = feature_array[i, :, :, :]
    mean = feature_array[i, :, :, :].mean()
    var = feature_array[i, :, :, :].var()
    print(mean)
    print(var)

    feature_array[i, :, :, :] = (feature_array[i, :, :, :] - mean) / np.sqrt(var + 1e-5)
feature_array = feature_array.reshape((2, 2, 2, 2, 2)).reshape((2, 4, 2, 2))
print(feature_array)
```

## 迁移学习

迁移学习的优势：

1. 能够快速训练出一个理想的结果
2. 当数据集较小时也能训练出理想的效果

**但是使用别人预训练模型的参数时、要注意别人的数据预处理方式**

为什么迁移学习有用？

在迁移学习中，模型初期已经训练好了一部分参数，这部分参数是浅层的特征，可以被重复利用

![image-20231212200959259](./.assets/image-20231212200959259.png)

**常见的迁移学习方式**

1. 载入权重后训练所有的参数
2. 载入权重后只训练最后几层参数
3. 载入权重后在原网络基础上再添加一层全连接层、仅训练最后一个全连接层

```python
import os
import torch
import torch.nn as nn
from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # option1
    # 第一种方式，先加载原来的模型 并导入模型权重
    # 再修改fc层为自己需要的类别数
    net = resnet34()
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)

    # option2
    # 第二种方式，在模型中修改好我的最后的fc层对应的类别数
    # 将权重放在一个字典中，进行剔除操作
    # 找到fc层
    net = resnet34(num_classes=5)
    pre_weights = torch.load(model_weight_path, map_location=device)
    del_key = []
    for key, _ in pre_weights.items():
        if "fc" in key:
            del_key.append(key)
    # 如果fc层在删除列表中，则删除这一层的权重
    for key in del_key:
        del pre_weights[key]
    
    missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    print("[missing_keys]:", *missing_keys, sep="\n")
    print("[unexpected_keys]:", *unexpected_keys, sep="\n")
    
    
    # option3
    net = resnet34(num_classes=5)
    pre_weights = torch.load(model_weight_path, map_location=device)
    # delete classifier weights
    # 载入除最后一层权重之外的所有权重
    # 只有当神经网络模型中的参数 k 的元素数量（.numel()）与预训练模型中的参数 v 的元素数量相同时，才将这个键值对		包含在 pre_dict 中。
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    # 通过 net.load_state_dict(pre_dict, strict=False) 将 pre_dict 中的权重加载到神经网络模型 net 中
    # missing_keys 是一个列表，包含在加载过程中，模型中存在但是在 pre_weights 中不存在的键。
    # unexpected_keys 是一个列表，包含在加载过程中，pre_weights 中存在但是在模型中不存在的键。
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    


if __name__ == '__main__':
    main()

```



### 查看网络的层

#### torchsummary.summary

1. 函数介绍

   作用：通过指定**输入尺寸**，来查看网络中每一层的输出尺寸和每一层的参数数量，**只需要指定input_size**即可，**不需要实际的输入数据**

   ```python
   torchsummary.summary(model, input_size=(3, 224, 224))
   ```

   参数：

   - `model` ：要查看的网络

   - `input_size` ：指定网络输入尺寸，可以指定四个维度:（B, C, H, W），也可以只指定三个维度:（C, H, W）

     如果仅指定三个维度的尺寸（不指定batch_size），那么每一层的输出尺寸 `output shape` 都显示为 -1

2. 使用举例

   ```python
   from torchsummary import summary
   import torchvision.models as models
   
   model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
   print(summary(model, (3, 224, 224)))
   ```

   输出

   ```
   ----------------------------------------------------------------
           Layer (type)               Output Shape         Param #
   ================================================================
               Conv2d-1           [-1, 64, 55, 55]          23,296
                 ReLU-2           [-1, 64, 55, 55]               0
            MaxPool2d-3           [-1, 64, 27, 27]               0
               Conv2d-4          [-1, 192, 27, 27]         307,392
                 ReLU-5          [-1, 192, 27, 27]               0
            MaxPool2d-6          [-1, 192, 13, 13]               0
               Conv2d-7          [-1, 384, 13, 13]         663,936
                 ReLU-8          [-1, 384, 13, 13]               0
               Conv2d-9          [-1, 256, 13, 13]         884,992
                ReLU-10          [-1, 256, 13, 13]               0
              Conv2d-11          [-1, 256, 13, 13]         590,080
                ReLU-12          [-1, 256, 13, 13]               0
           MaxPool2d-13            [-1, 256, 6, 6]               0
   AdaptiveAvgPool2d-14            [-1, 256, 6, 6]               0
             Dropout-15                 [-1, 9216]               0
              Linear-16                 [-1, 4096]      37,752,832
                ReLU-17                 [-1, 4096]               0
             Dropout-18                 [-1, 4096]               0
              Linear-19                 [-1, 4096]      16,781,312
                ReLU-20                 [-1, 4096]               0
              Linear-21                 [-1, 1000]       4,097,000
   ================================================================
   Total params: 61,100,840
   Trainable params: 61,100,840
   Non-trainable params: 0
   ----------------------------------------------------------------
   Input size (MB): 0.57
   Forward/backward pass size (MB): 8.38
   Params size (MB): 233.08
   Estimated Total Size (MB): 242.03
   ----------------------------------------------------------------
   None
   ```

   

#### netron

1. **在线版**

   浏览器访问：https://netron.app/ 点击 “Open Model” 按钮，选择要可视化的模型文件即可

2. **离线版**

   终端进行安装： `pip install netron`

   安装完成后，在脚本中 调用包 `import netron`

   运行程序 `netron.start("model.onnx")`， 会自动打开浏览器进行可视化 （最后有例子）

**3. 支持的网络模型**

![](./.assets/qph2mn.png)

我习惯用 pytorch，但是 netron 对 pytorch 的 `.pt` 和 `.pth` 文件不是很友好，所以，我都是先转换为 onnx 格式，再进行可视化，下面举例。

另外，netron 可以直接可视化 yolo （DarkNet 框架）的 .cfg文件，非常方便

4. 举例

   一般情况下，netron 只展示最初的输入尺寸 和 最后的输出尺寸，中间层的尺寸都是不展示的（如下）。

   ![](./.assets/vrjo84.png)

   可以通过 `onnx.save(onnx.shape_inference.infer_shapes(onnx.load("model.onnx")), "model.onnx")` 进行处理，

   这样中间的每一层的输入输出就都会推理出 并可视化出来了。

   ```python
   import torch
   import torch.nn as nn
   import netron
   import onnx
   from onnx import shape_inference
   
   
   class My_Net(nn.Module):
       def __init__(self):
           super(My_Net, self).__init__()
           self.layer1 = nn.Sequential(
               nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
               nn.BatchNorm2d(16),
               nn.LeakyReLU(),
           )
   
           self.layer2 = nn.Sequential(
               nn.Conv2d(16, 32, kernel_size=1, bias=False),
               nn.BatchNorm2d(32),
               nn.LeakyReLU(),
           )
   
       def forward(self, x):
           x = self.layer1(x)
           x = self.layer2(x)
           return x
   
   
   net = My_Net()
   # 输入图像
   img = torch.rand((1, 3, 224, 224))
   # 转为onnx格式
   torch.onnx.export(model=net, args=img, f='model.onnx', input_names=['image'], output_names=['feature_map'])
   # 保存中间层输出
   onnx.save(onnx.shape_inference.infer_shapes(onnx.load("model.onnx")), "model.onnx")
   # 网络要是搭建不对会报错
   netron.start("model.onnx")
   ```

   ![](./.assets/izi7zp.png)

#### TensorBoardX

![](./.assets/image-20231205194618897.png)

使用 `TensorBoardX` 查看网络结构，分为三个步骤：

 1）实例化一个 `SummaryWriter` 对象

​			实例化 `SummaryWriter` 对象的时候，有三种指定参数的方式

```python
from tensorboardX import SummaryWriter

# 提供一个路径，将使用该路径来保存日志
writer1 = SummaryWriter(log_dir='./runs')

# 无参数，默认使用 runs/日期时间 路径来保存日志，比如：'runs/Aug20-17-20-33'
writer2 = SummaryWriter()

# 提供一个 comment 参数，将使用 runs/日期时间-comment 路径来保存日志，比如： 'runs/Aug20-17-20-33-resnet'
writer3 = SummaryWriter(comment='_resnet')
```

 2）让 `SummaryWriter ` 对象调用 `add_graph` 方法来获取模型结构

参数

- `model` : 待可视化的网络模型
- `input_to_model` : 要输入网络的数据（一个tensor）

```python
add_graph(model, input_to_model=None, verbose=False, **kwargs)
```

 3）打开浏览器查看 模型的可视化起结构

在终端 cd 到 logs目录所在的同级目录，输入如下命令

```bash
tensorboard --logdir ./logs --port 6006
```

注意：./logs 是日志路径，**路径不要加双引号**

在浏览器窗口输入地址：http://localhost:6006/ ， 查看模型网络结构

### 查看网络的参数

#### net.parameters()

`net.parameters()` 用于查看网络中的参数

```python
import torch

# 搭建网络
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 3),
        )
        self.layer2 = torch.nn.Linear(3, 6)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(6, 7),
            torch.nn.Linear(7, 5),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
      
# 构造网络对象
net = MyModel()

for param in net.parameters():
    print(param.shape)
```

输出

```python
torch.Size([4, 3])
torch.Size([4])
torch.Size([3, 4])
torch.Size([3])
torch.Size([6, 3])
torch.Size([6])
torch.Size([7, 6])
torch.Size([7])
torch.Size([5, 7])
torch.Size([5])
```

#### net.named_parameters()

`net.parameters()` 用于查看网络中的参数名和 参数

```python
import torch

# 搭建网络
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 3),
        )
        self.layer2 = torch.nn.Linear(3, 6)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(6, 7),
            torch.nn.Linear(7, 5),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
      
# 构造网络对象
net = MyModel()

for name, param in net.named_parameters():
    print(name, param.shape)
```

输出

```python
layer1.0.weight torch.Size([4, 3])
layer1.0.bias torch.Size([4])
layer1.1.weight torch.Size([3, 4])
layer1.1.bias torch.Size([3])
layer2.weight torch.Size([6, 3])
layer2.bias torch.Size([6])
layer3.0.weight torch.Size([7, 6])
layer3.0.bias torch.Size([7])
layer3.1.weight torch.Size([5, 7])
layer3.1.bias torch.Size([5])
```



#### net.state_dict()

`net.state_dict()` 会将网路中的 参数名 和参数 转换为 字典的格式

（我们经常用它 将参数转换为字典的格式，然后再保存起来）

```python
import torch

# 搭建网络
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 3),
        )
        self.layer2 = torch.nn.Linear(3, 6)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(6, 7),
            torch.nn.Linear(7, 5),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
      
# 构造网络对象
net = MyModel()

for key, value in net.state_dict().items():
    print(key, value.shape)
```

输出

```python
layer1.0.weight torch.Size([4, 3])
layer1.0.bias torch.Size([4])
layer1.1.weight torch.Size([3, 4])
layer1.1.bias torch.Size([3])
layer2.weight torch.Size([6, 3])
layer2.bias torch.Size([6])
layer3.0.weight torch.Size([7, 6])
layer3.0.bias torch.Size([7])
layer3.1.weight torch.Size([5, 7])
layer3.1.bias torch.Size([5])
```



### 增删改网络的层

#### 删除网络的层

1. 删除整个classifier层

   `del alexnet.classifier`

   ```python
   import torch.nn as nn
   import torchvision.models as models
   
   alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
   
   del alexnet.classifier
   print(alexnet)
   ```

   输出

   ```
   AlexNet(
     (features): Sequential(
       (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
       (1): ReLU(inplace=True)
       (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
       (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
       (4): ReLU(inplace=True)
       (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
       (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (7): ReLU(inplace=True)
       (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (9): ReLU(inplace=True)
       (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (11): ReLU(inplace=True)
       (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
     )
     (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
   )
   ```

   

2. 删除整个classifier第6层

   del alexnet.classifier[6]`

   ```python
   import torch.nn as nn
   import torchvision.models as models
   
   alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
   
   del alexnet.classifier[6]
   print(alexnet)
   ```

   输出

   ```python
   AlexNet(
     (features): Sequential(
       (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
       (1): ReLU(inplace=True)
       (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
       (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
       (4): ReLU(inplace=True)
       (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
       (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (7): ReLU(inplace=True)
       (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (9): ReLU(inplace=True)
       (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (11): ReLU(inplace=True)
       (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
     )
     (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
     (classifier): Sequential(
       (0): Dropout(p=0.5, inplace=False)
       (1): Linear(in_features=9216, out_features=4096, bias=True)
       (2): ReLU(inplace=True)
       (3): Dropout(p=0.5, inplace=False)
       (4): Linear(in_features=4096, out_features=4096, bias=True)
       (5): ReLU(inplace=True)
     )
   )
   ```

   

3. 删除classifier层的最后两层

   `alexnet.classifier = alexnet.classifier[:-2]`

   ```python
   import torch.nn as nn
   import torchvision.models as models
   
   alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
   # 切片
   alexnet.classifier = alexnet.classifier[:-2]
   print(alexnet)
   ```

#### 修改网络的层

1. 修改classifier层第6层

   将 classifier 层中的第6层由 `Linear(in_features=4096, out_features=1000, bias=True)` 修改为 `Linear(in_features=4096, out_features=1024, bias=True)`

   ```python
   import torch.nn as nn
   import torchvision.models as models
   
   alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
   
   alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=1024)
   print(alexnet)
   ```

#### 增加网络的层

1. 添加单层

   ```
   import torch.nn as nn
   import torchvision.models as models
   
   alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
   
   alexnet.classifier.add_module('7', nn.ReLU(inplace=True))
   alexnet.classifier.add_module('8', nn.Linear(in_features=1000, out_features=20))
   print(alexnet)
   ```

2. 一次添加多层

   通过 `nn.Sequential` 构造网络片段，一次性往网络中添加多个层

   ```
   import torch.nn as nn
   import torchvision.models as models
   
   alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
   
   block = nn.Sequential(nn.ReLU(inplace=True),
                         nn.Linear(in_features=1000, out_features=20))
   alexnet.add_module('block', block)
   print(alexnet)
   ```

   



### 模型/参数保存与加载

1. 加载模型 + 参数

   `'resnet50.pth'` 为模型+预训练参数文件（.pth 文件中既有模型又有参数）

   ```python
   import torch
   import torchvision.models as models
   
   # 加载模型+参数
   net = torch.load("resnet50.pth")
   ```

2. 只加载参数

   `'resnet50_weight.pth'` 为预训练参数文件（.pth 文件中只有参数）

   ```python
   import torch
   import torchvision.models as models
   
   
   # 加载不带参数的 resnet50 网络
   resnet50 = models.resnet50(weights=None)  
   
   # 往 resnet50 网络中加载参数
   resnet50.load_state_dict(torch.load('resnet50_weight.pth'))  # 使用参数 map_location=device 指定下载到cpu还是gpu
   ```

3. 保存模型 + 参数

   ```python
   import torch
   import torchvision.models as models
   
   #  下载 resnet50 的网络和参数
   resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
   
   # 保存模型 + 参数 到 resnet50.pth 文件中
   torch.save(resnet50, 'resnet50.pth')
   ```

4. 只保存参数

   ```python
   import torch
   import torchvision.models as models
   
   # 下载 resnet50 的网络和参数
   resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
   
   # 仅保存模型的参数 到 resnet50_weight.pth 文件中
   torch.save(resnet50.state_dict(), 'resnet50_weight.pth')
   ```

   说明：`resnet50.state_dict()` 是将resnet50网络中的参数 读取出来，并转换为字典

### 下载预训练模型torchvision.models

`Pytorch` 提供了许多任务的 模型 和 预训练好的参数，可直接通过 `torchvision.models()` 进行下载。

比如像分类任务，pytorch 就提供了如下模型以及预训练好的参数，更多请查看官网：https://pytorch.org/vision/stable/models.html

![image-20231011082923311](./.assets/n0k4rw.png)

1. 使用方法

   调用方式类似如下，如果需要**加载带预训练参数的网络**，类似 `ResNet50_Weights.IMAGENET1K_V1` 的参数名称，可从官网进行查看（如上，右图左列）

   ```python
   from torchvision import models
   
   # 加载resnet50网络（不带参数）
   models.resnet50(weights=None)
   models.resnet50()
   
   # 加载带预训练参数的resnet50网络
   models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
   models.resnet50(weights="IMAGENET1K_V1")
   ```

   权重参数 `weights=model.ResNet50_Weights.IMAGENET1K_V1` 中， `IMAGENET1K_V1` 表示的是 ResNet-50 在 ImageNet 数据集上进行预训练的第一个版本的权重参数文件。是一个版本标识符。

   如果你**不知道哪个权重文件的版本是最新的**，没关系，直接**选择默认DEFAULT**即可。

   官方会随着 torchvision 的升级而让 DEFAULT 权重文件版本保持在最，如下所示：

   ```python
   from torchvision import models
   
   model_new = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
   ```

   

2. 旧版本报错

   如果PyTorch 和 torchvision 更新到了较新的版本，却使用如下旧版本的写法，则会报错

   ```python
   from torchvision import models
   
   # 加载resnet50网络（不带参数）
   models.resnet50(pretrained=False)  # 旧版本写法，已弃用
   
   # 加载带预训练参数的resnet50网络
   models.resnet50(pretrained=True)  # 旧版本写法，已弃用
   ```

   

3. why新版写法

   使用新版本写法 `weights=预训练模型参数版本` ，相当于我们掌握了预训练权重参数文件的选择权

   ```python
   from torchvision import models
   
   # 加载精度为76.130%的旧权重参数文件V1
   model_v1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
   # 等价写法
   model_v1 = models.resnet50(weights="IMAGENET1K_V1")
   
   # 加载精度为80.858%的新权重参数文件V2
   model_v2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
   # 等价写法
   model_v1 = models.resnet50(weights="IMAGENET1K_V2")
   ```

   

### 参数冻结

将resnet50前段参数加载到yolov1的backbone

![image-20231205202806972](./.assets/image-20231205202806972.png)

```python
from torchvision import models
import torch
from torchsummary import summary
from resnet import resnet50


# ------------------------------------------------------------
#  任务一 ：
#  1、将模型A 作为backbone，修改为 模型B
#  2、模型A的预训练参数 加载到 模型B上
# ------------------------------------------------------------

resnet_modified = resnet50()
new_weights_dict = resnet_modified.state_dict()

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
weights_dict = resnet.state_dict()

for k in weights_dict.keys():
    if k in new_weights_dict.keys() and not k.startswith('fc'):
        new_weights_dict[k] = weights_dict[k]
resnet_modified.load_state_dict(new_weights_dict)
# resnet_modified.load_state_dict(new_weights_dict, strict=False)


# --------------------------------------------------
#  任务二：
#  冻结与训练好的参数
# --------------------------------------------------
params = []
train_layer = ['layer5', 'conv_end', 'bn_end']
for name, param in resnet_modified.named_parameters():
    if any(name.startswith(prefix) for prefix in train_layer):
        print(name)
        params.append(param)
    else:
        param.requires_grad = False

optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=5e-4)


```

**迁移学习中模型参数加载的方式**

```

```



## Transformer

### 注意力机制

**Attention 机制听起来高大上，其关键就是学出一个权重分布，然后作用在特征上**。

#### Self Attention

1. qkv的理解

   相似度 = query * key

   总分 = 相似度 * value

2. 计算过程

   ![](./.assets/image-20231206164225809.png)

   self attention中的self，表示query key value 都来自自己，每个token都能提取出自己的query key value

   ![](./.assets/image-20231206164354287.png)

   **计算公式：**
   $$
   \mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
   $$
   为什么要除以dk？QK的数值可能会比较大，交叉熵比较小，不方便计算

   ![image-20231206164534879](./.assets/image-20231206164534879.png)

   ![](./.assets/image-20231206164726615.png)

   **self attention得到的b1、b2、b3、b4**可以看作是每个token的feature

   **这里的W1 W2 W3权重是共享的，因此可以进行并行计算**

   ![](./.assets/image-20231206164856370.png)

   

   

3. 代码实现

   ![](./.assets/扫描件_X=(142)_1.jpg)

   ```python
   import torch.nn as nn
   import torch
   # import matplotlib.pyplot as plt
   
   
   class Self_Attention(nn.Module):
       def __init__(self, dim, dk, dv):
           super(Self_Attention, self).__init__()
           self.scale = dk ** -0.5
           self.q = nn.Linear(dim, dk)
           self.k = nn.Linear(dim, dk)
           self.v = nn.Linear(dim, dv)
   
   
       def forward(self, x):
           q = self.q(x) # [1, 4, 2]
           print(q)
           k = self.k(x) # [1, 4, 2]
           print(k)
           v = self.v(x) # [1, 4, 3]
           print(v)
   
   		# 计算相似度
           attn = (q @ k.transpose(-2, -1)) * self.scale #[1, 4, 4]
           print(attn)
           #print(attn.shape)
           attn = attn.softmax(dim=-1)
           print(attn)
   		
           # 计算最终得分向量
           x = attn @ v
           #print(x.shape)
           print(x)
           return x
   
   
   att = Self_Attention(dim=2, dk=2, dv=3)
   x = torch.rand((1, 4, 2))
   output = att(x)
   
   ```

   

4. 位置编码

   self  attention没有位置信息，交换token，输出的feature是一样的，因此加入位置编码

   ![](./.assets/image-20231206165200888.png)

   - 通过公式生成位置编码
     $$
     \begin{aligned}PE_{(pos,2i)}&=sin(pos/10000^{2i/d_{\mathrm{model}}})\\\\PE_{(pos,2i+1)}&=cos(pos/10000^{2i/d_{\mathrm{model}}})\end{aligned}
     $$
     ![](./.assets/image-20231206165241174.png)

     直接对token的feature 加上 pe1，这个a1 和 pe1维度是一样的，数值进行相加

     位置编码的生成，第几个token第几个位置，对应不同的编码

     ![](./.assets/image-20231206165336678.png)

   - 生成可学习的位置编码

#### Multi-Head Attention

![image-20231206171352709](./.assets/image-20231206171352709.png)

实际计算中，先计算出dmodel长度的qkv，然后对其拆分成h份，每一份是dmodel / h 的长度

在每个head中,进行qkv计算，得到每个token的feature

最后将每个token在多个head中的feature concat起来

![](./.assets/image-20231206171636403.png)

![image-20231206171744201](./.assets/image-20231206171744201.png)

最后连一个全连接层，将直接concat起来的feature再平滑一下

**代码实现**

```python
from math import sqrt
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, dim_in, d_model, num_heads=3):
        super(MultiHeadSelfAttention, self).__init__()

        self.dim_in = dim_in
        self.d_model = d_model
        self.num_heads = num_heads

        # 维度必须能被num_head 整除
        assert d_model % num_heads == 0, "d_model must be multiple of num_heads"

        # 定义线性变换矩阵
        self.linear_q = nn.Linear(dim_in, d_model)
        self.linear_k = nn.Linear(dim_in, d_model)
        self.linear_v = nn.Linear(dim_in, d_model)
        self.scale = 1 / sqrt(d_model // num_heads)

        # 最后的线性层
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.d_model // nh  # dim_k of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)

        dist = torch.matmul(q, k.transpose(2, 3)) * self.scale  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.d_model)  # batch, n, dim_v

        # 最后通过一个线性层进行变换
        output = self.fc(att)

        return output


x = torch.rand((1, 4, 2))
multi_head_att = MultiHeadSelfAttention(x.shape[2], 6, 3)  # (6, 3)
output = multi_head_att(x)

```



#### Cross Attention

### Vision Transformer（ViT）

将图像转换成一系列的token，将图像分割开，每一个作为一个token，显然比较吃计算

因此选择将图像分割成小的patchs，将每一个patchs enbedding成一个token

![](./.assets/image-20231206172342744.png)

![](./.assets/image-20231206172427580.png)

1. patch embeding

   ![image-20231206172814623](./.assets/image-20231206172814623.png)

![image-20231206173448490](./.assets/image-20231206173448490.png)



2. class embedding and positional embedding

   **位置编码和class编码**都是可学习的编码

   ![image-20231206173616674](./.assets/image-20231206173616674.png)

   

### Swin Transformer

### DETR 目标检测

### Deformable DETR





## 损失函数

核心思想：对于损失函数，我们认为，损失函数越接近于0，预测的越精确

因此我们定义损失函数的值域时，要保证：

精确程度      精确       不精确

Loss                0             比较大

### 交叉熵损失

假设类别预测的结果为如下表：

![](./.assets/image-20231205165414259.png)

交叉熵损失函数公式：$loss(x,class)=-log(\frac{e^{x[class]}}{\sum_je^{xj}})=-x_{[class]}+log(\sum_je^{x_j})$

1. softmax函数    $softmax(x_i)=\frac{e^{x_i}}{\sum_je^{x_j}}$

   - 经过$e^{x}$运算，将$x$转换为非负数，因为概率为非负
   - softmax函数得出的结果是样本被预测到每个类别的概率，所有类别的概率相加和为1

   ![](./.assets/u62.png)

2. 取出真实分类的那个概率，我们希望它的值是100%

   class = 2   $\frac{e^{x_{[class]}}}{\sum_je^{x_j}}=0.1538$

3. 作为损失函数，后面需要求导比较麻烦，所以最好想办法转换为加减形式，最理想的办法是取对数

   $log\frac{e^{x_{[class]}}}{\sum_je^{x_j}}=loge^{x_{[class]}}-log\sum_je^{x_j}=x_{[class]}-log\sum_je^{x_j}$

   由于对数单调递增，求$\frac{e^{x_{[class]}}}{\sum_je^{x_j}}$最大值的问题，变为求$\log\frac{e^{x[class]}}{\sum_je^{x_j}}$最大值的问题。$\frac{e^{x_{[class]}}}{\sum_je^{x_j}}$取值范围是（0，1），最大值是1，取对数之后$\log\frac{e^{x[class]}}{\sum_je^{x_j}}$取值范围是$\begin{bmatrix}-\infty,0\end{bmatrix}$，最大值是0

4. 作为损失函数的意义是：当**预测结果越接近真实值**，**损失函数的值越接近0**

   所以我们把$\log\frac{e^{x[class]}}{\sum_je^{x_j}}$取反

   $loss(x,class)=-log(\frac{e^{x[class]}}{\sum_je^{xj}})=-x_{[class]}+log(\sum_je^{x_j})$

   **实际上交叉熵损失就是真实类别下对应的softmax概率的-log**
   $$
   loss(x,class) = -log(P_{class})\\
   P_{class} =\frac{e^{x[class]}}{\sum_je^{xj}}
   $$
   
   
   

### 目标检测的损失

目标检测任务的损失函数由Classification和Bounding Box Regression Loss两部分构成

近几年目标检测bbx loss演进路线是

Smooth L1 Loss $\rarr$ IoU Loss $\rarr$ GIoU Loss $\rarr$ CIoU Loss

#### L1 Loss、L2 Loss

![](./.assets/u24.png)

x表示模型的预测值，y表示真实值，z = x - y表示预测值和真实值之间的差异

L1 Loss
$$
L_{L1}\left(x,y\right)=\left|x-y\right|=\left|z\right|
$$
L2 Loss
$$
L_{L2}\left(x,y\right)=0.5(x-y)^2=0.5z^2
$$
L1、L2损失函数对z的导数分别为：
$$
\frac{\partial L_{L1}\left(x,y\right)}{\partial z}=\begin{cases}1,&\mathrm{~if~}z\geq0\\-1,&\mathrm{~otherwise}&\end{cases}
$$

- L1 损失函数对z的导数为常数，在训练后期，z很小时，如果learning rate不变，损失函数会在稳定值附近波动，很难收敛到更高的精度

$$
\frac{\partial L_{L2}\left(x,y\right)}{\partial z}=z
$$

- L2 损失函数对z的导数为z，在训练初期，z值很大，使得导数也很大，在训练初期不稳定



基于以上，我们想平衡一下，二者，出现了Smooth L1 Loss



#### Smooth L1 Loss

![](./.assets/u8.png)

Smooth L1 Loss最初是在Fast R-CNN论文中提出
$$
L_{smoothL1}\left(x,y\right)=\begin{cases}0.5(x-y)^2=0.5z^2,&\mathrm{~if~}|x-y|<1\\|x-y|-0.5=|z|-0.5,&\mathrm{~otherwise}&\end{cases}
$$
梯度：
$$
\frac{\partial L_{smoothL1}\left(x,y\right)}{\partial z}=\begin{cases}z,&\mathrm{~if~}|x-y|<1\\\pm1,&\mathrm{~otherwise}&\end{cases}
$$


- 当训练初期，z比较大的时候，取下面的，导数为正负1，可用保证较快的收敛
- 当训练后期，z比较小的时候，取上面的，导数为z，可以保证微调

**用于计算bbx坐标的误差**

实际目标检测回归任务中的损失loss为：
$$
L_{loc(t,v)}=\sum_{i\in(x,y,w,h)}smooth_{L1}\left(t_i-v_i\right)
$$
分别求 x y w h的loss，然后相加

缺点：

上面三种Loss用于计算bbx的loss，独立求出x y w h的loss，然后进行相加得到bbx的loss，这种做法假设x y w h是相互独立的，实际上，x y w h是有相关性的

这就引出了IoU Loss

#### IoU Loss

![](./.assets/image-20231206130455335.png)

![](./.assets/image-20231206130506264.png)

以上情况，L1 Loss 和 L2 Loss分别相同，但是 IoU不相同，明显有的预测的不太好，但是 L1 Loss L2 Loss没法克服这种情况，除此之外，我们在训练的时候用Smooth Loss 在评估的时候却用的 IoU指标，所以，为了避免**模型学习优化和模型评估**目标不一致，我们最好也用IoU来计算损失，并且IoU Loss是将**x y w h这四个值构成的bbx**看作一个整体来进行损失计算的



![](./.assets/image-20231206130948866.png)

IoU的取值范围为（0， 1），并且当IoU为1时，我们认为是精确的，IoU为0时，是不精确的，因此
$$
\textit{IoU loss}=-ln^{IoU}
$$
IoU Loss取值变为了（0，+∞），并且IoU=1,loss=0

**考虑极端情况**

当预测框和目标框不相交时，$IoU(A,B)=0$，不能反映A B的远近程度，此时的损失函数不可导，IoU loss无法优化两个框不相交的情况，当然也无法评估出不相交情况下，预测框和真实框相差多远，预测效果到什么程度
$$
Loss=-ln^{IoU},\quad\frac{\partial loss}{\partial IoU}=(-ln^{IoU})^{\prime}=-\frac1{IoU}
$$
![](./.assets/image-20231206131551701.png)



#### GIoU Loss

![](./.assets/image-20231206131801347.png)

外接矩形框C：能正好把A和B都框起来的一个**最小的外接矩形**
$$
GIoU=IoU-\frac{|C-(A\cup B)|}{|C|}
$$
其中ABC分别表示面积

**考虑两种极端情况**

![image-20231206132000152](./.assets/image-20231206132000152.png)
$$
GIoU~Loss=1-GIoU=1-IoU+\frac{|C-(A\cup B)|}{|C|}
$$
预测准的情况下，GIoU Loss = 0

预测不准的情况下，GIoU Loss = 2



**GIoU的缺点**

![](./.assets/image-20231206132223234.png)

这种情况下，GIoU相同，模型没法区分两个框的位置关系时怎么样的，从而没法快速收敛





#### DIoU Loss

![](./.assets/image-20231206132329410.png)
$$
DIoU=IoU-\frac{d^2}{c^2}
$$
d：预测框与真实框中心点的欧式距离

c：外界矩形对角线的长度


$$
DIoULoss=1-DIoU=1-IoU+\frac{d^2}{c^2}
$$
**考虑两种极端情况**

1. 当预测非常准的时候，IoU=1，DIoU = 1 Loss = 0
2. 当预测非常不准的时候，IoU  = 0，DIoU = -1， Loss = 2

**通过最小化 DIoU 损失函数，可以直接拉进预测框 和 真实框之间的距离，加快收敛速度**



#### CIoU Loss

CIoU在DIou的基础上，多考虑了**bbx的长宽比**
$$
CIoU=DIoU-\alpha v=\textit{Io}U-\frac{d^2}{c^2}-\alpha v
$$

- $\alpha$是用来做trade-off的参数：$\alpha=\frac v{(1-IoU)+v}$
  - 当IoU越大，$\alpha$越大，则表示优先考虑高宽比
  - 当IoU越小，$\alpha$越小，则表示优先考虑距离比
- $v$是用来衡量长宽比一致性的参数：$v=\frac4{\pi^2}(arctan\frac{w^{gt}}{h^{gt}}-arctan\frac wh)^2$
  - 当长宽比一致时，v = 0
  - 当长宽比差别比较大时，v = 1

$$
CIoU~Loss=1-CIoU=~1-IoU+\frac{d^2}{c^2}+\alpha v
$$



## 优化器与学习率

### 优化器

![image-20231012231744691](./.assets/8zbxe9.png)

超参数：根据经验确定的变量

#### 梯度下降法

传统权重更新算法为最常见、最简单的一种参数更新策略。

**（1）基本思想 ：**先设定一个学习率 $\eta$，参数沿梯度的**反方向移动**。假设需要更新的参数为$w$，梯度为$g$，则其更新策略可表示为：$w\leftarrow w-\eta*g$

**（2）梯度下降法有三种不同的形式：**

- BGD（Batch Gradient Descent）：批量梯度下降，每次参数更新使用 **所有样本**
- SGD（Stochastic Gradient Descent）：随机梯度下降，每次参数更新只使用 **1个样本**
- MBGD（Mini-Batch Gradient Descent）：小批量梯度下降，每次参数更新使用 **小部分数据样本**（mini_batch）

这三个优化算法在训练的时候虽然所采用的的数据量不同，但是他们在进行参数优化的时候，采用的方法是相同的：

 step 1 ：$g=\frac{\partial loss}{\partial w}$

 step 2 ：求梯度的**平均值**

 step 3 ： 更新权重 ：$w\leftarrow w-\eta*g$

**（3）优缺点**

优点：

- 算法简洁，当学习率取值恰当时，可以收敛到 全局最优点(凸函数) 或 局部最优点(非凸函数)。

缺点：

- 对超参数学习率比较敏感：过小导致收敛速度过慢，过大又越过极值点

- 学习率除了敏感，有时还会因其在迭代过程中保持不变，很容易造成算法被**卡在鞍点**的位置。

- **在较平坦的区域**，由于梯度接近于0，优化算法会因误判，在还未到达极值点时，就提前结束迭代，陷入局部极小值。

  ![](./.assets/g57zuh.png)

  在训练的时候一般都是使用小批量梯度下降算法，即选择部分数据进行训练，这里把这三种算法统称为传统梯度下降法 。而更优的优化算法从**梯度方面** 和 **学习率方面**对参数更新方式进行优化

##### 一维梯度下降法

我们以 目标函数（损失函数）$f(x)=x^2$ 为例来看一看梯度下降是如何工作的（这里 $x$为参数）

迭代方法为：
$$
x\leftarrow x-\eta*g=x-\eta*\frac{\partial loss}{\partial x}
$$
虽然我们知道最小化$f(x)=x^2$ 的解为 $x=0$，这里依然使用如下代码来观察$x$ 是如何迭代的

这里$x$为模型参数，使用 $x=10$ 作为初始值，并设 学习率$\eta = 0.5$，使用梯度下降法 对$x$迭代10次

```python
import numpy as np
import matplotlib.pyplot as plt

x = 10
lr = 0.2
result = [x]

for i in range(10):
    x -= lr * 2 * x
    result.append(x)
    
f_line = np.arange(-10, 10, 0.1)
plt.plot(f_line, [x * x for x in f_line])
plt.plot(result, [x * x for x in result], '-o')
plt.title('learning rate = {}'.format(lr))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
```

![](https://p.ipic.vip/c4sk08.png)

不同的学习率

- 如果使用的学习率太小，将导致$x$的更新非常缓慢，需要更多的迭代。
- 相反，当使用过大的学习率，$x$的迭代不能保证降低$f(x)=x^2$的值，例如，当学习率为$\eta = 1.1$，$x$超出了最优解$x=0$，并逐渐发散



![在这里插入图片描述](./.assets/k2ipej.png)

##### 多维梯度下降法

在对一元梯度下降有了了解之后，下面看看多元梯度下降，即考虑 $X=[x_1,x_2,\cdots x_d]^T$ 的情况。

多元损失函数，它的梯度也是多元的，是一个由d个偏导数组成的向量:
$$
\nabla f(X)=[\frac{\partial f_x}{\partial x_1},\frac{\partial f_x}{\partial x_2},\cdots,\frac{\partial f_x}{\partial x_d}]^T
$$
然后选择合适的学率进行梯度下降：
$$
x_i\leftarrow x_i-\eta*\nabla f(X)
$$
下面通过代码可视化它的参数更新过程。构造一个目标函数$f(X)=x_1^2+2x_2^2$，并有二维向量作$X=[x_1,x_2]$为输入，标量作为输出。

损失函数的梯度为 $\nabla f(x)=[2x_1,4x_2]^T$ 。使用梯度下降法，观察从$x_1,x_2$初始位置[-5, -2] 的更新轨迹。

```python
import numpy as np
import matplotlib.pyplot as plt


def loss_func(x1, x2):  # 定义目标函数
    return x1 ** 2 + 2 * x2 ** 2


x1, x2 = -5, -2
eta = 0.4
num_epochs = 20
result = [(x1, x2)]

for epoch in range(num_epochs):
    gd1 = 2 * x1
    gd2 = 4 * x2

    x1 -= eta * gd1
    x2 -= eta * gd2

    result.append((x1, x2))

# print('x1:', result1)
# print('\n x2:', result2)

plt.figure(figsize=(8, 4))
plt.plot(*zip(*result), '-o', color='#ff7f0e')
x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
plt.contour(x1, x2, loss_func(x1, x2), colors='#1f77b4')
plt.title('learning rate = {}'.format(eta))
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
```

![](./.assets/1b3b9i.png)



#### 带动量的梯度下降法

![](https://p.ipic.vip/5w2e5v.gif)

思想：让参数的更新具有**惯性**， 每一步更新 都是由前面梯度的累积$v$ 和当前点梯度 $g$组合而成

公式 ：

累计梯度更新：$v\leftarrow\alpha v+(1-\alpha)g$，其中，$\alpha$为动量参数，$v$为累计梯度，$g$为当前梯度，$\eta$为学习率

权重（x1 , x2）更新：$w\leftarrow w-\eta*v$

优点：

1. 加快收敛能帮助参数在 正确的方向上加速前进

2. 他可以帮助跳出局部最小值

   ![](./.assets/m4ppgm.png)

   

   

   

**实验一 ：**

损失函数： $f(x)=0.1x_1^2+2x_2^2$，$x_1,x_2$初始值分别为-5， -2， 学习率设为 0.4，我们使用 **不带动量**的传统梯度下降法 ，观察其下降过程

预期分析 ： 因为 $x_1,x_2$的系数分别是 0.1 和 2， 这就使得$x_1,x_2$ 的梯度值相差一个量级，如果使用相同的学习率，$x_2$的更新幅度会比$x_1$ 的更大些。

```python
import numpy as np
import matplotlib.pyplot as plt

def loss_func(x1, x2): #定义目标函数
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

x1, x2 = -5, -2
eta = 0.4
num_epochs = 20
result = [(x1, x2)]

for epoch in range(num_epochs):
    gd1 = 0.2 * x1
    gd2 = 4 * x2
    
    x1 -= eta * gd1
    x2 -= eta * gd2
    
    result.append((x1, x2))

plt.plot(*zip(*result), '-o', color='#ff7f0e')
x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
plt.contour(x1, x2, loss_func(x1, x2), colors='#1f77b4')
plt.title('learning rate = {}'.format(eta))
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
```

![](https://p.ipic.vip/b7tuv7.png)

结果分析：与预想一致，使用相同的学习率，$x_2$的更新幅度会较$x_1$的更大些，变化快得多，而 $x_1$ 收敛速度太慢

**实验二 ：**

依然使用**不带动量**的梯度下降算法，将学习率设置为 0.6。

更新过程如下图：

![](https://p.ipic.vip/jstet7.png)

这时，我们会陷入一个两难的选择：

- 如果选择较小的学习率。$x_1$收敛缓慢
- 如果选择较大的学习率，$x_1$方向会收敛很快，但在 $x_2$方向不会收敛

**实验三：**

我们使用**带动量的** 梯度下降法，将**历史的梯度**考虑在内： 动量参数设为 0.5，将学习率设置为 0.4

累计梯度更新：$v\leftarrow\alpha v+(1-\alpha)g$，其中，$\alpha$为动量参数，$v$为累计梯度，$g$为当前梯度，$\eta$为学习率

权重更新(x1, x2)：$w\leftarrow w-\eta*v$

```python
import numpy as np
import matplotlib.pyplot as plt


def loss_func(x1, x2): #定义目标函数
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


x1, x2 = -5, -2
v1, v2 = 0, 0
eta, alpha = 0.4, 0.5
num_epochs = 20
result = [(x1, x2)]

for epoch in range(num_epochs):
    # 累计梯度更新（考虑历史梯度）
    v1 = alpha * v1 + (1 - alpha) * (0.2 * x1)
    v2 = alpha * v2 + (1 - alpha) * (4 * x2)
    
    x1 -= eta * v1
    x2 -= eta * v2
    
    result.append((x1, x2))

plt.plot(*zip(*result), '-o', color='#ff7f0e')
x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
plt.contour(x1, x2, loss_func(x1, x2), colors='#1f77b4')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
```

![](https://p.ipic.vip/qa7ds4.png)

即使我们将学习率设置为0.6， $x_2$的梯度也不会发散了

![](https://p.ipic.vip/j0b1s9.png)

#### Adagrad

Adagrad优化算法 被称为**自适应学习率**优化算法，

之前我们讲的随机梯度下降法，对所有的参数，都是使用相同的、固定的学习率进行优化的，但是**不同的参数的梯度差异可能很大**，使用相同的学习率，效果不会很好

举例：

假设损失函数是$f(x)=x_1^2+10x_2^2$， $x_1,x_2$的初值分别为$x_1=40, x_2=20$

(通过观察，我们即可知道，$x_1=0,x2=0$ 就是两个参数的极值点 )

$\to\frac{\partial loss}{\partial x_1}=80,\frac{\partial loss}{\partial x_2}=400$    $x_1$将要移动的幅度 小于$x_2$将移动的幅度

而$x_1$距离离极值点 $x_1=0$ 是较远的，所以，我们使用梯度下降法，效果并不会好

**Adagrad 思想：对于不同参数，设置不同的学习率**

方法：**对于每个参数**，初始化一个 **累计平方梯度** $r=0$，然后每次将该参数的梯度平方求和累加到这个变量 $r$上：
$$
r\leftarrow r+g^2
$$
然后，在更新这个参数的时候，学习率就变为：
$$
\frac \eta{\sqrt{r+\delta}}
$$

**权重更新**：
$$
w\leftarrow w-\frac\eta{\sqrt{r+\delta}}*g
$$
其中$g$为梯度，$r$为累计平方梯度(初始为0)；$\eta$为学习率，$\delta$为小参数，避免分母为0，一般取$10^{-10}$

这样，不同的参数由于梯度不同，他们对应的$r$大小也就不同，所以学习率也就不同，这也就实现了自适应的学习率。

**总结**： Adagrad 的核心想法就是，如果一个参数的梯度一直都非常大，那么其对应的学习率就变小一点，防止震荡，而一个参数的梯度一直都非常小，那么这个参数的学习率就变大一点，使得其能够更快地更新，这就是Adagrad算法加快深层神经网络的训练速度的核心。(简而言之，就是让梯度小的参数学习率大一点，梯度大的参数学习率小一点，这样更新更平稳)



#### RMSProp

RMSProp：Root Mean Square Propagation 均方根传播

RMSProp 是在 adagrad 的基础上，进一步在学习率的方向上优化

累计平方梯度：$r\leftarrow\lambda r+(1-\lambda)g^2$

参数更新：$w\leftarrow w-\frac\eta{\sqrt{r+\delta}}*g$

其中$g$为梯度，$r$为累计平方梯度(初始为0)；$\lambda$ 为衰减系数，$\eta$为学习率，$\delta$为小参数，避免分母为0，一般取$10^{-10}$

#### Adam

在Grandient Descent 的基础上，做了如下几个方面的改进：

1. 梯度方面增加了momentum，使用累积梯度：$v\leftarrow\alpha v+(1-\alpha)g$

2. 同 RMSProp 优化算法一样, 对学习率进行优化，使用累积平方梯度：$r\leftarrow\lambda r+(1-\lambda)g^2$

3. 偏差纠正：$\hat{v}=\frac v{1-\alpha^t},\quad\hat{r}=\frac r{1-\lambda^t}$

4. 再如上3点改进的基础上，权重更新：
   $$
   w\leftarrow w-\frac\eta{\sqrt{\hat{r}+\delta}}*\hat{v}
   $$

**为啥要偏差纠正？**

第1次更新时，$v_1\leftarrow\alpha v_0+(1-\alpha)g_1$，$v_0$初始值为0，且$\alpha$初始值一般会设置为接近1，因此$t$较小时，$v$的值时偏向于0的

```python
def adam(learning_rate, beta1, beta2, epsilon, var, grad, m, v, t):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    var = var - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return var, m, v
```



### 学习率与warm up策略

#### 学习率

![](./.assets/zh4izh.png)

#### warm up策略

在训练初期，让学习率高一点，能尽快找到比较不错的解

在训练后期，学习率调小一点，能收敛到局部最优解

![](./.assets/9oyd05.png)

李沐的paper中提到，开始时学习率大，训练不稳定

开始时学习率线性增长，然后再降低一点

![](./.assets/ls9qna.png)



### 自定义学习率调度器

```python
torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
```

参数：

- optimizer：优化器对象，表示需要调整学习率的优化器。
- Ir_lambda：一个**函数**，接受一个整数参数 epoch，并返回一个浮点数值，表示当前 epoch 下的学习率变化系数。

![image-20231206112034289](./.assets/image-20231206112034289.png)

![image-20231206112433005](./.assets/image-20231206112433005.png)

![image-20231011222211723](./.assets/r249g1.png)

### 不通层设置不同的学习率

**在迁移学习的时候，可能需要用到如下训练技巧**

1. 将预训练好的部分参数冻结，只训练新加的那一部分网络参数
2. 然后收敛的差不多的时候，再将预训练部分的参数解冻，同时训练所有的网络进行微调

或者是

1. 将 预训练好的 backbone 的 参数学习率设置为较小值
2. backbone 之外的部分 (新增的部分，一般为分类头、检测头，等)，设置为较大的学习率



**学习率设置的方式**

在定义优化器的时候，用 list **将参数设置为不同的组**，每个组( list 中的每个元素 )用字典表示，在字典中指明 参数组、该组的学习率

```python
optimizer = optim.SGD([
    {'params': params_group_1, 'lr': 0.001},   # 参数组 学习率
    {'params': params_group_2, 'lr': 0.0005}]) 
```



```python
optimizer = optim.SGD([
    {'params': base_params},   # 未指定学习率的，使用默认学习率 0.001
    {'params': net.linear3.parameters(), 'lr': 0.0005}],
    lr=0.001, momentum=0.9)
```



params_group_1 和 params_group_2 可以是任何实现了 __iter__() 方法的对象，例如 list、tuple

举例

```python
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim

net = nn.Sequential(OrderedDict([
    ("linear1", nn.Linear(10, 20)),
    ("linear2", nn.Linear(20, 30)),
    ("linear3", nn.Linear(30, 40))]))


linear3_params = list(map(id, net.linear3.parameters()))
base_params = filter(lambda p: id(p) not in linear3_params, net.parameters())

optimizer = optim.SGD([
    {'params': base_params},   # 未指定学习率的，使用默认学习率 0.001
    {'params': net.linear3.parameters(), 'lr': 0.0005}],
    lr=0.001, momentum=0.9)


# print(optimizer)
print(optimizer.param_groups[0]['lr'])
print(optimizer.param_groups[1]['lr']) 
```

### 问题

1. 叶子节点和中间节点的梯度

   Pytorch 使用的是动态图计算，在计算过程中，会生成大量的中间结果，**这些中间结果只在计算过程中使用，计算结束后就会被释放。**

   叶子节点 ： 如下 w1 和 w2 就是叶子节点， y1 和 y2 不是

   **只有叶子节点同时`requires_grad=True`的张量，才能通过`.grad`访问累计梯度**

   ![](./.assets/image-20231206151040947.png)

   Pytorch 默认只会记录并暂时保存叶子节点的梯度(`当使用 optimizer.zero_grad() 时`，梯度会清零)，其他中间节点的梯度只会在求叶子节点梯度的过程中使用到（链式求导过程中），并不会被保存起来

   ```python
   import torch
   
   x1 = torch.tensor(2.)
   w1 = torch.tensor(3., requires_grad=True)
   b1 = torch.tensor(2., requires_grad=True)
   
   x2 = torch.tensor(7.)
   w2 = torch.tensor(3., requires_grad=True)
   # 这里的x1 x2 是固定的量
   
   y1 = x1 * w1 + b1
   y2 = x2 * w2
   z = y1 + y2
   
   print('y1 value is ', y1)
   print('y2 value is ', y2)
   print('z value is ', z)
   print('*'*30)
   
   z.backward()  # 反向传播计算叶子节点的梯度
   print('w1\'s grad is ', w1.grad)
   print('b1\'s grad is ', b1.grad)
   print('w2\'s grad is ', w2.grad)
   print('y1\'s grad is ', y1.grad)
   print('y2\'s grad is ', y2.grad)
   ```

   梯度的优化迭代

   ```python
   import torch
   
   x1 = torch.tensor(2.)
   w1 = torch.tensor(3., requires_grad=True)
   b1 = torch.tensor(2., requires_grad=True)
   
   x2 = torch.tensor(7.)
   w2 = torch.tensor(3., requires_grad=True)
   
   
   y1 = x1 * w1 + b1
   y2 = x2 * w2
   z = y1 + y2
   
   print('y1 value is ', y1)
   print('y2 value is ', y2)
   print('z value is ', z)
   print('*'*30)
   
   z.backward()  # 反向传播计算叶子节点的梯度
   print('w1\'s grad is ', w1.grad)
   print('b1\'s grad is ', b1.grad)
   print('w2\'s grad is ', w2.grad)
   print('*'*30)
   
   # 根据梯度更新参数（梯度下降法）
   # 根据对w1的导数在w=w1处的梯度，更新w1的数值
   lr = 1e-3
   w1.data -= lr * w1.grad.data
   b1.data -= lr * b1.grad.data
   w2.data -= lr * w2.grad.data
   
   print('new w1 is ', w1)
   print('new b1 is ', b1)
   print('new w2 is ', w2)
   ```

   

2. 一个batch的数据如何进行反向传播

   假设 `batch_size =4`，`loss = nn.CrossEntropyLoss()((predicted_label, label))`

   ```
   nn.CrossEntropyLoss() 默认参数 reduction="mean"
   所以，通过 loss = nn.CrossEntropyLoss()((predicted_label, label)) 计算得到的损失值是 batch 中所有样本损失值的平均值
   ```

    执行 `loss.backward()`  

   - 使用该 loss 分别计算 batch 中第1个、第2个、第3个、第4 个样本的（每个参数的）梯度 g1、g2、g3、g4（这里的梯度是多个参数的梯度向量）
   - 计算每个参数的梯度平均值 ：  g =（g1 + g2 + g3 + g4）/ 4

   执行 `optim.step()`   

   - 使用梯度平均值更新参数 ：  $$w \leftarrow w - lr * g$$
   - 所以，一个 batch_size 只会做一次反向传播，参数更新时使用的是样本平均梯度。

## 

## 数据集

### PASCAL VOC 2007&2012



坐标格式（xmin，ymin，xmax，ymax）其中 (xmin, ymin) 是左上角的坐标，（ymin, ymax）是右下角的坐标



1. 简介

   PASCAL 全称：Pattern Analysis, Statical Modeling and Computational Learning

   PASCAL VOC（The PASCAL Visual Object Classes ）是一个经典的计算机视觉数据集，由牛津大学、马里兰大学和微软剑桥研究院的研究人员创建的。 该数据集于2005年首次发布，从那时起就被用于训练和评估目标检测算法。

   PASCAL VOC 从 2005年开始举办挑战赛，每年的内容都有所不同，主要包括：

   - 图像分类（Classification ）
   - 目标检测（Detection）
   - 目标分割（Segmentation）
   - 人体布局（Human Layout）
   - 动作识别（Action Classification）

   我们知道在 ImageNet挑战赛上涌现了一大批优秀的分类模型，而PASCAL挑战赛上则是涌现了一大批优秀的目标检测和分割模型，这项挑战赛已于2012年停止举办了，但是研究者仍然可以在其服务器上提交预测结果以评估模型的性能。

   虽然近期的目标检测或分割模型更倾向于使用MS COCO数据集，但是这丝毫不影响 PASCAL VOC数据集的重要性，毕竟PASCAL对于目标检测或分割类型来说属于先驱者的地位。对于现在的研究者来说比较重要的两个年份的数据集是 PASCAL VOC 2007 与 PASCAL VOC 2012，这两个数据集频频在现在的一些检测或分割类的论文当中出现。

2. 地址汇总

   官网地址：http://host.robots.ox.ac.uk/pascal/VOC/

   官方文档 ： http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf

   |              | Pascal VOC 2007                                              | Pascal VOC 2012                                              |
   | ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | 主页地址     | http://host.robots.ox.ac.uk/pascal/VOC/voc2007/              | http://host.robots.ox.ac.uk/pascal/VOC/voc2012/              |
   | 数据集下载   | [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (450MB tar file) | [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (2GB tar file) |
   | 数据统计信息 | http://host.robots.ox.ac.uk/pascal/VOC/voc2007/dbstats.html  | http://host.robots.ox.ac.uk/pascal/VOC/voc2012/dbstats.html  |
   | 标注标准     | http://host.robots.ox.ac.uk/pascal/VOC/voc2007/guidelines.html | http://host.robots.ox.ac.uk/pascal/VOC/voc2012/guidelines.html |

​			Pascal VOC 2007

​					训练集和验证集 下载地址 ： [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (450MB tar file)

​					测试集（图像 + 标注）下载地址： [annotated test data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) (430MB tar file)

​					测试集（仅标注文件）下载地址： [annotation only](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar) (12MB tar file, no images)

​		  Pascal VOC 2012

​					训练集和验证集 下载地址： [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (2GB tar file)

​					测试集标注未公开

3. 数据集发展与20个类别

   | 年份 | 数据统计                                                     | 发展                                                         | 备注                                                         |
   | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | 2005 | 只有4类别：自行车，汽车，摩托车，人。 训练/验证/测试：1578张图像，包含2209个注释的对象。 | 两项比赛:分类和检测                                          | 这些图片大部分来自现有的公共数据集，不像后来使用的flickr图片那么具有挑战性。 这个数据集已经过时了。 |
   | 2006 | 10类别：自行车、公共汽车、汽车、猫、牛、狗、马、摩托车、人、羊。 训练/验证/测试：包含2618张图像， 4754个注释对象 | 图片来自flickr 和 微软剑桥研究院(MSRC)数据集                 | MSRC的图片比flickr的更简单，因为这些照片通常集中在感兴趣的对象上。 这个数据集已经过时了。 |
   | 2007 | 20类：见下面类别说明 训练/验证/测试：包含9,963张图像，其中包含 24,640个标注对象 | 1、类别数量从10个增加到20个 2、引入了 分割任务 3、引入了 人体布局任务 4、注释中添加了 “截断标志” 5、分类挑战的评价指标改为平均精度(Average Precision)，以前是ROC-AUC | 这一年设立了20个类别，之后类别就再也没有变动 2007年之前（包括2007年）的 test数据集 都是公布的，之后的test数据集就都没有公布 |
   | 2008 | 20类 数据被分割(像往常一样)分割为大约：50% 训练/验证 + 50% 测试 训练/val数据有4,340张图像，其中包含10,363个标注对象。 | 1、遮挡标志被添加到注释中 2、测试数据标注不再公开 3、分割和人物布局数据集包括来自相应VOC2007集的图像 |                                                              |
   | 2009 | 20类 训练/验证：包括7,054张图像，其中包含17,218个ROI标注对象 和 3,211个分割 | 1、从当前开始，所有任务的数据都由前几年的图像和新图像组成。在之前几年，每年都会发布一个全新的数据集用于分类/检测任务。 2、数据扩增使得图像数量每年增长，并且意味着测试结果可以与前几年的进行比较。 3、分割任务 从一个尝试比赛转变成为一个标准的挑战 | 增加的图像 没有标注“困难标记（difficult flags）” (遗漏)。 测试数据标注未公开 |
   | 2010 | 20类 训练/验证：包括10,103个图像，其中包含23,374个ROI标注对象 和 4,203个分割 | 1、引入了 动作识别任务 2、引入了基于 ImageNet的大规模分类的关联挑战 3、Amazon Mechanical Turk 被用于标注的早期阶段 | AP计算方法改变。现在使用所有数据点而不是TREC采样。 测试数据标注未公开 |
   | 2011 | 20类 训练/验证：包括11,530张图像，其中包含27,450个ROI标注对象 和 5,034个分割 | 动作分类扩展到 10个类别 + “其他”。                           | 布局标注 现在不是“完整的”：只有人被注释，并且有些人可能没有被注释。 |
   | 2012 | 20类 训练/验证：包括11,530张图像，其中包含27,450个ROI标注对象 和 6,929个分割 | 1、分割数据集的大小 大幅增加。 2、动作识别数据集 中人体被 增加使用 参考点标注 | 用于 分类、检测 和 人体布局数据集 与 VOC2011相同             |

对于 **分类 和 检测** 来说，下图所示为数据集的发展历程，相同颜色的代表相同的数据集：

![5721695897947_.pic](./.assets/wfo08l.png)

- 05年、06年、07年、08年数据集，为**互斥的，独立的、完全不相同**的数据集
- 09年开始，所有数据集由前几年的部分图像 和 新图像组成

 		 09年的数据集 = 07年部分图像 + 08年部分图像 + 09年新图像

- 10、11 年的数据集，均是在前一年的数据集上进行扩充
- 12 年的数据集 和 11年的数据集一样

虽然 Pascal VOC 2012 和 2007 版本的数据集存在一些共享的部分，但是它们的图像和标注文件在细节上还是有所不同的，因此在使用数据集时需要注意版本和文件的正确匹配。

4. 20个类

   ![](./.assets/klq3sa.png)

   ```python
       "aeroplane": 1,
       "bicycle": 2,
       "bird": 3,
       "boat": 4,
       "bottle": 5,
       "bus": 6,
       "car": 7,
       "cat": 8,
       "chair": 9,
       "cow": 10,
       "diningtable": 11,
       "dog": 12,
       "horse": 13,
       "motorbike": 14,
       "person": 15,
       "pottedplant": 16,
       "sheep": 17,
       "sofa": 18,
       "train": 19,
       "tvmonitor": 20
      
   ```

5. 数据集使用

   目前广大研究者们普遍使用的是 **VOC2007和VOC2012**数据集。

   论文中针对 VOC2007和VOC2012 的具体用法有以下几种：

   - 只用VOC2007的trainval 训练，使用VOC2007的test测试
   - 只用VOC2012的trainval 训练，使用VOC2012的test测试，这种用法很少使用，因为大家都会结合VOC2007使用
   - 使用 VOC2007 的 train+val 和 VOC2012的 train+val 训练，然后使用 VOC2007的test测试，**这个用法是论文中经常看到的 07+12** ，研究者可以自己测试在VOC2007上的结果，因为VOC2007的test是公开的。
   - 使用 VOC2007 的 train+val+test 和 VOC2012的 train+val训练，然后使用 VOC2012的test测试，这个用法是论文中经常看到的 07++12 ，这种方法**需提交到VOC官方服务器上评估结果，因为VOC2012 test没有公布。**
   - 先在 MS COCO 的 trainval 上预训练，再使用 VOC2007 的 train+val、 VOC2012的 train+val 微调训练，然后使用 VOC2007的test测试，这个用法是论文中经常看到的 07+12+COCO 。
   - 先在 MS COCO 的 trainval 上预训练，再使用 VOC2007 的 train+val+test 、 VOC2012的 train+val 微调训练，然后使用 VOC2012的test测试 ，这个用法是论文中经常看到的 07++12+COCO，这种方法需提交到VOC官方服务器上评估结果，因为VOC2012 test没有公布。

6. 数据集结构

   - PASCAL VOC 2007

     ```python
     .
     └── VOCdevkit
         └── VOC2007
             ├── Annotations                 标注文件（图像分类、目标检测、人体布局)
             │   ├── 000005.xml
             │   ├── 000007.xml
             │   ├── 000009.xml
             │   └── ... (共 5011个标注文件)
             ├── ImageSets               划分 数据集分割信息 （训练集、验证集、训练集+验证集）
             │   ├── Layout                  用于人体布局图像信息
             │   │   ├── train.txt
             │   │   ├── trainval.txt
             │   │   └── val.txt
             │   ├── Main                    用于图像分类和目标检测图像信息
             │   │   ├── train.txt          
             │   │   ├── trainval.txt       
             │   │   ├── val.txt            
             │   │   └── ... (共63个文件)
             │   └── Segmentation            用于语义分割和实例分割图像信息
             │       ├── train.txt
             │       ├── trainval.txt
             │       └── val.txt
             ├── JPEGImages                  所有原图像
             │   ├── 000005.jpg
             │   ├── 000007.jpg
             │   ├── 000009.jpg
             │   └── ... (共5011张图像)
             ├── SegmentationClass           语义分割标注图像
             │   ├── 000032.png
             │   ├── 000033.png
             │   ├── 000039.png
             │   └── ... (共422张图像)
             └── SegmentationObject          实例分割标注图像
                 ├── 000032.png
                 ├── 000033.png
                 ├── 000039.png
                 └── ... (共422张图像)
     ```

     ![](./.assets/biiapc.png)

   - PASCAL  VOC 2012

     ```
     .
     └── VOCdevkit
         └── VOC2012
             ├── Annotations                  标注文件（图像分类、目标检测、人体布局)
             │   ├── 2007_000027.xml
             │   ├── 2007_000032.xml
             │   ├── 2007_000033.xml
             │   ├── 2007_000039.xml
             │   └── ...(共17125张图像)
             ├── ImageSets                     数据集分割信息 （训练集、验证集、训练集+验证集）
             │   ├── Action                      用于动作识别
             │   │   ├── train.txt                2296张图像
             │   │   ├── trainval.txt             4588张图像
             │   │   ├── val.txt                  2292张图像
             │   │   └── ...
             │   ├── Layout                      用于人体布局
             │   │   ├── train.txt                4425张图像
             │   │   ├── trainval.txt             850张图像
             │   │   └── val.txt                  425张图像
             │   ├── Main                        用于图像分类和目标检测  
             │   │   ├── train.txt                5717张图像 
             │   │   ├── train_val.txt            11540张图像
             │   │   └── trainval.txt             5823张图像 
             │   └── Segmentation                用于语义分割和实例分割 
             │       ├── train.txt                 1464张图像
             │       ├── trainval.txt              2913张图像
             │       └── val.txt                   1449张图像
             ├── JPEGImages                     所有原图像
             │   ├── 2007_000027.jpg
             │   ├── 2007_000032.jpg
             │   ├── 2007_000033.jpg
             │   ├── 2007_000039.jpg
             │   └── ...(共17125张图像)
             ├── SegmentationClass              语义分割标注图像
             │   ├── 2007_000032.png
             │   ├── 2007_000033.png 
             │   ├── 2007_000039.png
             │   ├── 2007_000042.png
             │   └── ...（共2913张图像）
             └── SegmentationObject             实例分割标注图像
                 ├── 2007_000032.png
                 ├── 2007_000033.png
                 ├── 2007_000039.png
                 ├── 2007_000042.png
                 └── ...（共2913张图像）
     ```

   - 2007和2012区别

     **Pascal VOC 2012 的数据集 因为是在前几年的数据集上进行扩增，所以文件名中包含年份，而 Pascal VOC 2007 的文件名中不包含**

     - Pascal VOC 2007 的标注文件名 和 图像文件名 类似为： 000005.xml、 000005.jpg
     - Pascal VOC 2012 的标注文件名 和 图像文件名 类似为： 2007_000027.xml、 2007_000039.png

     **Pascal VOC 2012 的 ImageSets 中包括 Action 文件：用于动作识别任务的数据集划分，而 Pascal VOC 2007 的 ImageSets 文件中不包含， 因为 动作识别任务（Action Classification） 是2010年才有的。**

     **.xml 的标注文件内容 有所不同**，比如： 12版本中有的图像标注 是有 动作信息

7. 标注文件结构

   - 目标检测 Annoation

     ```xml
     <annotation>
             <folder>VOC2007</folder>
             <filename>000001.jpg</filename>
             <source>
                     <database>The VOC2007 Database</database>
                     <annotation>PASCAL VOC2007</annotation>
                     <image>flickr</image>
                     <flickrid>341012865</flickrid>
             </source>
             <owner>
                     <flickrid>Fried Camels</flickrid>
                     <name>Jinky the Fruit Bat</name>
             </owner>
             <size>
                     <width>353</width>
                     <height>500</height>
                     <depth>3</depth>
             </size>
             <segmented>0</segmented>
             <object>
                     <name>dog</name>
                     <pose>Left</pose>
                     <truncated>1</truncated>
                     <difficult>0</difficult>
                     <bndbox>
                             <xmin>48</xmin>
                             <ymin>240</ymin>
                             <xmax>195</xmax>
                             <ymax>371</ymax>
                     </bndbox>
             </object>
             <object>
                     <name>person</name>
                     <pose>Left</pose>
                     <truncated>1</truncated>
                     <difficult>0</difficult>
                     <bndbox>
                             <xmin>8</xmin>
                             <ymin>12</ymin>
                             <xmax>352</xmax>
                             <ymax>498</ymax>
                     </bndbox>
             </object>
     </annotation>
     ```

     - `annotation`：标注文件的根节点，包含了整个标注信息
     - `folder`：图像所在的文件夹名称
     - `filename`：图像的文件名
     - `source`：图像来源
     - `owner`：图像拥有者
     - `size`：图像的尺寸信息，包括宽度、高度、深度。
     - `segmented`：是否被分割标注过： 值为 0，未被过分割；值为 1，被分割标注。
     - `object`：图像中的一个物体，其中的 信息包括：
       - `name`：物体的类别名称， 20个类别
       - `bndbox`：物体的边界框信息，包括左上角和右下角的坐标
         - `xmin`：边界框左上角的 x 坐标
         - `ymin`：边界框左上角的 y 坐标
         - `xmax`：边界框右下角的 x 坐标
         - `ymax`：边界框右下角的 y 坐标
       - `difficult`：标记物体是否难以识别的标志，0 表示容易识别，1 表示难以识别
       - `truncated`：标记物体是否被截断：0 表示未被截断，1 表示被截断（比如在图片之外，或者被遮挡超过15%）
     - `pose`：标记物体的姿态，例如正面、侧面等

   - 语义分割标注图像Segmentation Class

     - 背景部分的 标注像素值 为 0
     - 边界部分的标注像素值为 255
     - 难以分割的区域，例如有重叠物体或遮挡的区域，标注像素值为255
     - 被分割出的object 内部， 标注像素值为其类别索引。 比如，被分割的飞机部分的像素值为飞机类别索引值 1

     ![WeChat550d2504c599e59a3d245f9f2bfc76b9](./.assets/4ky925.png)

   - 实例分割标注图像Segmentation Object

     - 背景部分的 标注像素值 为 0
     - 边界部分的标注像素值为 255
     - 难以分割的区域，例如有重叠物体或遮挡的区域，标注像素值为255
     - **被分割出的 object 内部，使用 物体实例的 ID 来标识它。物体实例的 ID ：为该物体在 `.xml` 标注文件中的 index 。比如，在 `.xml` 标注文件中，排位第2个的 object，ID = 2，在标注图像中，该 object 的像素值，就为2**

     ![5741695898563_.pic](./.assets/0hcel8.png)

8. 数据集解析--目标检测任务

   将数据集转换为**yolo格式**

   ```python
   import xml.etree.ElementTree as ET
   import os
   
   
   # voc的20个类别
   classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
   
   
   def convert(size, bbox):
       x = (bbox[0] + bbox[1]) / 2.0
       y = (bbox[2] + bbox[3]) / 2.0
       w = bbox[1] - bbox[0]
       h = bbox[3] - bbox[2]
       x = x / size[0]
       w = w / size[0]
       y = y / size[1]
       h = h / size[1]
       return (x, y, w, h)
   
   
   def convert_annotation(xml_file, save_file):
   
       # 保存yolo格式 的label 的 .txt 文件地址
       save_file = open(save_file, 'w')
   
       tree = ET.parse(xml_file)
       size = tree.find('size')
       w = int(size.find('width').text)
       h = int(size.find('height').text)
   
       for obj in tree.findall('object'):
           difficult = obj.find('difficult').text
           cls = obj.find('name').text
           if cls not in classes or int(difficult) == 1:
               continue
           cls_id = classes.index(cls) + 1   # 类别索引从1开始，类别0是背景
           bbox = obj.find('bndbox')
           b = (float(bbox.find('xmin').text),
                float(bbox.find('xmax').text),
                float(bbox.find('ymin').text),
                float(bbox.find('ymax').text))
           bb = convert((w, h), b)
           save_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
       save_file.close()
   
   
   if __name__ == "__main__":
       # 数据集根目录地址
       data_root = "/Users/enzo/Documents/GitHub/dataset/VOCdevkit/VOC2007"
   
       # 标注文件地址
       annotation = os.path.join(data_root, 'Annotations')
   
       # yolo格式的文件保存地址
       save_root = './labels'
       if not os.path.exists(save_root):
           os.makedirs(save_root)
   
       for train_val in ["train", "val"]:
           if not os.path.exists(os.path.join(save_root, train_val)):
               os.makedirs(os.path.join(save_root, train_val))
   
           # 数据集划分的 .txt 文件地址
           txt_file = os.path.join(data_root, 'ImageSets/Main', train_val+'.txt')
   
           with open(txt_file, 'r') as f:
               lines = f.readlines()
           file_names = [line.strip() for line in lines if len(line.strip())>0]
   
           for file_name in file_names:
               xml_file = os.path.join(annotation, file_name+'.xml')
               save_file = os.path.join(save_root, train_val, file_name+'.txt')
               convert_annotation(xml_file, save_file)
   ```

   生成的文件结构

   ![](https://p.ipic.vip/znfcia.png)

   .txt 文件内容， 以 `labels/val/000005.txt` 举例：

   每行的5个值表示 ：【 label，center_x ，center_y，height，width】

   都是归一化之后的值，bbx的高宽 / 整个图像的高宽

   ![WeChat94f3ab0ed22213a0f795b9a56f43c0c8](./.assets/dmife9.png)

​			![](https://p.ipic.vip/mq58q2.png)



说明：.xml 文件中有 5个 object， 其中2个difficult=1， 没有被转存出来

### MS COCO



坐标格式（x，y，w，h），其中 x ,y 是左上角的坐标



1. 简介

   MS COCO ：Microsoft Common Objects in Context

    是一个由微软公司创建的用于**图像识别**和**物体检测**的大型数据集。

   官网地址：https://cocodataset.org/

   论文地址: https://arxiv.org/pdf/1405.0312.pdf

2. 数据集说明

   - 数据集特征

     ** stuff 指的是 天空，街道， 草地等这种没有明显边界的目标

     ![5681695896943_.pic](./.assets/676skv.png)

   - coco 2014 和 coco2017区别

     - 2017版数据集 是对 2014版数据集 的扩展和更新

     - 2017版 和 2014版 使用完全相同的图像，不一样的是：

       - 2017版 **训练集/验证集 的划分是 118K/5K**
       - 2014版 是 83K/41K

     - 2017版中用于 检测类任务 / 关键点检测 的注释 和 2014版 是一样的，

       但是增加了 40K 张训练图像 （118k 训练集中的子集） 和 所有验证集 的stuff 标注 （后面有介绍 stuff categories）

       2017年的测试集只有两个部分(开发集/挑战集)，而2014版的测试集有四个部分(开发集/标准集/储备集/挑战集)。

       2017版 发布12万张来自COCO的无标记图像，这些图像遵循与标记图像相同的类分布 ，可用于半监督学习。

   - 80个类

     ![5701695897020_.pic](./.assets/2lh2sc.png)

3. 数据集下载

   官网下载地址：https://cocodataset.org/#download

   或者，直接点击如下链接直接下载 ：

   - 2017 - 训练数据集 ： [2017 Train images [118K/18GB\]](http://images.cocodataset.org/zips/train2017.zip)
   - 2017 - 验证数据集： [2017 Val images [5K/1GB\]](http://images.cocodataset.org/zips/val2017.zip)
   - 2017 - 标注文件 ： [2017 Train/Val annotations [241MB\]](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

4. 数据集解析

   - 目标检测任务

     数据结构

     ```python
     .
     ├── annotations
     │   ├── captions_train2017.json          图像描述的 训练集标注文件
     │   ├── captions_val2017.json            图像描述的 验证集标注文件
     │   ├── instances_train2017.json         对应目标检测、分割任务的 训练集标注文件
     │   ├── instances_val2017.json           目标检测、分割任务的 验证集标注文件
     │   ├── person_keypoints_train2017.json  人体关键点检测的 训练集标注文件
     │   └── person_keypoints_val2017.json    人体关键点检测的 验证集标注文件
     ├── train2017
     │   ├── 000000000009.jpg
     │   ├── 000000000025.jpg
     │   ├── 000000000030.jpg
     │   ├── 000000000034.jpg
     │   ├── 000000000036.jpg
     │   └── ... (共118287张图像)
     └── val2017
         ├── 000000000139.jpg
         ├── 000000000285.jpg
         ├── 000000000632.jpg
         ├── 000000000724.jpg
         ├── 000000000776.jpg
         └── ... (共5000张图像)
     ```

     **instances_train2017.json 文件的数据结构如下 （ instances_val2017.json 文件结构也是一样 ）**

     ![5711695897153_.pic](./.assets/nhc8b4.png)

5. 80个类别

   | supercategory | id   | name           |
   | ------------- | ---- | -------------- |
   | person        | 1    | person         |
   | vehicle       | 2    | bicycle        |
   | vehicle       | 3    | car            |
   | vehicle       | 4    | motorcycle     |
   | vehicle       | 5    | airplane       |
   | vehicle       | 6    | bus            |
   | vehicle       | 7    | train          |
   | vehicle       | 8    | truck          |
   | vehicle       | 9    | boat           |
   | outdoor       | 10   | traffic light  |
   | outdoor       | 11   | fire hydrant   |
   | outdoor       | 13   | stop sign      |
   | outdoor       | 14   | parking meter  |
   | outdoor       | 15   | bench          |
   | animal        | 16   | bird           |
   | animal        | 17   | cat            |
   | animal        | 18   | dog            |
   | animal        | 19   | horse          |
   | animal        | 20   | sheep          |
   | animal        | 21   | cow            |
   | animal        | 22   | elephant       |
   | animal        | 23   | bear           |
   | animal        | 24   | zebra          |
   | animal        | 25   | giraffe        |
   | accessory     | 27   | backpack       |
   | accessory     | 28   | umbrella       |
   | accessory     | 31   | handbag        |
   | accessory     | 32   | tie            |
   | accessory     | 33   | suitcase       |
   | sports        | 34   | frisbee        |
   | sports        | 35   | skis           |
   | sports        | 36   | snowboard      |
   | sports        | 37   | sports ball    |
   | sports        | 38   | kite           |
   | sports        | 39   | baseball bat   |
   | sports        | 40   | baseball glove |
   | sports        | 41   | skateboard     |
   | sports        | 42   | surfboard      |
   | sports        | 43   | tennis racket  |
   | kitchen       | 44   | bottle         |
   | kitchen       | 46   | wine glass     |
   | kitchen       | 47   | cup            |
   | kitchen       | 48   | fork           |
   | kitchen       | 49   | knife          |
   | kitchen       | 50   | spoon          |
   | kitchen       | 51   | bowl           |
   | food          | 52   | banana         |
   | food          | 53   | apple          |
   | food          | 54   | sandwich       |
   | food          | 55   | orange         |
   | food          | 56   | broccoli       |
   | food          | 57   | carrot         |
   | food          | 58   | hot dog        |
   | food          | 59   | pizza          |
   | food          | 60   | donut          |
   | food          | 61   | cake           |
   | furniture     | 62   | chair          |
   | furniture     | 63   | couch          |
   | furniture     | 64   | potted plant   |
   | furniture     | 65   | bed            |
   | furniture     | 67   | dining table   |
   | furniture     | 70   | toilet         |
   | electronic    | 72   | tv             |
   | electronic    | 73   | laptop         |
   | electronic    | 74   | mouse          |
   | electronic    | 75   | remote         |
   | electronic    | 76   | keyboard       |
   | electronic    | 77   | cell phone     |
   | appliance     | 78   | microwave      |
   | appliance     | 79   | oven           |
   | appliance     | 80   | toaster        |
   | appliance     | 81   | sink           |
   | appliance     | 82   | refrigerator   |
   | indoor        | 84   | book           |
   | indoor        | 85   | clock          |
   | indoor        | 86   | vase           |
   | indoor        | 87   | scissors       |
   | indoor        | 88   | teddy bear     |
   | indoor        | 89   | hair drier     |
   | indoor        | 90   | toothbrush     |

### Yolo数据集格式



坐标格式（cx，cy，w，h）其中 cx，cy 是中心点的坐标



1. 格式介绍

   - YOLO数据集格式: 使用 .txt 的文本文件 来存储目标检测任务中使用的图像标注信息

   - 每张图像的 bbox标注 存储在一个 .txt 文件中

   - .txt 文件中的每一行，为图像中的一个bounding box 信息

     ```
     <object-class> <x> <y> <width> <height>
     ```

   - `object-class` ：对象类别

   - `x` ：bounding box 的中心点 x坐标。是**归一化**后的值（由图像的宽度归一化），取值范围为[0, 1]

   - `y` ：bounding box 的中心点y坐标，是**归一化**后的值（由图像的高度归一化），取值范围为[0, 1]

   - `width`：bounding box 的宽度，是**归一化**后的值（由图像的宽度归一化）。取值范围为[0,1]。

   - `height`：bounding box 的高度，是**归一化**后的值（由图像的高度归一化）。取值范围为[0,1]。

2. 举例

   一张图像中有 1匹马 和 3个人

   ![](./.assets/85hhxw.png)

   其标注文件

   ```python
   13 0.339 0.6693333333333333 0.402 0.42133333333333334
   15 0.379 0.5666666666666667 0.158 0.38133333333333336
   15 0.612 0.7093333333333334 0.084 0.3466666666666667
   15 0.555 0.7026666666666667 0.078 0.34933333333333333
   ```

   类别信息

   ```python
   {
       "aeroplane": 1,
       "bicycle": 2,
       "bird": 3,
       "boat": 4,
       "bottle": 5,
       "bus": 6,
       "car": 7,
       "cat": 8,
       "chair": 9,
       "cow": 10,
       "diningtable": 11,
       "dog": 12,
       "horse": 13,
       "motorbike": 14,
       "person": 15,
       "pottedplant": 16,
       "sheep": 17,
       "sofa": 18,
       "train": 19,
       "tvmonitor": 20
   }
   ```

3. x、y、width、height的归一化

   以上面的例子中的 horse 来说：

   - 图像的尺寸是： （image_width， image_height）=（500， 375）
   - horse 的 左上角坐标为：（xmin，ymin）=（69，172）
   - horse 的 右下角坐标为：（xmax，ymax）=（270，330）

   计算 x_center、y_center、bbox_width、bbox_height

   - 中心点x坐标为：`x_center` =（xmin + xmax) / 2 = 169.5
   - 中心点y坐标为：`y_center` =（ymin + ymax) / 2 = 251.0
   - bbox 的宽度：`bbox_width` = xmax - xmin = 201.0
   - bbox 的高度：`bbox_height` = ymax - ymin = 158.0

   归一化：

   - `x` = x_center / image_width = 0.339
   - `y` = y_center / image_height = 0.6693333333333333
   - `w` = bbox_width / image_width = 0.402
   - `h` = bbox_height / image_height = 0.42133333333333334

## 基础算法

### 插值算法

上采样中会用到

#### 最近邻插值

简介：将每个目标像素找到距离它最近的原图像像素点，然后将该像素的值字节赋值给目标像素

![](./.assets/image-20231204222824641.png)

优点： 实现简单，计算速度快

缺点：插值结果缺乏连续性，可能会产生锯齿状的边缘，对于图像质量的影响较大。

因此，当处理精度要求较高的图像时，通常会采用更加精细的插值算法，如双线性插值、双三次插值等。

![image-20231204223138780](./.assets/image-20231204223138780.png)

计算前后的缩放因子

```python
import numpy as np
from PIL import Image


def nearest_neighbor_interpolation(image, scale_factor):
    """
    最邻近插值算法
    :param input_array: 输入图像数组
    :param output_shape: 输出图像的 shape
    :return: 输出图像数组
    """
    # 输入图像、输出图像的宽高
    height, width = image.shape[:2]
    out_height, out_width = int(height * scale_factor), int(width * scale_factor)

    # 创建输出图像
    output_image = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    # 遍历输出图像的每个像素，分别计算其在输入图像中最近的像素坐标，并将其像素值赋值给当前像素
    for out_y in range(out_height):
        for out_x in range(out_width):
            # 计算当前像素在输入图像中的坐标
            input_x = round(out_x / scale_factor)
            input_y = round(out_y / scale_factor)
            # 判断计算出来的输入像素坐标是否越界，如果越界则赋值为边界像素
            input_x = min(input_x, width - 1)
            input_y = min(input_y, height - 1)
            # 将输入像素的像素值赋值给输出像素
            output_image[out_y, out_x] = image[input_y, input_x]
    return output_image


# 读取原始图像
input_image = Image.open('original_image.jpg')
image_array = np.array(input_image)

# 输出缩放后的图像
output_array = nearest_neighbor_interpolation(image_array, 1.5)
output_image = Image.fromarray(output_array)

input_image.show()
output_image.show()
```



#### 双线性插值

#### 双三次插值

### IOU

交并比IoU是 计算两个检测框之间重叠程度的一种方式

在目标检测任务中，IoU也用于衡量**模型预测的边界框**与**真实世界的边界框**之间的匹配程度

![](./.assets/扫描件_X Aym_1.jpg)

![image-20231015083830664](./.assets/ynekcj.png)

当出现以下两种特殊情况

![](./.assets/image-20231204220214386.png)

第一种不影响计算交并比，第二种计算出来的h和w为负数，这时候要进行简单的判断，让交集为0

```python
import torch

def box_iou(boxes1, boxes2):
    # bbox1 和 bbox2 的面积
    # (x1, y1, x2, y2)
    area1 = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
    area2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

    # 交集的左上角点 和 右下角点的坐标
    lt = torch.max(boxes1[:2], boxes2[:2]) #(20, 20)
    rb = torch.min(boxes1[2:], boxes2[2:]) #(30, 30)

    # 交集的面积
    wh = (rb - lt).clamp(min=0) #最小是0不能出现负数
    inter = wh[0] * wh[1]

    # 并集的面积
    union = area1 + area2 - inter

    # 交并比
    iou = inter / union
    return iou


boxes1 = torch.tensor([10, 10, 30, 30])
boxes2 = torch.tensor([20, 20, 40, 40])
iou1 = box_iou(boxes1, boxes2)
print(iou1)
```



### NMS

非极大值抑制

![](./.assets/image-20231213115118287.png)

1. NMS是一种常用的目标检测算法中的后处理方法
2. 作用：删除冗余的候选框，从而提高目标检测的精度

![](./.assets/image-20231204221841839.png)

- 将候选框根据置信度进行排序，保留置信度最大的检测框

  ![](./.assets/image-20231204221934518.png)

- 分别计算保留框与所有候选框的IoU，丢弃IoU值大于阈值（0.5）的检测框

  ![](./.assets/image-20231204222039023.png)

- 再根据置信度进行排序，保留置信度最大的候选框，重复上述步骤，直到没有候选框

- 实际的多类别目标检测任务中，是针对每一个类单独进行的NMS，最终得到筛选完成的候选框

  ![](./.assets/image-20231204222245546.png)

## 必备技能

### 命令行参数解析 argparse

1. 简介

   argparse 模块是 Python 标准库中提供的一个 **命令行解析模块** ，它可以让使用者以类似 Unix/Linux 命令参数的方式输入参数（**在终端以命令行的方式指定参数**），argparse 会自动将命令行指定的参数解析为 Python 变量，从而让使用者更加快捷的处理参数。

   ```python
   import argparse
   
   parser = argparse.ArgumentParser(description="description")
   
   parser.add_argument('-gf', '--girlfriend', choices=['jingjing', 'lihuan'])
   parser.add_argument('food')
   parser.add_argument('--house', type=int, default=0)
   
   args = parser.parse_args()
   print('args :',args)
   print('girlfriend :', args.girlfriend)
   print('food :', args.food)
   print('house :', args.house)
   ```

   

2. 使用步骤

   1. 导入argparse模块，并创建解释器

      ```python
      import argparse
      
      # 创建解释器
      parser = argparse.ArgumentParser(description="可写可不写，此处会在命令行参数出现错误的时候，随着错误信息打印出来。")
      ```

   2. 添加所需参数

      ```python
      parser.add_argument('-gf', '--girlfriend', choices=['jingjing', 'lihuan'])
      # --girlfriend 代表完整的参数名称，可以尽量做到让人见名知意，需要注意的是如果想通过解析后的参数取出该值，必须使用带--的名称
      # -gf 代表参数名缩写，在命令行输入 -gf 和 --girlfriend 的效果是一样的，用简称的作用是简化参数输入
      # choices 代表输入参数的值只能是这个choices里面的内容，其他内容则会报错
      
      parser.add_argument('food')
      # 该种方式则要求必须输入该参数； 输入该参数不需要指定参数名称，指定反而会报错，解释器会自动将输入的参数赋值给food
      
      parser.add_argument('--house', type=int, default=0)
      # type  代表输入参数的类型，从命令行输入的参数，默认是字符串类型
      # default 如果不指定该参数的值，则会使用该默认值
      
      parser.add_argument('--modelname', '-m', type=str, required=True, choices=['model_A', 'model_B'])
      # required 参数用于指定参数是否必需。如果设置为 True，则在命令行中必须提供该参数，否则将引发异常。
      ```

   3. 解析参数

      ```python
      # 进行参数解析
      args = parser.parse_args() 
      print('args :', args)
      print('gf :', args.girlfriend)
      print('food :', args.food)
      print('house :', args.house)
      ```

   4. 结果

      ![image-20230924164226469](./.assets/51okka.png)

3. 其他参数说明

   1. action

      向 `add_argument` 方法中传入参数 `action=‘store_true’/‘store_false’` ，解析出来就是 **bool**型 参数 True/False，具体规则为:

      - store_true：如果未指定该参数，默认状态下其值为False；若指定该参数，将该参数置为 True
      - store_false：如果未指定该参数，默认状态下其值为True；若指定该参数，将该参数置为 False

      ```python
      import argparse
      
      parser = argparse.ArgumentParser(description="description")
      
      parser.add_argument('--pa', '-a', action='store_true')  # 指定为true
      parser.add_argument('--pb', '-b', action="store_false") # 指定为false
      args = parser.parse_args() 
      print(args)
      ```

      若 该参数 同时指定了 **action** 和 **default**，在**未指定该参数的情况下**，以 default 值为准；在**指定该参数的情况**下，以 action 的值为准。

   

   上面的操作，我们是通过 命令行 给 python 程序传递参数，一般适合于我们在 GPU 等 linux 操作系统中训练模型使用。 但是**在平时我们调试算法的时候就会很不方便，没有办法利用 IDE 进行debug**，所以接下来介绍如何在 Pycharm 中传递参数给 argparse，方便 pycharm 进行运行这类python程序。

   1、如下图，点击 Edit configurations

   ![在这里插入图片描述](./.assets/o594kv.png)

   2、在Parameters中添加需要的参数（只写参数），多个参数之间用空格隔开

   ![在这里插入图片描述](./.assets/s7k00n.png)

   3、最后直接运行，可以看到，把我们设置在Parameters里面设置的参数，会在Run的时候自动补全

   ![在这里插入图片描述](./.assets/g8jkzk.png)

### linux命令

[见我的另一个文档](D:\Note\linux常用文件管理命令.md)

### requirements.txt 导出 / 搭建

1. 生成requirement.txt文件到当前工作目录下

   ```bash
   pip freeze > ./requirements.txt
   ```

2. 安装requiremen.txt的包

   ```bash
   pip install -r ./requirements.txt
   ```

### 进度条tqdm

1. tqdm基础使用

   ```python
   import time
   from tqdm import tqdm
   
   for i in tqdm(range(100)):
       time.sleep(0.01)
   ```

   - `tqdm(range(i))`的简约写法 ：`trange(i)`

   ```python
   from tqdm import trange
   for t in trange(100):
       time.sleep(0.01)
   ```

2. 手动更新进度条(update方法)

   ```python
   pbar = tqdm(total=200)
   
   for i in range (20):
       time.sleep(1)
       pbar.update(10)
   
   pbar.close()
   ```

   

3. 实际应用

   ```python
   for i, (train_data, train_label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
           train_data, train_label = train_data.to(device), train_label.to(device)
   ```

   

### \_\_call\_\_ 方法

_call__ 是 Python 中的一个魔法方法，也称为类方法。

它的作用是**将类的实例变成可调用对象**，类似于**像函数一样被调用**。



举例

```python
class MyClass:
    def __call__(self, x, y):
        return x + y


obj = MyClass()
print(obj(1, 2))
```

其他类

```python
class MyClass:
    def add(self, x, y):
        return x + y


obj = MyClass()
print(obj.add(1, 2))
```

所以，不用像普通对象调用方法，不用写成 ：`obj.__call__(1, 2)` ， 

而是像直接使用函数一样，直接使用 `obj(1, 2)` 即可

### 文件路径处理

#### pathlib

https://peps.python.org/pep-0428/

pathlib模块是Python标准库的一部分。它在Python 3.4版本中首次引入，提供了一种面向对象的文件系统路径表示方式。

1. 获取路径、拼接路径

   ```python
   from pathlib import Path
   
   # 获取home路径
   print(Path.home())  # /Users/enzo
   
   # 获取当前工作路径(必须在终端中cd到当前脚本所在的目录才有效)
   print(Path.cwd())  # /Users/enzo/Documents
   
   # 拼接路径
   file_path = Path.cwd() / "test.py"
   print(file_path)   # /Users/enzo/Documents/test.py
   ```

2. 常用属性

   ```python
   from pathlib import Path
   
   # 生成路径对象
   path = Path("/users/enzo/Documents/test.py")
   print(path)  # /users/enzo/Documents/test.py
   print(path.name)  # test.py 名称
   print(path.stem)  # test    前缀
   print(path.suffix)  # .py   后缀
   print(path.anchor)  # /
   print(path.parent)  # /users/enzo/Documents 父亲
   print(path.parent.parent)  # /users/enzo    父亲的父亲
   ```

3. 检查文件是否存在

   ```python
   from pathlib import Path
   
   filename = Path("/users/enzo/Documents/test.py")
   print(filename.exists())
   ```

4. 创建文件

   ```python
   from pathlib import Path
   
   filename = Path("/users/enzo/Documents/demo.py")
   if not filename.exists():
       filename.touch()
   ```

5. 创建目录（**创建的目录是从当前所在的终端路径开始的**）

   ```python
   import pathlib
   
   # 创建目录
   pathlib.Path("my_dir").mkdir()
   
   # 创建父目录
   pathlib.Path("my_dir/sub_dir").mkdir(parents=True)
   
   # 即使目录已存在，也继续创建
   pathlib.Path("my_dir").mkdir(exist_ok=True)
   ```

   参数：

   - `path`：要创建的目录路径，可以是字符串或者path对象
   - `parents`：是否创建父目录，默认为false，如果为true，会在创建所有目标目录之前创建父目录
   - `exist_ok`：如果目录存在，是否继续创建，默认为false，如果为true，即使目录存在也不会抛出异常

# 服务器

## 软链接

**创建软链接**

ln  -s  [源文件或目录]  [目标文件或目录]

例如：

- 当前路径创建test 引向/var/www/test 文件夹 


​	`ln –s  /var/www/test  test`

- 创建/var/test 引向/var/www/test 文件夹 


​	`ln –s  /var/www/test   /var/test` 



运行这条命令后，便在**当前目录**下创建了一个**nuscenes**的文件夹（实则为链接），其中的内容就是/data/nuscenes文件夹下的内容

```bash
ln -s /data/nuscenes nuscenes
```

**软链接删除**

`ls -l`查看链接情况

![image-20231212103524558](./.assets/image-20231212103524558.png)

`rm flower_data`源文件还在

## vscode相关

### vscode配置远程服务器

ssh config 毫末智行 A100 * 2 160 G显存



数据集：`/mnt/share_disk/wangbin/`



### vscode 配置有代理的服务器

1. 下载nmap

2. 找到nmap中ncat.exe的地址

   ![](./.assets/image-20231129130752996.png)

3. 配置vscode 的sshconfig





### vscode debug复杂项目 

**python a.py -- args**

[掌握VSCode远程调试技巧：轻松Debug复杂Python项目_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1794y1B7M5/?spm_id_from=333.337.search-card.all.click&vd_source=ddc3faf2cc3b56c47bf503fde12217e3)

1. 我的文件树路径，我要对BEVFormer进行Debug

   ![image-20240111102717272](./.assets/image-20240111102717272.png)

2. 配置launch.json文件

   cwd：是debug时跳转到的目录，以后所有的路径都是基于cwd下的路径

   ```json
   {
       // Use IntelliSense to learn about possible attributes.
       // Hover to view descriptions of existing attributes.
       // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Python: Current File",
               "type": "python",
               "request": "launch",
               "program": "/root/miniconda3/envs/bevformer/lib/python3.8/site-packages/torch/distributed/launch.py",
               "console": "integratedTerminal",
               "justMyCode": true,
               "cwd": "/root/tianyao/BEVFormer", // debug时初始跳转到的目录
               "args": [ //Python 调试配置中传递给 launch.py 脚本的命令行参数
                   "--nproc_per_node=1", // 指定每个节点的进程数为 1，这表明在分布式训练中每个计算节点只运行一个进程
                   "--master_port", // 指定主进程的端口号，这里设置为 24524。在分布式训练中，不同的进程需要通过网络通信，主端口是协调它们之间通信的一种方式
                   "24524",
                   "./tools/train.py", // 指定要运行的 Python 脚本的路径  从这里往下都是train.py需要传递的参数
                   "./projects/configs/bevformer/bevformer_base.py", //指定训练配置文件的路径
                   "--gpus", //指定GPU个数
                   "1",
                   "--work-dir", //指定工作目录
                   "./my_work",
                   "--launcher", // 指定启动器，这里是使用 PyTorch 启动器
                   "pytorch",
                   "--deterministic" // 启用确定性训练，这意味着训练过程将是确定性的，每次运行相同的输入都会产生相同的输出
               ]
           }
       ]
   }
   
```

![image-20240111104231839](./.assets/image-20240111104231839.png)

3. 依次是

   - 继续/暂停 `F5`：执行到下一个断点
   - 单步跳过 `F10`：从断点处执行单步调试
   - 单步调试 `F11`：进入函数内部
   - 单步跳出 `shift+F11`：跳出函数内部
   - 重启`shift+command+F11`
   - 结束`shift+F5`




**pdb方式**

在需要打印信息的地方写这个，然后在控制台执行	

```python
import pdb;pdb.set_trace()
```

### vscode使用anconda环境



## Linux命令

**常用文件管理命令**

1. `Ctrl c`：取消命令，并且换行
2. `Ctrl u`：清空本行命令
3. `tab键`：可以补全命令和文件名，如果补全不了快速按两下`tab`键，可以显示备选选项

   - `ls`：列出当前目录下所有文件，蓝色的是文件夹，白色的是普通文件，绿色的是可执行文件
   - `ls -a`：查看所有文件包括隐藏文件（以`.`开头的文件就是隐藏文件）
   - `ls -l`：查看当前路径下文件的读、写、执行权限
   - `ls -h`：展示出文件的大小，如k、M、G
     - `-h`必须和`-l`同时使用，否则不起作用
     - 不使用`-h`，文件大小单位为byte
   - `ls | wc -l`：查看`ls`下有多少个文件
4. `pwd`：显示当前路径
5. `cd XXX`：进入`XXX`目录下，`cd ..`返回上层目录

   - `.`：当前目录 `..`：上级目录

   - `~`：家目录，回回到路径`/home/acs`下

   - `cd -`：返回改变路径前的路径，比如当前在`/home/acs/homework`然后`cd** **/`这个时候就处于`/`目录下，然后`cd -`就会回到改变路径前的路径也就是`/home/acs/homework`
6. `cp XXX YYY`：将`XXX`文件复制成`YYY`，`XXX`和`YYY`可以是同一个路径，比如`../dir_c/a.txt`，表示上层目录下的`dir_c`文件夹下的文件`a.txt`

   - `cp XXX YYY -r`将`XXX`目录（文件夹）复制到`YYY`下

   - 非当前路径重命名方法：`cp a.txt ../b.txt`


7. `mkdir XXX`：创建目录（文件夹）`XXX`

   - `mkdir -p`：`-p`：如果文件夹不存在，则创建

8. `rm XXX`：删除普通文件； `rm XXX -r`：删除文件夹

   - 支持正则表达式，删除所有`.txt`类型文件：`rm *.txt`


      - 删除所有文件（不包括文件夹）：`rm *`


      - 正则表达式删除所有文件夹：`rm * -r`即可
      - 强力删除（root权限）：`rm -f`


9. `mv XXX YYY`：将`XXX`文件移动到`YYY`下，和`cp`命令一样，`XXX`和`YYY`可以是同一个路径；重命名也是用这个命令
   - 非当前路径移动方法：`mv a.txt ../b.txt`


10. `touch XXX`：创建一个文件
11. `cat XXX`：展示文件`XXX`中的内容
    - 复制文本：`windows/Linux`下：`Ctrl + insert`
    - 粘贴文本：`windows/Linux`下：`Shift + insert`


12. `history`：查看历史输入指令

13. `tree`：以树形显示文件目录结构

```bash
tree  -a -d -L 1 
#tree:显示目录树 
# -a：显示所有文件目录
# -d:只显示目录 
# -L:选择显示的目录深度 
# 1：只显示一层深度，即不递归子目录
```

14. `which`：查找程序文件（查看命令程序文件在哪）
    - `which cp`输出`usr/bin/cp`
15. `find`：查找文件(按文件名字、按文件大小)
    - `find 起始路径 -name “被查找文件名”`
    - `find 起始路径 -size -10k`：查找小于10kb的文件



## 服务器实用命令

1. 查看ubuntu版本

   ```bash
   lsb_release -a 
   ```

2. 查看文件占用多大内存

    `du -h [filepath]` 直接得出人好识别的文件大小

    `du [目标目录] -sh`得到目录下总占用内存

    ![image-20240119184407827](./.assets/image-20240119184407827.png)

## vim

**vim 教程**
**功能：**

1. 命令行模式下的文本编辑器

2. 根据文件扩展名自动判别编程语言。支持代码缩进、代码高亮等功能

3. 使用方式：`vim filename`

   - ㅤ如果已有该文件，则打开它


      - ㅤ如果没有该文件，则打开一个新的文件，并命名为`filename`

**模式：**

1. 一般命令模式

   默认模式。命令输入方式：类似于打游戏放技能，按不同字符，即可进行不同操作。可以复制、粘贴、删除文本等

2. 编辑模式

   在一般命令模式下按`i`，会进入编辑模式按下`ESC`会退出编辑模式，返回到一般命令模式

3. 命令行模式

   在一般命令模式里按下`: / ?`三个字母中的任意一个，会进入命令行模式命令行在最下面。可以查找、替换、保存、退出、配置编辑器等

**操作：**

1. `i`：进入编辑模式
2. `ESC`：进入一般命令模式
3. `←`：光标向左移动一个字符
4. `↓`：光标向下移动一个字符
5. `↑`：光标向上移动一个字符
6. `→`：光标向右移动一个字符
7. `n<Space>`：`n`表示数字，按下数字后再按空格，光标会往右移动这一行的`n`个字符
8. `n<Enter>`：`n`为数字，光标向下移动`n`行
9. `:n`：`n`为数字，光标移动到第`n`行
10. `Home`：光标移动到本行开头
11. `End`：光标移动到本行末尾
12. `G`：光标移动到最后一行
13. `gg`：光标移动到第一行，相当于`1G`
    - **删除组合键：`gg + d + G gg + d + nG`**

14. `/word`：向光标之下寻找第一个值为`word`的字符串ㅤㅤ
15. `?word`：向光标之上寻找第一个值为`word`的字符串

16. `n`：重复前一个查找操作

17. `N`：反向重复前一个查找操作ㅤ
18. `:n1， n2s/word1/word2/g`：`n1`与`n2`为数字，在`n1`行与`n2`行之间寻找`word1`这个字符串，并将该字符串替换为`word2`
19. `:1， $s/word1/word2/g`：将全文的`word1`替换为`word2`

20. `:1， $s/word1/word2/gc`：将全文的`word1`替换为`word2`，且在替换前要求用户确认

21. **`v`：选中文本，按两下`ESC`取消选中状态**

22. **`d`：删除选中的文本**

23. **`dd`：删除当前行**

24. **`y`：复制选中的文本**

25. **`yy`：复制当前行**

26. **`p`：将复制的数据在光标的下一行(`yy`)/下一个位置(`y`)粘贴**

27. **`u`：撤销**

28. **`Ctrl + r`：取消撤销**

29. `Shift + >`：将选中的文本整体向右缩进一次 n Shift + > 向右缩进n次

30. `Shift + <`：将选中的文本整体向左缩进一次 n Shift + < 向左缩进n次

31. `:w`：保存

32. `:w!`：强制保存

33. `:q`：退出

34. `:q!`：强制退出

35. `:wq`：保存并退出

36. **`:set paste`：设置成粘贴模式，取消代码自动缩进**

37. **`set nopaste`：取消粘贴模式，开启代码自动缩进**

38. **`set nu`：显示行号**

39. **`set nonu`：隐藏行号**

40. `gg=G`：将全文代码格式化

41. `:noh`：关闭查找关键词高亮

42. `Ctrl + q`：当vim卡死时，可以取消当前正在执行的命令

43. 异常处理：

    - 每次用`vim`编辑文件时，会自动创建一个`.filename.swp`的临时文件


    - 如果打开某个文件时，该文件的`swp`文件已存在，则会报错。此时解决办法有两种：


​				找到正在打开该文件的程序，并退出；

​				直接删除掉该`swp`文件即可；

## tmux

**tmux 教程**
**功能**：

1. 分屏

2. 允许断开`Terminal`连接后，继续运行进程

**结构**：

一个`tmux`可以包含多个`session`，一个`session`可以包含多个`window`，一个`window`可以包含多个`pane`

```shell
tmux：
    session 0：
        window 0：
            pane 0
            pane 1
            pane 2
            ...
        window 1
        window 2
        ...
    session 1
    session 2
    ...
```

**操作**：

1. **`tmux`：新建一个`session`，其中包含一个`window`，`window`中包含一个`pane`，`pane`里打开了一个`shell`对话框**

2. **按下`Ctrl + a`后手指松开，然后按`%`：将当前`pane`左右平分成两个`pane`**

3. **按下`Ctrl + a`后手指松开，然后按`"`：将当前`pane`上下平分成两个`pane`**

4. **`Ctrl + d`：关闭当前`pane`;如果当前`window`的所有`pane`均已关闭，则自动关闭`window`如果当前`session`的所有`window`均已关闭，则自动关闭`session`**

5. 鼠标点击可以选择`pane`

6. 按下`Ctrl + a`后手指松开，然后按方向键：选择相邻的`pane`

7. 鼠标拖动`pane`之间的分割线，可以调整分割线的位置

8. 按下`Ctrl + a`的同时按方向键，可以调整pane之间分割线的位置

9. 按下`Ctrl + a`后手指松开，然后按`z`：将当前pane全屏/取消全屏

10. **按下`Ctrl + a`后手指松开，然后按`d`：挂起当前`session`**

11. **`tmux a`：打开之前挂起的`session`**

12. 按下`Ctrl + a`后手指松开，然后按s：选择其它session

    - 方向键 ——上：选择上一项 `session/window/pane`

    - 方向键 ——下：选择下一项 `session/window/pane`

    - 方向键 ——左：展开当前项 `session/window`

    - 方向键 ——右：闭合当前项 `session/window`


13. 按下`Ctrl + a`后手指松开，然后按`c`：在当前`session`中创建一个新的`window`

14. 按下`Ctrl + a`后手指松开，然后按`w`：选择其它`window`，操作方法与(12)一致

15. 按下`Ctrl + a`后手指松开，然后按`Page Up`：翻阅当前`pane`内的内容

16. 鼠标滚轮：翻阅当前`pane`内的内容

17. **在`tmux`中选中文本时，需要按住`Shift`键**

18. **`tmux`中复制/粘贴文本的通用方式：**
    - **复制：`ctrl + insert`**

-  **粘贴：`shift + insert`**





## ssh scp

### ssh

#### 基本用法

**远程登录服务器：**

```bash
ssh user@hostname
```

- `user`：用户名
- `hostname`：IP地址或域名

**第一次登录时会提示：**

```bash
The authenticity of host '123.57.47.211 (123.57.47.211)' can't be established.
ECDSA key fingerprint is SHA256:iy237yysfCe013/l+kpDGfEG9xxHxm0dnxnAbJTPpG8.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

输入`yes`，然后`回车`即可

这样会将该服务器的信息记录在`~/.ssh/known_hosts`文件中

然后输入密码即可登录到远程服务器中

**退出当前服务器：**

```bash
logout
```

**默认登录端口号为22，如果想登录某一特定端口：**

```bash
ssh user@hostname -p 22
```

#### 配置文件

**创建文件：**

```bash
~/.ssh/config
```

**在文件中输入：**

```shell
Host myserver1
    HostName IP地址或域名
    User 用户名

Host myserver2
    HostName IP地址或域名
    User 用户名
```

之后再使用服务器时，可以直接使用别名`myserver1、myserver2`

#### 免密登录

**创建密钥：**

```bash
ssh-keygen
```

然后**一直回车**即可

**执行结束后，~/.ssh/目录下会多两个文件：**

- `id_rsa`：私钥
- `id_rsa.pub`：公钥

ㅤㅤ==之后想免密码登录哪个服务器，就将公钥传给哪个服务器即可==

ㅤㅤ例如，想免密登录`myserver`服务器。则将公钥中的内容，复制到`myserver`中的`~/.ssh/authorized_keys`文件里即可。

ㅤㅤ也可以使用如下命令**一键添加公钥**：

```
ssh-copy-id myserver
```



### scp

#### 基本用法

![image-20231022143201946](./.assets/image-20231022143201946.png)

**`scp`传文件时，服务器后对应的路径是 `/`(最底层的路径)** 

**`/home/acs/xxxx`**

**`/root/xxx`**

**命令格式：**

将`source`路径下的文件复制到`destination`中

```bash
scp source destination
```

**一次复制多个文件：**

```bash
scp source1 source2 destination
```

**复制文件夹：**ㅤㅤ

将本地`家目录`中的`tmp`文件夹复制到`myserver`服务器中的`/home/acs/`目录下

```bash
scp -r ~/tmp myserver:/home/acs/
```

**mysever服务器没有root权限**

将`myserver`服务器中的`~/homework/`文件夹复制到本地的当前路径下

```bash
scp -r myserver:/home/acs/homework .
```

指定服务器的端口号：

```bash
scp -P 22 source1 source2 destination
```

注意： `scp`的`-r -P`等参数尽量加在`source`和`destination`之前。

**使用scp配置其他服务器的vim和tmux**

```bash
scp ~/.vimrc ~/.tmux.conf ~/.bashrc myserver:/home/acs/
```

​        源                                                                目的

```shell
scp -r -P 10002 root@192.168.43.101:~/datasets/nuscenes ./
```



## cuda相关

### 服务器修改cuda版本

https://zhuanlan.zhihu.com/p/581634820#:~:text=%E5%88%9B%E5%BB%BA%E5%A4%9A%E4%B8%AAcuda%E7%89%88%E6%9C%AC%EF%BC%8C%E5%8F%AF%E4%BB%A5%E8%87%AA%E7%94%B1%E5%88%87%E6%8D%A2%EF%BC%8C%E4%B8%8D%E5%B9%B2%E6%89%B0%E6%BA%90%E7%8E%AF%E5%A2%83%EF%BC%8C%E4%B8%94%E4%B8%8D%E7%94%A8sudo%E6%8C%87%E4%BB%A4%2C%E6%93%8D%E4%BD%9C%E7%AE%80%E5%8D%95%201%201.%E5%88%9D%E5%A7%8B%E7%8E%AF%E5%A2%83%E5%BB%BA%E7%AB%8B%20%E9%A6%96%E5%85%88%E6%BA%90base%E7%89%88%E6%9C%AC%E4%B8%BA11.6%EF%BC%8C%E7%9B%AE%E6%A0%87%E5%9C%A8conda%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83%E4%B8%AD%E4%B8%8B%E8%BD%BD%E4%B8%80%E4%B8%AA11.3%E7%9A%84cuda%E7%89%88%E6%9C%AC%20%E5%8F%AF%E4%BB%A5%E4%BD%BF%E7%94%A8%E4%BB%A5%E4%B8%8B%E6%8C%87%E4%BB%A4%E6%9F%A5%E7%9C%8B%20nvcc%20-V%20%E9%A6%96%E5%85%88%E6%88%91%E4%BB%AC%E9%9C%80%E8%A6%81%E5%88%9B%E5%BB%BA%E7%8E%AF%E5%A2%83%2C%E6%88%91%E7%9A%84%E5%90%8D%E5%AD%97%E4%B8%BAnerfies_shi%EF%BC%8C%E6%8C%89%E7%85%A7%E8%87%AA%E5%B7%B1%E7%9A%84%E6%94%B9,%E7%BB%99cuda%E6%9D%83%E9%99%90%20chmod%20%2Bx%20cuda_11.3.0_465.19.01_linux.run%20%E8%BF%90%E8%A1%8Crun%E6%96%87%E4%BB%B6%20sh%20cuda_11.3.0_465.19.01_linux.run%20

https://blog.csdn.net/qq_44961869/article/details/115954258

https://blog.csdn.net/zong596568821xp/article/details/80880204

https://blog.csdn.net/qq_35082030/article/details/110387800



最好的教程

https://developer.nvidia.com/cuda-11.1.1-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=runfilelocal官网

https://blog.csdn.net/qq_35082030/article/details/110387800



https://developer.nvidia.com/cuda-toolkit-archive  全部版本

![](./.assets/image-20240109113322330.png)

mkdir -p  ~/../usr/local/cuda-11.1

sudo sh cuda_11.1.1_455.32.00_linux.run



![](./.assets/image-20240109114302439.png)

![](./.assets/image-20240109114425109.png)

![](./.assets/image-20240109114437604.png)

![](./.assets/image-20240109114445515.png)

![](./.assets/image-20240109114533796.png)



修改bashrc

![image-20240109114832909](./.assets/image-20240109114832909.png)

source .bashrc

sudo /root/miniconda3/envs/monocon/bin/python3.6 setup.py develop

### 服务器的cuda切换

1. 利用如下命令打开环境变量参数

   ```bash
   sudo gedit ~/.bashrc
   ```

2. 写入模板

   ```bash
   export PATH=/usr/local/cuda-版本/bin:$PATH  
   export LD_LIBRARY_PATH=/usr/local/cuda-版本/lib64:$LD_LIBRARY_PATH
   export CUDA_HOME=/usr/local/cuda
   ```

3. 实例

   ```bash
   export PATH=/usr/local/cuda-11.6/bin:$PATH  
   export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
   export CUDA_HOME=/usr/local/cuda
   ```

## Docker

**docker**

https://docs.docker.com/engine/install/linux-postinstall/

### docker用户组

将当前用户添加到`docker`用户组

为了避免每次使用`docker`命令都需要加上`sudo`权限，可以将当前用户加入安装中自动创建的`docker`用户组

```bash
sudo usermod -aG docker $USER
```

### 镜像（images）

1. `docker pull ubuntu:20.04`：拉取一个镜像（名称:版本号）
2. `docker images`：列出本地所有镜像
3. `docker image rm ubuntu:20.04 或 docker rmi ubuntu:20.04`：删除镜像`ubuntu:20.04`
4. `docker [container] commit CONTAINER IMAGE_NAME:TAG`：创建某个`container`的镜像
5. `docker save -o ubuntu_20_04.tar ubuntu:20.04`：将镜像`ubuntu:20.04`导出到本地文件`ubuntu_20_04.tar`中 **还要加上可读权限 chmod + r ubuntu_20_04.tar**
6. `docker load -i ubuntu_20_04.tar`：将镜像`ubuntu:20.04`从本地文件`ubuntu_20_04.tar`中加载出来

### 容器（container）

1. `docker [container] create -it ubuntu:20.04`：利用镜像`ubuntu:20.04`创建一个容器。

2. `docker ps -a`：查看本地的所有容器状态

3. `docker [container] start CONTAINER`：启动容器

4. `docker [container] stop CONTAINER`：停止容器

5. `docker [container] restart CONTAINER`：重启容器

6. `docker [contaienr] run -itd ubuntu:20.04`：创建并启动一个容器

7. `docker [container] attach CONTAINER`：进入容器

8. **先按`Ctrl + p`，再按`Ctrl + q`可以挂起容器**

9. **`Ctrl + d`直接关掉容器**

10. `docker [container] exec CONTAINER COMMAND`：在容器中执行命令

11. `docker [container] rm CONTAINER`：删除容器

12. `docker container prune`：删除所有已停止的容器

13. `docker export -o xxx.tar CONTAINER`：将容器`CONTAINER`导出到本地文件`xxx.tar`中

14. `docker import xxx.tar image_name:tag`：将本地文件`xxx.tar`导入成镜像，并将镜像命名为`image_name:tag`

15. `docker export/import`与`docker save/load`的区别：

    `export/import`会丢弃历史记录和元数据信息，仅保存容器当时的快照状态

    `save/load`：会保存完整记录，体积更大

16. `docker top CONTAINER`：查看某个容器内的所有进程

17. `docker stats`：查看所有容器的统计信息，包括CPU、内存、存储、网络等信息

18. `docker cp xxx CONTAINER`:`xxx` 或 `docker cp CONTAINER:xxx xxx`：在本地和容器间复制文件

19. `docker rename CONTAINER1 CONTAINER2`：重命名容器

20. `docker update CONTAINER --memory 500MB`：修改容器限制

### 实战

进入AC Terminal，然后：

```bash
# 将镜像上传到自己租的云端服务器
scp /var/lib/acwing/docker/images/docker_lesson_1_0.tar server_name: 
# 登录自己的云端服务器
ssh server_name 
# 将镜像加载到本地
docker load -i docker_lesson_1_0.tar  
# 创建并运行docker_lesson:1.0镜像
docker run -p 20000:22 --name my_docker_server -itd docker_lesson:1.0  
# 进入创建的docker容器
docker attach my_docker_server 
# 设置root密码
passwd  
```

**去云平台控制台中修改安全组配置，放行端口20000。**

返回AC Terminal，即可通过ssh登录自己的docker容器：

`ssh root@xxx.xxx.xxx.xxx -p 20000`  # 将xxx.xxx.xxx.xxx替换成自己租的服务器的IP地址
然后，可以仿照上节课内容，创建工作账户acs。**这样root改为acs**

最后，可以参考ssh——ssh登录配置docker容器的别名和免密登录。

**创建用户**

`adduser acs`

**分配sudo权限**

`usermod -aG sudo acs`

**免密登录**

在本地服务器的 `~/.ssh/config`文件中添加docker的配置信息

```bash
Host myserver1
    HostName IP地址或域名
    User 用户名

Host myserver2_docker
    HostName IP地址(跟服务器的ip地址一样)
    User 用户名
    Port 20000
```

一键添加公钥

```bash
ssh-copy-id myserver2_docker
```



## 下载相关

### 下载文件

```
wget -c -o [保存的名字] “[下载链接]”

wget -c -o yolov5s.pt "https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt"
```

[wget命令使用及参数详解_wget -o和-o区别-CSDN博客](https://blog.csdn.net/u011598193/article/details/99412491)



### 服务器下载miniconda

[Miniconda — miniconda documentation](https://docs.conda.io/projects/miniconda/en/latest/)

```bash
# 在root路径下执行操作
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
# 初始化.bashrc
~/miniconda3/bin/conda init bash
```





## 服务器配置

### 配置tmux和vim

#### vim配置

- 安装vim

  sudo apt install vim

- 新建一个 .vmrc

- 复制保存


```shell
" An example for a vimrc file.
"
" To use it, copy it to
"     for Unix and OS/2:  ~/.vimrc
"             for Amiga:  s:.vimrc
"  for MS-DOS and Win32:  $VIM\_vimrc
"           for OpenVMS:  sys$login:.vimrc

" When started as "evim", evim.vim will already have done these settings.
if v:progname =~? "evim"
  finish
endif

" Use Vim settings, rather then Vi settings (much better!).
" This must be first, because it changes other options as a side effect.
set nocompatible

" allow backspacing over everything in insert mode
set backspace=indent,eol,start

if has("vms")
  set nobackup          " do not keep a backup file, use versions instead
else
  set backup            " keep a backup file
endif
set history=50          " keep 50 lines of command line history
set ruler               " show the cursor position all the time
set showcmd             " display incomplete commands
set incsearch           " do incremental searching
"==========================================================================
"My Setting-sunshanlu
"==========================================================================
vmap <leader>y :w! /tmp/vitmp<CR>
nmap <leader>p :r! cat /tmp/vitmp<CR>

"语法高亮
syntax enable
syntax on
"显示行号
set nu

"修改默认注释颜色
"hi Comment ctermfg=DarkCyan
"允许退格键删除
"set backspace=2
"启用鼠标
set mouse=a
set selection=exclusive
set selectmode=mouse,key
"按C语言格式缩进
set cindent
set autoindent
set smartindent
set shiftwidth=4

" 允许在有未保存的修改时切换缓冲区
"set hidden

" 设置无备份文件
set writebackup
set nobackup

"显示括号匹配
set showmatch
"括号匹配显示时间为1(单位是十分之一秒)
set matchtime=5
"显示当前的行号列号：
set ruler
"在状态栏显示正在输入的命令
set showcmd

set foldmethod=syntax
"默认情况下不折叠
set foldlevel=100
" 开启状态栏信息
set laststatus=2
" 命令行的高度，默认为1，这里设为2
set cmdheight=2


" 显示Tab符，使用一高亮竖线代替
set list
"set listchars=tab:\|\ ,
set listchars=tab:>-,trail:-


"侦测文件类型
filetype on
"载入文件类型插件
filetype plugin on
"为特定文件类型载入相关缩进文件
filetype indent on
" 启用自动补全
filetype plugin indent on 


"设置编码自动识别, 中文引号显示
filetype on "打开文件类型检测
"set fileencodings=euc-cn,ucs-bom,utf-8,cp936,gb2312,gb18030,gbk,big5,euc-jp,euc-kr,latin1
set fileencodings=utf-8,gb2312,gbk,gb18030
"这个用能很给劲，不管encoding是什么编码，都能将文本显示汉字
"set termencoding=gb2312
set termencoding=utf-8
"新建文件使用的编码
set fileencoding=utf-8
"set fileencoding=gb2312
"用于显示的编码，仅仅是显示
set encoding=utf-8
"set encoding=utf-8
"set encoding=euc-cn
"set encoding=gbk
"set encoding=gb2312
"set ambiwidth=double
set fileformat=unix


"设置高亮搜索
set hlsearch
"在搜索时，输入的词句的逐字符高亮
set incsearch

" 着色模式
set t_Co=256
"colorscheme wombat256mod
"colorscheme gardener
"colorscheme elflord
colorscheme desert
"colorscheme evening
"colorscheme darkblue
"colorscheme torte
"colorscheme default

" 字体 && 字号
set guifont=Monaco:h10
"set guifont=Consolas:h10

" :LoadTemplate       根据文件后缀自动加载模板
"let g:template_path='/home/ruchee/.vim/template/'

" :AuthorInfoDetect   自动添加作者、时间等信息，本质是NERD_commenter && authorinfo的结合
""let g:vimrc_author='sunshanlu'
""let g:vimrc_email='sunshanlu@baidu.com'
""let g:vimrc_homepage='http://www.sunshanlu.com'
"
"
" Ctrl + E            一步加载语法模板和作者、时间信息
""map <c-e> <ESC>:AuthorInfoDetect<CR><ESC>Gi
""imap <c-e> <ESC>:AuthorInfoDetect<CR><ESC>Gi
""vmap <c-e> <ESC>:AuthorInfoDetect<CR><ESC>Gi



" ======= 引号 && 括号自动匹配 ======= "
"
":inoremap ( ()<ESC>i

":inoremap ) <c-r>=ClosePair(')')<CR>
"
":inoremap { {}<ESC>i
"
":inoremap } <c-r>=ClosePair('}')<CR>
"
":inoremap [ []<ESC>i
"
":inoremap ] <c-r>=ClosePair(']')<CR>
"
":inoremap < <><ESC>i
"
":inoremap > <c-r>=ClosePair('>')<CR>
"
"":inoremap " ""<ESC>i
"
":inoremap ' ''<ESC>i
"
":inoremap ` ``<ESC>i
"
":inoremap * **<ESC>i

" 每行超过80个的字符用下划线标示
""au BufRead,BufNewFile *.s,*.asm,*.h,*.c,*.cpp,*.java,*.cs,*.lisp,*.el,*.erl,*.tex,*.sh,*.lua,*.pl,*.php,*.tpl,*.py,*.rb,*.erb,*.vim,*.js,*.jade,*.coffee,*.css,*.xml,*.html,*.shtml,*.xhtml Underlined /.\%81v/
"
"
" For Win32 GUI: remove 't' flag from 'guioptions': no tearoff menu entries
" let &guioptions = substitute(&guioptions, "t", "", "g")

" Don't use Ex mode, use Q for formatting
map Q gq

" This is an alternative that also works in block mode, but the deleted
" text is lost and it only works for putting the current register.
"vnoremap p "_dp

" Switch syntax highlighting on, when the terminal has colors
" Also switch on highlighting the last used search pattern.
if &t_Co > 2 || has("gui_running")
  syntax on
  set hlsearch
endif

" Only do this part when compiled with support for autocommands.
if has("autocmd")

  " Enable file type detection.
  " Use the default filetype settings, so that mail gets 'tw' set to 72,
  " 'cindent' is on in C files, etc.
  " Also load indent files, to automatically do language-dependent indenting.
  filetype plugin indent on

  " Put these in an autocmd group, so that we can delete them easily.
  augroup vimrcEx
  au!

  " For all text files set 'textwidth' to 80 characters.
  autocmd FileType text setlocal textwidth=80

  " When editing a file, always jump to the last known cursor position.
  " Don't do it when the position is invalid or when inside an event handler
  " (happens when dropping a file on gvim).
  autocmd BufReadPost *
    \ if line("'\"") > 0 && line("'\"") <= line("$") |
    \   exe "normal g`\"" |
    \ endif

  augroup END

else

  set autoindent                " always set autoindenting on

endif " has("autocmd")

" 增加鼠标行高亮
set cursorline
hi CursorLine  cterm=NONE   ctermbg=darkred ctermfg=white

" 设置tab是四个空格
set ts=4
set expandtab

" 主要给Tlist使用
let Tlist_Exit_OnlyWindow = 1
let Tlist_Auto_Open = 1

```



#### tmux配置

- 安装tmux

```bash
sudo apt-get install tmux
```

- 不要着急进tmux
- 新建一个 .tmux.conf


```shell
set-option -g status-keys vi
setw -g mode-keys vi

setw -g monitor-activity on

# setw -g c0-change-trigger 10
# setw -g c0-change-interval 100

# setw -g c0-change-interval 50
# setw -g c0-change-trigger  75


set-window-option -g automatic-rename on
set-option -g set-titles on
set -g history-limit 100000

#set-window-option -g utf8 on

# set command prefix
set-option -g prefix C-a
unbind-key C-b
bind-key C-a send-prefix

bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

bind < resize-pane -L 7
bind > resize-pane -R 7
bind - resize-pane -D 7
bind + resize-pane -U 7


bind-key -n M-l next-window
bind-key -n M-h previous-window



set -g status-interval 1
# status bar
set -g status-bg black
set -g status-fg blue


#set -g status-utf8 on
set -g status-justify centre
set -g status-bg default
set -g status-left " #[fg=green]#S@#H #[default]"
set -g status-left-length 20


# mouse support
# for tmux 2.1
# set -g mouse-utf8 on
set -g mouse on
#
# for previous version
#set -g mode-mouse on
#set -g mouse-resize-pane on
#set -g mouse-select-pane on
#set -g mouse-select-window on


#set -g status-right-length 25
set -g status-right "#[fg=green]%H:%M:%S #[fg=magenta]%a %m-%d #[default]"

# fix for tmux 1.9
bind '"' split-window -vc "#{pane_current_path}"
bind '%' split-window -hc "#{pane_current_path}"
bind 'c' new-window -c "#{pane_current_path}"

# run-shell "powerline-daemon -q"

# vim: ft=conf

```



### 服务器换源

1. anaconda换源

   https://blog.csdn.net/moshiyaofei/article/details/122058922

2. pip换源

   https://blog.csdn.net/limengshi138392/article/details/111315014



### PYTHONPATH

修改.bashrc文件

:是分隔符，可以填入多个PYTHONPAYTH

```shell
export PYTHONPATH=/root/tianyao/BEVFormer:$PYTHONPATH
```

也可以临时导入

```shell
export PYTHONPATH=/root/tianyao/BEVFormer
```

### 服务器配置vpn

### 一键配置conda openmmlab环境

基于服务器cuda 11.1 -python 3.8

```shell
conda create -n project python=3.8
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html
pip install mmdet==2.24.0
pip install mmsegmentation==0.24.0
pip install numba==0.53.0 numpy==1.23.4 nuscenes-devkit==1.1.11 pandas==2.0.3 scikit-image===0.19.3 scipy==1.10.1 setuptools==59.5.0 tensorboardX
```



## git相关

### git命令

#### 全局设置

1. `git config --global user.name xxx`：设置全局用户名，信息记录在`~/.gitconfig`文件中
2. `git config --global user.email xxx@xxx.com`：设置全局邮箱地址，信息记录在`~/.gitconfig`文件中
3. **`git init`：将当前目录配置成git仓库，信息记录在隐藏的`.git`文件夹中**

#### 常用命令

1. **`git add XX `：将XX文件添加到暂存区**
2. **`git commit -m "给自己看的备注信息"`：将暂存区的内容提交到当前分支**
3. **`git status`：查看仓库状态**
4. **`git log`：查看当前分支的所有版本**
5. **`git push -u (第一次需要-u以后不需要)` ：将当前分支推送到远程仓库**
6. **`git clone git@git.acwing.com:xxx/XXX.git`：将远程仓库XXX下载到当前目录下**
7. **`git branch`：查看所有分支和当前所处分支**

#### 查看命令

1. `git diff XX`：查看XX文件相对于暂存区修改了哪些内容
2. `git status`：查看仓库状态
3. `git log`：查看当前分支的所有版本
4. `git log --pretty=oneline`：用一行来显示
5. `git reflog`：查看HEAD指针的移动历史（包括被回滚的版本）
6. `git branch`：查看所有分支和当前所处分支
7. `git pull` ：将远程仓库的当前分支与本地仓库的当前分支合并

#### 删除命令

1. `git rm --cached XX`：将文件从仓库索引目录中删掉，不希望管理这个文件
2. **`git restore --staged xx`：==将xx从暂存区里移除==**
3. **`git checkout — XX或git restore XX`：==将XX文件尚未加入暂存区的修改全部撤销==**

#### 代码回滚

1. `git reset --hard HEAD^ `或`git reset --hard HEAD~` ：将代码库回滚到上一个版本
2. `git reset --hard HEAD^^`：往上回滚两次，以此类推
3. `git reset --hard HEAD~100`：往上回滚100个版本
4. **`git reset --hard` 版本号：回滚到某一特定版本**

#### 远程仓库

==**使用远程仓库要先将当前服务器的公钥添加到仓库的ssh密钥设置中**==

1. **`git remote add origin git@git.acwing.com:xxx/XXX.git`：将本地仓库关联到远程仓库**
2. ` git push --set-upstream origin master`
3. `git push -u (第一次需要-u以后不需要) `：将当前分支推送到远程仓库
4. `git push origin branch_name`：将本地的某个分支推送到远程仓库
5. **`git clone git@git.acwing.com:xxx/XXX.git`：将远程仓库XXX下载到当前目录下**
6. `git push --set-upstream o`
7. `rigin branch_name`：设置本地的branch_name分支对应远程仓库的branch_name分支
8. `git push -d origin branch_name`：删除远程仓库的branch_name分支
9. `git checkout -t origin/branch_name` ：将远程的branch_name分支拉取到本地
10. `git pull` ：将远程仓库的当前分支与本地仓库的当前分支合并
11. `git pull origin branch_name`：将远程仓库的branch_name分支与本地仓库的当前分支合并
12. `git branch --set-upstream-to=origin/branch_name1 branch_name2`：将远程的branch_name1分支与本地的branch_name2分支对应

#### 分支命令

1. `git branch branch_name`：创建新分支
2. `git branch`：查看所有分支和当前所处分支
3. `git checkout -b branch_name`：创建并切换到branch_name这个分支
4. `git checkout branch_name`：切换到branch_name这个分支
5. `git merge branch_name`：将分支branch_name合并到当前分支上
6. `git branch -d branch_name`：删除本地仓库的branch_name分支
7. `git push --set-upstream origin branch_name`：设置本地的branch_name分支对应远程仓库的branch_name分支
8. `git push -d origin branch_name`：删除远程仓库的branch_name分支
9. `git checkout -t origin/branch_name `：将远程的branch_name分支拉取到本地
10. `git pull `：将远程仓库的当前分支与本地仓库的当前分支合并
11. `git pull origin branch_name`：将远程仓库的branch_name分支与本地仓库的当前分支合并
12. `git branch --set-upstream-to=origin/branch_name1 branch_name2`：将远程的branch_name1分支与本地的branch_name2分支对应

#### stash暂存

1. `git stash`：将工作区和暂存区中尚未提交的修改存入栈中
2. `git stash apply`：将栈顶存储的修改恢复到当前分支，但不删除栈顶元素
3. `git stash drop`：删除栈顶存储的修改
4. `git stash pop`：将栈顶存储的修改恢复到当前分支，同时删除栈顶元素
5. `git stash list`：查看栈中所有元素



### git clone历史版本代码

1. clone最新版本

   ![image-20231207120903195](./.assets/image-20231207120903195.png)

2. 查看历史版本id号

   ![image-20231207120927591](./.assets/image-20231207120927591.png)

3. `git checkout id`

   切换到当前分支

**第二种方法**

```bash
git clone https://mirror.ghproxy.com/https://github.com/ultralytics/yolov5.git -b v5.0
```

**第三种方法**

```bash
git clone https://mirror.ghproxy.com/https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v5.0
```



### git上传大文件 >100MB
### git设置clash代理

参考知乎

[解决 Github port 443 : Timed out - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/636418854)

```bash
# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy

# 查看代理
git config --global --get http.proxy
git config --global --get https.proxy
```



因为用clash，设置clash代理

```bash
git config --global http.proxy 127.0.0.1:7890
git config --global https.proxy 127.0.0.1:7890
```

速度飞起！



## 各种数据集下载

### nuscenes数据集下载



1. 登录官网[nuScenes](https://www.nuscenes.org/nuscenes)

![](./.assets/image-20231129133153988.png)



2. 从官网获取下载链接，写入脚本`nuscenes.sh`

   （这个链接几天更新一次，所以不能经常用，得更换）

   **注意：vim脚本向下打几个回车，不然会吞掉第一个wget**

```shell
wget -c -O v1.0-trainval_meta.tgz "https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-trainval_meta.tgz?AWSAccessKeyId=ASIA6RIK4RRMEZVQSHGH&Signature=kokRnDkRz1jy3iq7oixPlq%2FePoI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHYaCXVzLWVhc3QtMSJIMEYCIQDYSFf669tSVA078drpV90pqDQedTmrUsqjsh093rFzpwIhALWN8tGIAhbT%2FWS7M%2FDwvTt5Od4UVQ066nXCPK1pNSWEKv0CCM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMOTk5MTM5NjA5Njg4IgxG3iMxsFnLZwkuD7Mq0QKIkr5ksHhDDWPsU1LSNON3Mgf0utli73g4VhnF7EWs6uRxhVEWmm2xUcJ2oUdnTDWy6jW3IGwZx4WKKZf1cQ1sfDNSwZ5PcOAwnKqRDXMfTW%2BuhZ0yuvNsPFZf8JHjn%2FUS9RMJGfvqr7IZAWpePPt2eyJ12JTxRuJOr9mnCLQJ2d8AUMaftEEXkQPcISJ9q53W3YHp3l835ioaDZCg966zrpfoL59RecVVMT1Id4rB%2BdN%2BXCFWd7iUBvtskdTzae34hnUibgnKTyrJOL2aSyWrE2JTl2IQVzZaHYycwWdpmJkwuj%2Bx7LcMNect63h3KTV053z8rCDQ%2BReOnMd3i2KZnYg5C6aznqYgjQ9IzM%2B%2BDzua2TxuN7JnA53FoJ4wOR1iufncNGX5ZaDMDGTEoXwtgyFq5n7cEgqiPxRmjJJCRBuoRILX%2Bw4DQAxZSZXrsq%2BsMKuUm6sGOp0BCaB2M7Fm8chD%2BxybQocvGPWkql8j4l4Ek332SMRjve0YcYCFCx7HkkGdwSHTBmd3pV0jqyPlIsvctwcZirGlgzg4mTZD65h6uR7dYpq%2FscWzEO63f9mkvVm%2BX8P%2BQeNBz0a4BwV4wLjKbvn2LC9AC%2FTlm1YyWBit2qZ%2B717wNqsxURkfF2O%2FDpPGDNV71UxMLh8Uo2AZXxxLqiM7ew%3D%3D&Expires=1701668356"
wget -c -O v1.0-trainval01_blobs.tar "https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-trainval01_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMEZVQSHGH&Signature=FSUTV61285rBghAhtO231%2BfiMzI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHYaCXVzLWVhc3QtMSJIMEYCIQDYSFf669tSVA078drpV90pqDQedTmrUsqjsh093rFzpwIhALWN8tGIAhbT%2FWS7M%2FDwvTt5Od4UVQ066nXCPK1pNSWEKv0CCM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMOTk5MTM5NjA5Njg4IgxG3iMxsFnLZwkuD7Mq0QKIkr5ksHhDDWPsU1LSNON3Mgf0utli73g4VhnF7EWs6uRxhVEWmm2xUcJ2oUdnTDWy6jW3IGwZx4WKKZf1cQ1sfDNSwZ5PcOAwnKqRDXMfTW%2BuhZ0yuvNsPFZf8JHjn%2FUS9RMJGfvqr7IZAWpePPt2eyJ12JTxRuJOr9mnCLQJ2d8AUMaftEEXkQPcISJ9q53W3YHp3l835ioaDZCg966zrpfoL59RecVVMT1Id4rB%2BdN%2BXCFWd7iUBvtskdTzae34hnUibgnKTyrJOL2aSyWrE2JTl2IQVzZaHYycwWdpmJkwuj%2Bx7LcMNect63h3KTV053z8rCDQ%2BReOnMd3i2KZnYg5C6aznqYgjQ9IzM%2B%2BDzua2TxuN7JnA53FoJ4wOR1iufncNGX5ZaDMDGTEoXwtgyFq5n7cEgqiPxRmjJJCRBuoRILX%2Bw4DQAxZSZXrsq%2BsMKuUm6sGOp0BCaB2M7Fm8chD%2BxybQocvGPWkql8j4l4Ek332SMRjve0YcYCFCx7HkkGdwSHTBmd3pV0jqyPlIsvctwcZirGlgzg4mTZD65h6uR7dYpq%2FscWzEO63f9mkvVm%2BX8P%2BQeNBz0a4BwV4wLjKbvn2LC9AC%2FTlm1YyWBit2qZ%2B717wNqsxURkfF2O%2FDpPGDNV71UxMLh8Uo2AZXxxLqiM7ew%3D%3D&Expires=1701668375"
wget -c -O v1.0-trainval02_blobs.tar "https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-trainval02_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMEZVQSHGH&Signature=17Rz6BRm8BEqChr%2FuS361odUMz8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHYaCXVzLWVhc3QtMSJIMEYCIQDYSFf669tSVA078drpV90pqDQedTmrUsqjsh093rFzpwIhALWN8tGIAhbT%2FWS7M%2FDwvTt5Od4UVQ066nXCPK1pNSWEKv0CCM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMOTk5MTM5NjA5Njg4IgxG3iMxsFnLZwkuD7Mq0QKIkr5ksHhDDWPsU1LSNON3Mgf0utli73g4VhnF7EWs6uRxhVEWmm2xUcJ2oUdnTDWy6jW3IGwZx4WKKZf1cQ1sfDNSwZ5PcOAwnKqRDXMfTW%2BuhZ0yuvNsPFZf8JHjn%2FUS9RMJGfvqr7IZAWpePPt2eyJ12JTxRuJOr9mnCLQJ2d8AUMaftEEXkQPcISJ9q53W3YHp3l835ioaDZCg966zrpfoL59RecVVMT1Id4rB%2BdN%2BXCFWd7iUBvtskdTzae34hnUibgnKTyrJOL2aSyWrE2JTl2IQVzZaHYycwWdpmJkwuj%2Bx7LcMNect63h3KTV053z8rCDQ%2BReOnMd3i2KZnYg5C6aznqYgjQ9IzM%2B%2BDzua2TxuN7JnA53FoJ4wOR1iufncNGX5ZaDMDGTEoXwtgyFq5n7cEgqiPxRmjJJCRBuoRILX%2Bw4DQAxZSZXrsq%2BsMKuUm6sGOp0BCaB2M7Fm8chD%2BxybQocvGPWkql8j4l4Ek332SMRjve0YcYCFCx7HkkGdwSHTBmd3pV0jqyPlIsvctwcZirGlgzg4mTZD65h6uR7dYpq%2FscWzEO63f9mkvVm%2BX8P%2BQeNBz0a4BwV4wLjKbvn2LC9AC%2FTlm1YyWBit2qZ%2B717wNqsxURkfF2O%2FDpPGDNV71UxMLh8Uo2AZXxxLqiM7ew%3D%3D&Expires=1701668394"
wget -c -O v1.0-trainval03_blobs.tgz "https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-trainval03_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMEZVQSHGH&Signature=LlTK7oflOOEIbRc9g2oiZ%2Fv35zU%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHYaCXVzLWVhc3QtMSJIMEYCIQDYSFf669tSVA078drpV90pqDQedTmrUsqjsh093rFzpwIhALWN8tGIAhbT%2FWS7M%2FDwvTt5Od4UVQ066nXCPK1pNSWEKv0CCM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMOTk5MTM5NjA5Njg4IgxG3iMxsFnLZwkuD7Mq0QKIkr5ksHhDDWPsU1LSNON3Mgf0utli73g4VhnF7EWs6uRxhVEWmm2xUcJ2oUdnTDWy6jW3IGwZx4WKKZf1cQ1sfDNSwZ5PcOAwnKqRDXMfTW%2BuhZ0yuvNsPFZf8JHjn%2FUS9RMJGfvqr7IZAWpePPt2eyJ12JTxRuJOr9mnCLQJ2d8AUMaftEEXkQPcISJ9q53W3YHp3l835ioaDZCg966zrpfoL59RecVVMT1Id4rB%2BdN%2BXCFWd7iUBvtskdTzae34hnUibgnKTyrJOL2aSyWrE2JTl2IQVzZaHYycwWdpmJkwuj%2Bx7LcMNect63h3KTV053z8rCDQ%2BReOnMd3i2KZnYg5C6aznqYgjQ9IzM%2B%2BDzua2TxuN7JnA53FoJ4wOR1iufncNGX5ZaDMDGTEoXwtgyFq5n7cEgqiPxRmjJJCRBuoRILX%2Bw4DQAxZSZXrsq%2BsMKuUm6sGOp0BCaB2M7Fm8chD%2BxybQocvGPWkql8j4l4Ek332SMRjve0YcYCFCx7HkkGdwSHTBmd3pV0jqyPlIsvctwcZirGlgzg4mTZD65h6uR7dYpq%2FscWzEO63f9mkvVm%2BX8P%2BQeNBz0a4BwV4wLjKbvn2LC9AC%2FTlm1YyWBit2qZ%2B717wNqsxURkfF2O%2FDpPGDNV71UxMLh8Uo2AZXxxLqiM7ew%3D%3D&Expires=1701668405"
wget -c -O v1.0-trainval04_blobs.tgz "https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMEZVQSHGH&Signature=oV3VprFxu1zcERGRMxAyJpQmLdU%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHYaCXVzLWVhc3QtMSJIMEYCIQDYSFf669tSVA078drpV90pqDQedTmrUsqjsh093rFzpwIhALWN8tGIAhbT%2FWS7M%2FDwvTt5Od4UVQ066nXCPK1pNSWEKv0CCM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMOTk5MTM5NjA5Njg4IgxG3iMxsFnLZwkuD7Mq0QKIkr5ksHhDDWPsU1LSNON3Mgf0utli73g4VhnF7EWs6uRxhVEWmm2xUcJ2oUdnTDWy6jW3IGwZx4WKKZf1cQ1sfDNSwZ5PcOAwnKqRDXMfTW%2BuhZ0yuvNsPFZf8JHjn%2FUS9RMJGfvqr7IZAWpePPt2eyJ12JTxRuJOr9mnCLQJ2d8AUMaftEEXkQPcISJ9q53W3YHp3l835ioaDZCg966zrpfoL59RecVVMT1Id4rB%2BdN%2BXCFWd7iUBvtskdTzae34hnUibgnKTyrJOL2aSyWrE2JTl2IQVzZaHYycwWdpmJkwuj%2Bx7LcMNect63h3KTV053z8rCDQ%2BReOnMd3i2KZnYg5C6aznqYgjQ9IzM%2B%2BDzua2TxuN7JnA53FoJ4wOR1iufncNGX5ZaDMDGTEoXwtgyFq5n7cEgqiPxRmjJJCRBuoRILX%2Bw4DQAxZSZXrsq%2BsMKuUm6sGOp0BCaB2M7Fm8chD%2BxybQocvGPWkql8j4l4Ek332SMRjve0YcYCFCx7HkkGdwSHTBmd3pV0jqyPlIsvctwcZirGlgzg4mTZD65h6uR7dYpq%2FscWzEO63f9mkvVm%2BX8P%2BQeNBz0a4BwV4wLjKbvn2LC9AC%2FTlm1YyWBit2qZ%2B717wNqsxURkfF2O%2FDpPGDNV71UxMLh8Uo2AZXxxLqiM7ew%3D%3D&Expires=1701668432"
wget -c -O v1.0-trainval05_blobs.tgz "https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-trainval05_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMEZVQSHGH&Signature=phtsSl1B6v6hC81p8GnZDeKn%2FF8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHYaCXVzLWVhc3QtMSJIMEYCIQDYSFf669tSVA078drpV90pqDQedTmrUsqjsh093rFzpwIhALWN8tGIAhbT%2FWS7M%2FDwvTt5Od4UVQ066nXCPK1pNSWEKv0CCM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMOTk5MTM5NjA5Njg4IgxG3iMxsFnLZwkuD7Mq0QKIkr5ksHhDDWPsU1LSNON3Mgf0utli73g4VhnF7EWs6uRxhVEWmm2xUcJ2oUdnTDWy6jW3IGwZx4WKKZf1cQ1sfDNSwZ5PcOAwnKqRDXMfTW%2BuhZ0yuvNsPFZf8JHjn%2FUS9RMJGfvqr7IZAWpePPt2eyJ12JTxRuJOr9mnCLQJ2d8AUMaftEEXkQPcISJ9q53W3YHp3l835ioaDZCg966zrpfoL59RecVVMT1Id4rB%2BdN%2BXCFWd7iUBvtskdTzae34hnUibgnKTyrJOL2aSyWrE2JTl2IQVzZaHYycwWdpmJkwuj%2Bx7LcMNect63h3KTV053z8rCDQ%2BReOnMd3i2KZnYg5C6aznqYgjQ9IzM%2B%2BDzua2TxuN7JnA53FoJ4wOR1iufncNGX5ZaDMDGTEoXwtgyFq5n7cEgqiPxRmjJJCRBuoRILX%2Bw4DQAxZSZXrsq%2BsMKuUm6sGOp0BCaB2M7Fm8chD%2BxybQocvGPWkql8j4l4Ek332SMRjve0YcYCFCx7HkkGdwSHTBmd3pV0jqyPlIsvctwcZirGlgzg4mTZD65h6uR7dYpq%2FscWzEO63f9mkvVm%2BX8P%2BQeNBz0a4BwV4wLjKbvn2LC9AC%2FTlm1YyWBit2qZ%2B717wNqsxURkfF2O%2FDpPGDNV71UxMLh8Uo2AZXxxLqiM7ew%3D%3D&Expires=1701668442"
wget -c -O v1.0-trainval06_blobs.tgz "https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMEZVQSHGH&Signature=6OLO6Sx%2FixOY2xgJsjba2VcLRZ4%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHYaCXVzLWVhc3QtMSJIMEYCIQDYSFf669tSVA078drpV90pqDQedTmrUsqjsh093rFzpwIhALWN8tGIAhbT%2FWS7M%2FDwvTt5Od4UVQ066nXCPK1pNSWEKv0CCM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMOTk5MTM5NjA5Njg4IgxG3iMxsFnLZwkuD7Mq0QKIkr5ksHhDDWPsU1LSNON3Mgf0utli73g4VhnF7EWs6uRxhVEWmm2xUcJ2oUdnTDWy6jW3IGwZx4WKKZf1cQ1sfDNSwZ5PcOAwnKqRDXMfTW%2BuhZ0yuvNsPFZf8JHjn%2FUS9RMJGfvqr7IZAWpePPt2eyJ12JTxRuJOr9mnCLQJ2d8AUMaftEEXkQPcISJ9q53W3YHp3l835ioaDZCg966zrpfoL59RecVVMT1Id4rB%2BdN%2BXCFWd7iUBvtskdTzae34hnUibgnKTyrJOL2aSyWrE2JTl2IQVzZaHYycwWdpmJkwuj%2Bx7LcMNect63h3KTV053z8rCDQ%2BReOnMd3i2KZnYg5C6aznqYgjQ9IzM%2B%2BDzua2TxuN7JnA53FoJ4wOR1iufncNGX5ZaDMDGTEoXwtgyFq5n7cEgqiPxRmjJJCRBuoRILX%2Bw4DQAxZSZXrsq%2BsMKuUm6sGOp0BCaB2M7Fm8chD%2BxybQocvGPWkql8j4l4Ek332SMRjve0YcYCFCx7HkkGdwSHTBmd3pV0jqyPlIsvctwcZirGlgzg4mTZD65h6uR7dYpq%2FscWzEO63f9mkvVm%2BX8P%2BQeNBz0a4BwV4wLjKbvn2LC9AC%2FTlm1YyWBit2qZ%2B717wNqsxURkfF2O%2FDpPGDNV71UxMLh8Uo2AZXxxLqiM7ew%3D%3D&Expires=1701668477"
wget -c -O v1.0-trainval07_blobs.tgz "https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-trainval07_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMEZVQSHGH&Signature=dO5ImdQAiLbV9lgkTopHDSOhRMc%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHYaCXVzLWVhc3QtMSJIMEYCIQDYSFf669tSVA078drpV90pqDQedTmrUsqjsh093rFzpwIhALWN8tGIAhbT%2FWS7M%2FDwvTt5Od4UVQ066nXCPK1pNSWEKv0CCM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMOTk5MTM5NjA5Njg4IgxG3iMxsFnLZwkuD7Mq0QKIkr5ksHhDDWPsU1LSNON3Mgf0utli73g4VhnF7EWs6uRxhVEWmm2xUcJ2oUdnTDWy6jW3IGwZx4WKKZf1cQ1sfDNSwZ5PcOAwnKqRDXMfTW%2BuhZ0yuvNsPFZf8JHjn%2FUS9RMJGfvqr7IZAWpePPt2eyJ12JTxRuJOr9mnCLQJ2d8AUMaftEEXkQPcISJ9q53W3YHp3l835ioaDZCg966zrpfoL59RecVVMT1Id4rB%2BdN%2BXCFWd7iUBvtskdTzae34hnUibgnKTyrJOL2aSyWrE2JTl2IQVzZaHYycwWdpmJkwuj%2Bx7LcMNect63h3KTV053z8rCDQ%2BReOnMd3i2KZnYg5C6aznqYgjQ9IzM%2B%2BDzua2TxuN7JnA53FoJ4wOR1iufncNGX5ZaDMDGTEoXwtgyFq5n7cEgqiPxRmjJJCRBuoRILX%2Bw4DQAxZSZXrsq%2BsMKuUm6sGOp0BCaB2M7Fm8chD%2BxybQocvGPWkql8j4l4Ek332SMRjve0YcYCFCx7HkkGdwSHTBmd3pV0jqyPlIsvctwcZirGlgzg4mTZD65h6uR7dYpq%2FscWzEO63f9mkvVm%2BX8P%2BQeNBz0a4BwV4wLjKbvn2LC9AC%2FTlm1YyWBit2qZ%2B717wNqsxURkfF2O%2FDpPGDNV71UxMLh8Uo2AZXxxLqiM7ew%3D%3D&Expires=1701668505"
wget -c -O v1.0-trainval08_blobs.tgz "https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-trainval08_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMEZVQSHGH&Signature=k3D8YThbaXjxvD2jw4ORBwCHuf8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHYaCXVzLWVhc3QtMSJIMEYCIQDYSFf669tSVA078drpV90pqDQedTmrUsqjsh093rFzpwIhALWN8tGIAhbT%2FWS7M%2FDwvTt5Od4UVQ066nXCPK1pNSWEKv0CCM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMOTk5MTM5NjA5Njg4IgxG3iMxsFnLZwkuD7Mq0QKIkr5ksHhDDWPsU1LSNON3Mgf0utli73g4VhnF7EWs6uRxhVEWmm2xUcJ2oUdnTDWy6jW3IGwZx4WKKZf1cQ1sfDNSwZ5PcOAwnKqRDXMfTW%2BuhZ0yuvNsPFZf8JHjn%2FUS9RMJGfvqr7IZAWpePPt2eyJ12JTxRuJOr9mnCLQJ2d8AUMaftEEXkQPcISJ9q53W3YHp3l835ioaDZCg966zrpfoL59RecVVMT1Id4rB%2BdN%2BXCFWd7iUBvtskdTzae34hnUibgnKTyrJOL2aSyWrE2JTl2IQVzZaHYycwWdpmJkwuj%2Bx7LcMNect63h3KTV053z8rCDQ%2BReOnMd3i2KZnYg5C6aznqYgjQ9IzM%2B%2BDzua2TxuN7JnA53FoJ4wOR1iufncNGX5ZaDMDGTEoXwtgyFq5n7cEgqiPxRmjJJCRBuoRILX%2Bw4DQAxZSZXrsq%2BsMKuUm6sGOp0BCaB2M7Fm8chD%2BxybQocvGPWkql8j4l4Ek332SMRjve0YcYCFCx7HkkGdwSHTBmd3pV0jqyPlIsvctwcZirGlgzg4mTZD65h6uR7dYpq%2FscWzEO63f9mkvVm%2BX8P%2BQeNBz0a4BwV4wLjKbvn2LC9AC%2FTlm1YyWBit2qZ%2B717wNqsxURkfF2O%2FDpPGDNV71UxMLh8Uo2AZXxxLqiM7ew%3D%3D&Expires=1701668515"
wget -c -O v1.0-trainval09_blobs.tgz "https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-trainval09_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMEZVQSHGH&Signature=%2BMsiH9zsdW0MmfE0vZvozpT3QL0%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHYaCXVzLWVhc3QtMSJIMEYCIQDYSFf669tSVA078drpV90pqDQedTmrUsqjsh093rFzpwIhALWN8tGIAhbT%2FWS7M%2FDwvTt5Od4UVQ066nXCPK1pNSWEKv0CCM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMOTk5MTM5NjA5Njg4IgxG3iMxsFnLZwkuD7Mq0QKIkr5ksHhDDWPsU1LSNON3Mgf0utli73g4VhnF7EWs6uRxhVEWmm2xUcJ2oUdnTDWy6jW3IGwZx4WKKZf1cQ1sfDNSwZ5PcOAwnKqRDXMfTW%2BuhZ0yuvNsPFZf8JHjn%2FUS9RMJGfvqr7IZAWpePPt2eyJ12JTxRuJOr9mnCLQJ2d8AUMaftEEXkQPcISJ9q53W3YHp3l835ioaDZCg966zrpfoL59RecVVMT1Id4rB%2BdN%2BXCFWd7iUBvtskdTzae34hnUibgnKTyrJOL2aSyWrE2JTl2IQVzZaHYycwWdpmJkwuj%2Bx7LcMNect63h3KTV053z8rCDQ%2BReOnMd3i2KZnYg5C6aznqYgjQ9IzM%2B%2BDzua2TxuN7JnA53FoJ4wOR1iufncNGX5ZaDMDGTEoXwtgyFq5n7cEgqiPxRmjJJCRBuoRILX%2Bw4DQAxZSZXrsq%2BsMKuUm6sGOp0BCaB2M7Fm8chD%2BxybQocvGPWkql8j4l4Ek332SMRjve0YcYCFCx7HkkGdwSHTBmd3pV0jqyPlIsvctwcZirGlgzg4mTZD65h6uR7dYpq%2FscWzEO63f9mkvVm%2BX8P%2BQeNBz0a4BwV4wLjKbvn2LC9AC%2FTlm1YyWBit2qZ%2B717wNqsxURkfF2O%2FDpPGDNV71UxMLh8Uo2AZXxxLqiM7ew%3D%3D&Expires=1701668528"
wget -c -O v1.0-trainval10_blobs.tgz "https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-trainval10_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMEZVQSHGH&Signature=gcdBXGDL1rLvDr%2FEmN2wt0RNAnA%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHYaCXVzLWVhc3QtMSJIMEYCIQDYSFf669tSVA078drpV90pqDQedTmrUsqjsh093rFzpwIhALWN8tGIAhbT%2FWS7M%2FDwvTt5Od4UVQ066nXCPK1pNSWEKv0CCM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMOTk5MTM5NjA5Njg4IgxG3iMxsFnLZwkuD7Mq0QKIkr5ksHhDDWPsU1LSNON3Mgf0utli73g4VhnF7EWs6uRxhVEWmm2xUcJ2oUdnTDWy6jW3IGwZx4WKKZf1cQ1sfDNSwZ5PcOAwnKqRDXMfTW%2BuhZ0yuvNsPFZf8JHjn%2FUS9RMJGfvqr7IZAWpePPt2eyJ12JTxRuJOr9mnCLQJ2d8AUMaftEEXkQPcISJ9q53W3YHp3l835ioaDZCg966zrpfoL59RecVVMT1Id4rB%2BdN%2BXCFWd7iUBvtskdTzae34hnUibgnKTyrJOL2aSyWrE2JTl2IQVzZaHYycwWdpmJkwuj%2Bx7LcMNect63h3KTV053z8rCDQ%2BReOnMd3i2KZnYg5C6aznqYgjQ9IzM%2B%2BDzua2TxuN7JnA53FoJ4wOR1iufncNGX5ZaDMDGTEoXwtgyFq5n7cEgqiPxRmjJJCRBuoRILX%2Bw4DQAxZSZXrsq%2BsMKuUm6sGOp0BCaB2M7Fm8chD%2BxybQocvGPWkql8j4l4Ek332SMRjve0YcYCFCx7HkkGdwSHTBmd3pV0jqyPlIsvctwcZirGlgzg4mTZD65h6uR7dYpq%2FscWzEO63f9mkvVm%2BX8P%2BQeNBz0a4BwV4wLjKbvn2LC9AC%2FTlm1YyWBit2qZ%2B717wNqsxURkfF2O%2FDpPGDNV71UxMLh8Uo2AZXxxLqiM7ew%3D%3D&Expires=1701668537"

wget -c -O v1.0-test_meta.tgz "https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-test_meta.tgz?AWSAccessKeyId=ASIA6RIK4RRMFWUAB3PV&Signature=TFE8yharGpqxEecQktn7sMto8w0%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHYaCXVzLWVhc3QtMSJHMEUCIBSRc8QUocIGbRdjsv8l3qokyqnV9QJ7FH1wxYNnzVlIAiEAqlNtDdFqc8ljLZ0rD4JA03LXvLDNn10S6ninKgP05BMq%2FQIIz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgw5OTkxMzk2MDk2ODgiDFu6ZGBYp7DxUqeHqyrRAqXNw4xeyQFGFM4Dk%2BSPqWrJY6AholJvWk%2F%2BUHGBifLZ3ZIdJ2Y%2BP6I17BnAsmt1hhf3%2Ba9eV%2BacvhRIsbc9xgOqMws10ewCEvwU9lf3%2F3sC2t5dGFPaXP1CvqJjxFeqz24BvQglqWrb9MDN3KKAGE8cdo6iOlO5avXfkVbB7gsJRd5cP0%2FZI7NcRHBhSYnWT9p3B5R2QxKRJRMEyLUNq4eXlrCTeD3W11vbF7nGnBpnYemb1YkK9raZcA1w4UOxvcx9u0kRpUaq26G9Kj4HVhfCSwUeJuhkaZLIz0FXdFoQq%2FiwHcrkbR1PM7S1jkc4XIwKIq8bTmwZO0vS%2BYpRXBzdLE7iNCjulfnWEudAByRDAr9eMlrslTyRq85iSNJCcucr7IXxsjb5rPFcOa4XZnBMJA25GI%2B%2BK%2FWUnMCY5BKwa2%2FxiB5hpwE5iXCoJObm9XIwuZ6bqwY6ngFZbUH9zhnEU4BG%2BGTLtVXI1ihXmiqYZ20PenWPEFbBnIZUWcTPBWXYp56AIR5hMmmN0jzWLluytGbvW73mQ5goorTVZQ3wUtaldcmBghIW%2BErYZQcwg8ff5rYXuthPYBliHvlDABQgBwscE0c6aQkGyM%2Fdlq7LJ6jzR0OHdOLG3jBXyoGF3F8kVXpPjxmvwW4ds%2BxYYI%2BzxMXhLoe8cg%3D%3D&Expires=1701668548"
wget -c -O v1.0-test_blobs.tgz "https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/v1.0-test_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMFWUAB3PV&Signature=L%2F2kRslBJVUqc4UMBVY312QNVfs%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHYaCXVzLWVhc3QtMSJHMEUCIBSRc8QUocIGbRdjsv8l3qokyqnV9QJ7FH1wxYNnzVlIAiEAqlNtDdFqc8ljLZ0rD4JA03LXvLDNn10S6ninKgP05BMq%2FQIIz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgw5OTkxMzk2MDk2ODgiDFu6ZGBYp7DxUqeHqyrRAqXNw4xeyQFGFM4Dk%2BSPqWrJY6AholJvWk%2F%2BUHGBifLZ3ZIdJ2Y%2BP6I17BnAsmt1hhf3%2Ba9eV%2BacvhRIsbc9xgOqMws10ewCEvwU9lf3%2F3sC2t5dGFPaXP1CvqJjxFeqz24BvQglqWrb9MDN3KKAGE8cdo6iOlO5avXfkVbB7gsJRd5cP0%2FZI7NcRHBhSYnWT9p3B5R2QxKRJRMEyLUNq4eXlrCTeD3W11vbF7nGnBpnYemb1YkK9raZcA1w4UOxvcx9u0kRpUaq26G9Kj4HVhfCSwUeJuhkaZLIz0FXdFoQq%2FiwHcrkbR1PM7S1jkc4XIwKIq8bTmwZO0vS%2BYpRXBzdLE7iNCjulfnWEudAByRDAr9eMlrslTyRq85iSNJCcucr7IXxsjb5rPFcOa4XZnBMJA25GI%2B%2BK%2FWUnMCY5BKwa2%2FxiB5hpwE5iXCoJObm9XIwuZ6bqwY6ngFZbUH9zhnEU4BG%2BGTLtVXI1ihXmiqYZ20PenWPEFbBnIZUWcTPBWXYp56AIR5hMmmN0jzWLluytGbvW73mQ5goorTVZQ3wUtaldcmBghIW%2BErYZQcwg8ff5rYXuthPYBliHvlDABQgBwscE0c6aQkGyM%2Fdlq7LJ6jzR0OHdOLG3jBXyoGF3F8kVXpPjxmvwW4ds%2BxYYI%2BzxMXhLoe8cg%3D%3D&Expires=1701668559"

```



3. 执行下载

```bash
tmux            # 开tmux下载，因为下载时间太久
mkdir -p ~/datasets/nuscenes
vim nuscenes.sh # 将上面的粘贴进去
sh nuscenes.sh  # 开始下载
```

4. 下载示意

![](./.assets/image-20231129135143440.png)

5. 挂起tmux

   ctrl + a d

6. 下载完成之后得到下面的文件

![image-20231130111821046](./.assets/image-20231130111821046.png)

7. 解压操作，在当前`/datasets/nuscenes`目录下，执行`nuscenes_unzip.sh`

```
tar zxvf [文件名.tgz] -C ./
```

```bash
tar zxvf v1.0-trainval_meta.tgz -C ./
tar zxvf v1.0-trainval01_blobs.tar -C ./
tar zxvf v1.0-trainval02_blobs.tar -C ./
tar zxvf v1.0-trainval03_blobs.tgz -C ./ 
tar zxvf v1.0-trainval04_blobs.tgz -C ./ 
tar zxvf v1.0-trainval05_blobs.tgz -C ./ 
tar zxvf v1.0-trainval06_blobs.tgz -C ./ 
tar zxvf v1.0-trainval07_blobs.tgz -C ./ 
tar zxvf v1.0-trainval08_blobs.tgz -C ./ 
tar zxvf v1.0-trainval09_blobs.tgz -C ./
tar zxvf v1.0-trainval10_blobs.tgz -C ./ 

tar zxvf v1.0-test_meta.tgz -C ./
tar zxvf v1.0-test_blobs.tgz -C ./

```

8. 删除多余的压缩包和sh文件 `remove.sh`

```shell
rm v1.0-trainval_meta.tgz
rm v1.0-trainval01_blobs.tar 
rm v1.0-trainval02_blobs.tar 
rm v1.0-trainval03_blobs.tgz 
rm v1.0-trainval04_blobs.tgz 
rm v1.0-trainval05_blobs.tgz 
rm v1.0-trainval06_blobs.tgz 
rm v1.0-trainval07_blobs.tgz
rm v1.0-trainval08_blobs.tgz 
rm v1.0-trainval09_blobs.tgz 
rm v1.0-trainval10_blobs.tgz

rm v1.0-test_meta.tgz 
rm v1.0-test_blobs.tgz 
rm nuscenes.sh
rm nuscenes_unzip.sh
```

9. 删除remove.sh

```
rm remove.sh
```



10. 目录树

![image-20231130163510821](./.assets/image-20231130163510821.png)

### KITTI数据集操作

文件内容（老师scp传给我的文件 嘿嘿！！）

```
data_object_calib.zip  data_object_image_2.zip  data_object_image_3.zip  data_object_label_2.zip  data_object_velodyne.zip  train_planes.zip
```

解压

```
#可选,另一个视角的摄像头照片
unzip data_object_image_3.zip
#可选,道路平面信息，其在训练过程中作为一个可选项，用于提升模型性能
unzip train_planes.zip
```



1. 把这些文件放在同一个文件夹下，路径为 `~/datasets/kitti/`
2. 在上述路径执行下面脚本`kitti_unzip.sh`

```bash
unzip data_object_calib.zip
unzip data_object_image_2.zip
unzip data_object_label_2.zip  
unzip data_object_velodyne.zip  
```

3. 创建数据集的所有编号文档 data split，在上面路径下执行下面脚本`create_split.sh`

可能会被墙，我电脑上有（直接copy进去），这里面存了数据集的编号

![](./.assets/image-20231130105305602.png)

```bash
mkdir -p ./ImageSets
# Download data split
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O ./ImageSets/test.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O ./ImageSets/train.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O ./ImageSets/val.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./ImageSets/trainval.txt
```



4. 删除压缩包，sh文件，以及不需要的可选的压缩包 `remove.sh`

```bash
rm data_object_calib.zip  
rm data_object_image_2.zip  
rm data_object_image_3.zip  
rm data_object_label_2.zip  
rm data_object_velodyne.zip  
rm train_planes.zip
rm kitti_unzip.sh
rm create_split.sh
```

5. 删除`remove.sh`

```bash
rm  remove.sh
```

6. 文件组织好是这个样子

![](./.assets/image-20231130105119070.png)



7. **软链接**，处理标签数据

（对软连接后的文件进行修改**相当于在原来的目录下修改**）

现在我的数据集源文件在目录`~/datasets/kitti`下

现在项目需要数据集在目录`mmdetection3d/data/kitti`下

因此我 `cd mmdetection3d/data`

```
ln  -s  [源文件或目录]  [目标文件或目录]
cd ~/mmdetection3d/data
# 当前路径创建kitti 引向 ~/datasets/kitti 文件夹 
ln -s ~/datasets/kitti ./kitti
```



8. 软链接删除

   `ls -l`查看链接情况

   ![image-20231212103524558](./.assets/image-20231212103524558.png)

   `rm flower_data`源文件还在



## 各种依赖包的下载

### MMCV安装

一定要把cuda版本切换过来，不然会卡住

要source一下

![image-20231130121258348](./.assets/image-20231130121258348.png)

用pip安装，指定mmcv版本

```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/[cuda_version]/[torch version]/index.html
# 例如，我的服务器是cuda11.6 torch是1.12.0
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html

mmcv与cuda对应



### torch本地安装

https://download.pytorch.org/whl/torch_stable.html

pip install 对应的包即可

### torch和torch.lighting对应版本关系

[Versioning Policy — PyTorch Lightning 2.2.0dev documentation](https://lightning.ai/docs/pytorch/latest/versioning.html)

# Latex

## latex基本概念

### latex源代码结构

```latex
\documentclass{...} % ... 为某文档类
% 导言区 使用宏包/进行文档的全局设置
\begin{document}
% 正文内容
\end{document}
% 此后内容会被忽略

```

### 宏包和文档类

#### 文档类

指定文档类

```latex
\documentclass[⟨options⟩]{⟨class-name⟩}
```

`<class-name>`

- article 文章格式的文档类，广泛用于科技论文、报告、说明文档等。
- report 长篇报告格式的文档类，具有章节结构，用于综述、长篇论文、简单
- 的书籍等。
- book 书籍文档类，包含章节结构和前言、正文、后记等结构。
- proc 基于article 文档类的一个简单的学术文档模板。
- slides 幻灯格式的文档类，使用无衬线字体。
- minimal 一个极其精简的文档类，只设定了纸张大小和基本字号，用作代码测
- 试的最小工作示例（Minimal Working Example）

`<options>`

全局地规定一些排版的参数，如字号、纸张大小、单双面

`例如`

调用article 文档类排版文章，指定纸张为A4 大小，基本字号为11pt，双面排版

```latex
\documentclass[11pt,twoside,a4paper]{article}
```

#### 宏包

调用宏包，可以一次性调用多个宏包，在⟨package-name⟩ 中用逗号隔开

```latex
\usepackage[⟨options⟩]{⟨package-name⟩}
```

### latex中用到的文件

.sty 宏包文件。宏包的名称与文件名一致

.cls 文档类文件。文档类名称与文件名一致。

.bib BIBTEX 参考文献数据库文件。

.bst BIBTEX 用到的参考文献格式模板。

### 文件组织形式

当导言区内容较多时，常常将其单独放置在一个.tex 文件中，再用\input 命令插入。复杂的图、表、代码等也会用类似的手段处理。

```latex
\input{<filename>}
```



# 杂项

## 目标检测视频素材网站

https://www.sohu.com/a/133366976_692322

## 







