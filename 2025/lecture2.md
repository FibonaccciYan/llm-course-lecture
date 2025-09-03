---
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
# backgroundImage: url('https://marp.app/assets/hero-background.svg')
marp: true
---
<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
a[href='red'] {
    color: red;
    pointer-events: none;
    cursor: default;
    text-decoration: none;
}
</style>




![bg left:45% 80%](../images/course.webp)

# **大语言模型基础：从零到一实现之路**

第2讲: 特征空间的变换
前反向运行视角理解深度学习模型

<!-- https://marp.app/ -->

---

# 深度学习任务的核心

* 按照目标场景的需要去理解“世界”
<div style="display:contents;" data-marpit-fragment>
这是何物

![w:300 right](../images/2025/l2/man.webp)

</div>

---

# 这是何物第二关



<p align="center">
  <img width="380" height="400" src="../images/2025/l2/man_icon.jpg">
  <img width="200" height="400" src="../images/2025/l2/woman_icon.jpg">
</p>

---

# 这是何物第三关


<div style="display:contents;" data-marpit-fragment>
我可不想被发律师函（手动狗头）
</div>


---


# 理解世界的方式

* 人如何理解世界: 通过外延观察=>自动构建内涵
  
* 内涵: 一个概念的内涵是指它的“内容”或“定义”，即该概念所包含的性质、特征或条件
  * “人”: 有理性、社会性和自我意识的生物
* 外延: 一个概念的外延是指它所指代的所有对象的集合，或者说它所涵盖的实际事物的范围
  * “人”: 是所有的人类个体，如亚里士多德、牛顿、爱因斯坦

---

# 模型学习的核心

模型如何理解世界: 通过外延观察=>自动构建内涵

<div style="display:contents;" data-marpit-fragment>

那如何把上面的概念落到实处呢？

将世界“特征化”，基于特征化后的向量，通过函数计算完成具体的任务
$Y=f(X)=XW+b$

</div>

* 模型理解世界的过程
  * 设定任务目标
  * 收集外延: 数据集
  * 构建内涵: 学习特征

---

# 深度学习的特点

表示学习(representation learning)

![w:1100 center](../images/2025/l2/semanticrepresentation.png)

---

# 深度学习模型

![bg 65%](../images/2025/l2/deep-learning-model-arch.png)




---

# 深度学习模型

![bg right:55% 100%](../images/2025/l2/deep-learning-model.png)

典型结构
* 前n-1层(堆叠)
  * 输入: 特征表示
  * 输出: 特征表示
* 第n层
  * 输入: 特征表示
  * 输出: 任务目标

---

# 深度学习模型的线性层

$Y=f(X)=XW+b$
* $X$和$Y$: 特征表示
  * $X$: 当前层输入/前一层输出
  * $Y$: 当前层输出/后一层输入
* $W,b$: 当前层参数
    * 表示空间变换函数



![bg right:43% 80%](../images/2025/l2/one-layer.png)

---

# 线性层的计算

输入$X$：b个样本，每个样本表示在一个d维的空间中
函数$f(X)$：函数将表示在d维空间中的样本变换到e维的空间中
输出$Y$：$f(X)$的输出，b个样本，每个样本表示在一个e维的空间中


![w:900 center](../images/2025/l2/linear.png)



---

# 空间变换的具体实现

理论上：运用线性代数中的矩阵乘法

![w:900 center](../images/2025/l2/matmul_elem.png)

---

## 矩阵乘法的复杂度分析

- 计算 $Y = XW$ 时，每个元素 $y_{ij}$ 是一个 **点积**：  
  $$
  y_{ij} = \sum_{k=1}^{d} x_{ik} \cdot w_{kj}
  $$
- 每个 $y_{ij}$ 需要 $O(d)$ 次乘加运算  
- 总共有 $b \times e$ 个元素，$O(b\cdot d \cdot e)$
![w:900 center](../images/2025/l2/matmul_elem.png)

---

# 矩阵乘法的优化思路
- 内存局部性 (Blocking)
  - 数据按小块（tile）加载到cache中，避免反复访问大矩阵
- 向量化计算 (Tiling)
  - 一次处理多个数据 (SIMD/SIMT)，例如一次性做 8 个乘加
- 并行化
  - 多核CPU：不同线程处理不同矩阵块
  - GPU：上万线程同时处理小块矩阵

<!-- ---

# GPU 上的矩阵乘法

- GPU 将矩阵划分为很多小块（thread block）
- 每个 block 在显存和共享内存之间做高速数据交换
- CUDA 核函数中：
  - 每个线程负责计算 $Y$ 中的一个或一小片元素
  - 通过 warp/wavefront 协同提高吞吐 -->

---

# PyTorch手搓乘法

承载数据的基本类型：Tensor。那什么是 Tensor？

- Tensor可理解为多维数组
- 在 PyTorch 中是最基础的数据结构  
- 类似 NumPy 的 ndarray，但有两个关键特性：
  - 支持在GPU上高效计算  
  - 支持自动求导(autograd)

---

## Tensor 的维度

- 标量 (Scalar)：0 维张量  
- 向量 (Vector)：1 维张量
- 矩阵 (Matrix)：2 维张量，常用与模型参数W
- 高维 Tensor：常用于输入输出(Input/Output)/激活(Activation)

```python
  torch.tensor(3.14)   # shape = []
  torch.tensor([1, 2, 3])   # shape = [3]
  torch.tensor([[1, 2], [3, 4]])   # shape = [2, 2]
  x = torch.randn(2, 3, 4, 5) # shape = [2, 3, 4, 5]
  ```

---

# PyTorch手搓乘法

- 操作数有了，那操作符呢？
- 矩阵乘法 (Matrix multiplication)
  - Operator @
  - torch.matmul, tensor.matmul
- 元素乘法 (element-wise multiplication)
  - Operator \*
  - torch.mul, tensor.mul
- 高阶的: 
  - torch.einsum

---

# 编码时间

矩阵乘
```python
y = x@w
y = x.matmul(w)
y = torch.matmul(x, w)
```
元素乘
```python
y = x*w
y = x.mul(w)
y = torch.mul(x, w)
```

---

# einsum

- torch.einsum (equation, tensor list)
  - equation: 使用 **爱因斯坦求和约定 (Einstein Summation Convention)** 描述张量运算 (基于tensor下标)
  - tensor list: input
- 优点：直观，代码可读性强


---

# einsum 表达式
矩阵乘法$Y = XW$的数学形式：  
  $$
  Y_{be} = \sum_{d} X_{bd} \cdot W_{de}
  $$
einsum 表达式
  ```python
  import torch

  X = torch.randn(2, 3)   # B=2, D=3
  W = torch.randn(3, 4)   # D=3, E=4
  Y = torch.einsum("bd,de->be", X, W)
  print(Y.shape)  # torch.Size([2, 4])
```

---

# 批量矩阵乘
- 数学形式：
$$
Y_{b i j} = \sum_{k} A_{b i k} \cdot B_{b k j}
$$
- einsum 表达式：
```python
A = torch.randn(10, 3, 4)   # batch=10
B = torch.randn(10, 4, 5)

Y = torch.einsum("bik,bkj->bij", A, B)
print(Y.shape)  # torch.Size([10, 3, 5])
```

---

# 构建最基础/核心的模型"积木"

线性层 (torch.nn.Linear): $Y=XW$

* torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
  * in_features: size of each input sample
  * out_features: size of each output sample

* PyTorch中的输入/输出: 都是tensor
  * input: $(∗,H_{in})$
  * output: $(∗,H_{out})$
  
---

# "积木"nn.Linear的要素


```
self.in_features = in_features
self.out_features = out_features
self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
```
* weight: W
  * 与in_features和out_features共同规约了输入输出部分维度的尺寸
  * 实现了从一个空间到另一个空间的变换
  
---

# "积木"nn.Linear的要素

```python
def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
```

* 计算方法forward: 定义输入到输出的计算过程
  * nn.Linear的forward: 实现$Y=XW$


---

# nn.Linear的使用

Torch docs中的官方示例
```python
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
```

---

# 新手村之积木堆叠

```python
m1 = nn.Linear(20, 30)
m2 = nn.Linear(30, 40)
x = torch.randn(128, 20)
y1 = m1(x)
y2 = m2(y1)
```
* 基于nn.Linear实现以下函数组合
  * $Y_1 = XW_1$
  * $Y_2 = Y_1W_2$
  
---

# 思考：这样的输入行么？

```python
x = torch.randn(128, 4096, 30, 20)
y = m1(x)
y = m2(y)
```
---

# 多种多样的积木

* 线性层(Linear layer), 卷积层(Convolutional layer), 池化层(Pooling layer), 各类正则化(XNorm layer)
* 自定义layer
  * Attention layer, Transformer Block, Decoder Layer, ...

---

# 反向传播：深度学习的关键

* 前向传播：从输入到输出，计算预测结果
* 反向传播：从输出到输入，计算梯度，更新参数
* 核心思想：通过链式法则计算复合函数的导数

---

![w:1200 center](../images/2025/l2/forward_backward.png)


---

# 为什么需要反向传播？

* **目标**：找到使损失函数最小的参数
* **方法**：梯度下降，需要计算损失函数对每个参数的梯度
* **挑战**：深度学习模型是复合函数，直接计算梯度困难
* **解决**：反向传播算法，高效计算所有参数的梯度

---

### 模型的训练/学习
<!-- ![bg right:40% 100%](https://miro.medium.com/max/1024/1*G1v2WBigWmNzoMuKOYQV_g.png) -->

* 假设，构建模型$f$，其参数为$\theta$
* 目标: 设计一种可用来度量基于$\theta$的模型预测结果和真实结果差距的度量，差距越小，模型越接近需估计的函数$f^*$
  * $J(\theta)=\frac{1}{n}\sum_{X\in \mathcal{X}}(f^*(X)-f(X;\theta))^2$
* 学习方法：梯度下降，寻找合适的$\theta$ (被称之**训练模型**)
<div style="display:contents;" data-marpit-fragment>

![w:400 center](../images/2025/l2/grad.png)
</div>


---

# 模型的训练/学习 

* 目标: $J(\theta)=\frac{1}{n}\sum_{X\in \mathcal{X}}(Y-f(X;\theta))^2$

<div style="display:contents;" data-marpit-fragment>

1. 猜个$\theta$, 根据输入$X$，计算$\hat{Y}=f(X;\theta)$
2. 评估误差: $Y$和$\hat{Y}$的误差(loss)
3. 根据误差，更新$\theta$: $\theta=\theta -\lambda\cdot\Delta\theta$

![w:400 center](../images/2025/l2/grad.png)
</div>

---

![bg 80%](https://shashank-ojha.github.io/ParallelGradientDescent/non-convex.png)


---

# 训练模型(搜索/开发参数)

优化目标: $J(\theta)=\frac{1}{n}\sum_{X\in \mathcal{X}}(f^*(X)-f(X;\theta))^2$

梯度下降法 (Gradient descent): 求偏导
* $f^*(X)$通常以真值(groundtruth)体现，因此$\frac{\partial}{\partial \theta}J(\theta)$重点关注$f(X;\theta)$
  * $f(X)=XW$ --> $\frac{\partial}{\partial \theta}f(X)=\frac{\partial}{\partial \theta}(XW)$ 
  * 通常深度学习模型$f(X)$为复合函数，需利用链式法则求偏导

* 核心算法: 反向传播(backpropagation)
  * 核心步骤: 针对优化目标$J(\theta)$求其偏导数(partial derivative)


---

# 反向传播(backpropagation)

* 假设深度学习模型为$f(X)=XW$的复合函数
  * $y=f_3(f_2(f_1(X)))$
* 优化目标$J(\theta)$的偏导$\frac{\partial}{\partial \theta}J$的核心为$\frac{\partial}{\partial \theta}y=\frac{\partial}{\partial \theta}f_3(f_2(f_1(X)))$
* 链式法则展开:
  * $\frac{\partial J}{\partial \theta_{f_1}} = \frac{\partial J}{\partial y}\cdot \frac{\partial y}{\partial f_3}\cdot \frac{\partial f_3}{\partial f_2}\cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial \theta_{f_1}}$

* 偏导的构建
  * 传统手工实现 v.s. 基于计算图的autograd

---

# 计算图：模型计算的DAG图

* **节点**：表示变量（输入、参数、中间结果）
* **边**：表示操作（加法、乘法、激活函数等）
* **前向传播**：沿着边的方向计算
* **反向传播**：沿着边的反方向计算梯度

![w:800 center](https://docs.pytorch.org/tutorials/_images/comp-graph.png)

---

# 举个例子：线性层的反向传播

线性层：$z = x \cdot w + b$
损失函数: $loss = \text{CrossEntropy}(z, target)$
**前向传播**：
- 输入：$x$, 参数：$w$, $b$
- 输出：$z = x \cdot w + b$
**反向传播**（链式法则）：
- $\frac{\partial loss}{\partial w} = \frac{\partial loss}{\partial z} \cdot \frac{\partial z}{\partial w} = \frac{\partial loss}{\partial z} \cdot x$
- $\frac{\partial loss}{\partial b} = \frac{\partial loss}{\partial z} \cdot \frac{\partial z}{\partial b} = \frac{\partial loss}{\partial z} \cdot 1 = \frac{\partial loss}{\partial z}$
- $\frac{\partial loss}{\partial x} = \frac{\partial loss}{\partial z} \cdot \frac{\partial z}{\partial x} = \frac{\partial loss}{\partial z} \cdot w$

其中 $\frac{\partial loss}{\partial z}$ 是损失函数对输出的梯度，由损失函数的具体形式决定。

---

# 自动求导：PyTorch的核心特性

* **计算图**：PyTorch自动构建计算图
  * 基于`nn.Module`以及用户实现的`forward`方法
* **梯度追踪**：在PyTorch中，Tensor对象设置`requires_grad=True`即可启用对该Tensor的梯度计算
* **反向传播**：`backward()` 自动计算梯度

---

# 梯度清零：训练循环中的重要步骤

```python
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(100):
    # 前向传播
    y = net(x)
    loss = criterion(y, target)
    
    # 反向传播
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
```

