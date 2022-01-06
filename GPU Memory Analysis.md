## GPU Memory Analysis

`nvidia-smi`是Nvidia显卡命令行管理套件，基于NVIDIA Management Library (NVML) 库，旨在管理和监控Nvidia GPU设备。

其中最重要的两个指标：

- 显存占用
- GPU利用率

显存占用和GPU利用率是两个不一样的东西，显卡是由GPU计算单元和显存等组成的，显存和GPU的关系有点类似于内存和CPU的关系。

gpustat基于`nvidia-smi`，可以提供更美观简洁的展示，结合watch命令，可以**动态实时监控**GPU的使用情况。

```shell
pip install gpustat
watch --color -n1 gpustat -cpu 
```

**显存可以看成是空间，类似于内存。**

- 显存用于存放模型，数据
- 显存越大，所能运行的网络也就越大

**GPU计算单元**类似于CPU中的核，用来进行数值计算。衡量计算量的单位是flop；计算能力越强大，速度越快。衡量计算能力的单位是flops： 每秒能执行的flop数量

### 存储指标

```text
1Byte = 8 bit
1K = 1024 Byte
1M = 1024 K
1G = 1024 M
1T = 1024 G

10 K = 10*1024 Byte
```

除了`K`、`M`，`G`，`T`等之外，我们常用的还有`KB` 、`MB`，`GB`，`TB` 。二者有细微的差别。

```text
1Byte = 8 bit
1KB = 1000 Byte
1MB = 1000 KB
1GB = 1000 MB
1TB = 1000 GB

10 KB = 10000 Byte
```

`K`、`M`，`G`，`T`是以1024为底，而`KB` 、`MB`，`GB`，`TB`以1000为底。不过一般来说，在估算显存大小的时候，我们不需要严格的区分这二者。

在深度学习中会用到各种各样的数值类型，数值类型命名规范一般为`TypeNum`，比如Int64、Float32、Double64。

- Type：有Int，Float，Double等
- Num: 一般是 8，16，32，64，128，表示该类型所占据的比特数目

常用的数值类型如下图所示(*int64 准确的说应该是对应c中的long long类型， long类型在32位机器上等效于int32*)：

|   类型   |  大小   |                             备注                             |
| :------: | :-----: | :----------------------------------------------------------: |
|   int8   | 1个字节 |                           又名Byte                           |
|  int16   | 2个字节 |                          又名Short                           |
|  int32   | 4个字节 |                           又名Int                            |
|  int64   | 8个字节 |                        又名long long                         |
| float16  | 2个字节 |                         半精度浮点数                         |
| bfloat16 | 2个字节 | 对float32直接截取，<br />比float16更宽的表示范围<br />tensorflow+tpu提出 |
| float32  | 4个字节 |                   单精度浮点数，又名float                    |
| float64  | 8个字节 |                         双精度浮点数                         |

举例来说：有一个1000x1000的 矩阵，float32，那么占用的显存差不多就是

> 1000x1000x4 Byte = 4MB

32x3x256x256的四维数组（BxCxHxW）占用显存为：24M

### 神经网络显存占用

#### 1. 参数的显存占用

只有有参数的层，才会有显存占用。这部份的显存占用和**输入无关**，只与网络层的参数量有关，模型加载完成之后就会占用。

**有参数的层主要包括：**

- 卷积
- 全连接
- BatchNorm
- Embedding层
- ... ...

**无参数的层**：

- 多数的激活层(Sigmoid/ReLU)
- 池化层
- Dropout
- ... ...

更具体的来说，模型的参数数目(这里均不考虑偏置项b)为：

- Linear(M->N): 参数数目：M×N
- Conv2d(Cin, Cout, K): 参数数目：Cin × Cout × K × K
- BatchNorm(N): 参数数目： 2N
- Embedding(N,W): 参数数目： N × W

**参数占用显存 = 参数数目×n**，n为参数dtype，例如

- dtype=float32：n = 4 

-  dtype=float16：n = 2

-  dtype=double64：n = 8

在PyTorch中，当你执行完`model=MyGreatModel().cuda()`之后就会占用相应的显存，占用的显存大小基本与上述分析的显存差不多（会稍大一些，因为其它开销）。

#### 2. 梯度与动量的显存占用

- 优化器如果是SGD,可以看出来，除了保存W之外还要保存对应的梯度，因此显存占用等于参数占用的显存x2

$$
W_{t+1}=W_t-\alpha \nabla F(W_t)
$$

- 如果是带Momentum-SGD，这时候还需要保存动量， 因此显存x3

$$
v_{t+1}=\rho v_t+\nabla F(W_t) \\
W_{t+1}=W_t-\alpha v_{t+1}
$$

- 如果是Adam优化器，动量占用的显存更多，显存x4

总结一下，模型中**与输入无关的显存占用**包括：

- 参数 **W**
- 梯度 **dW**（一般与参数一样）
- 优化器的**动量**（普通SGD没有动量，momentum-SGD动量与梯度一样，Adam优化器动量的数量是梯度的两倍）

#### 3. 输入与输出的显存占用

这部份的显存主要看输出的feature map 的形状，计算出每一层输出的Tensor的形状，然后就能计算出相应的显存占用。

### 总结

深度学习中神经网络的显存占用，我们可以得到如下公式：

```text
显存占用 = 模型显存占用 + batch_size × 每个样本的显存占用
其中：
模型显存占用 = 参数占用 + 梯度占用 + 动量占用
样本的显存占用 = 输入tensor占用 + 每层的输出tensor占用
```

可以看出显存不是和batch-size简单的成正比，尤其是模型自身比较复杂的情况下：

- 当全连接或embedding很多时，模型显存占用很大
- 当卷积层很多时且针对高分辨率input size时，样本的显存占用很大

另外需要注意：

- 输入（数据，图片）一般不需要计算梯度
- 神经网络的每一层输入输出都需要保存下来，用来反向传播，但是在某些特殊的情况下，我们可以不要保存输入。比如ReLU，在PyTorch中，使用`nn.ReLU(inplace = True)` 能将激活函数ReLU的输出直接覆盖保存于模型的输入之中（因为BP时， $dx = dy.copy(),y>0;dx[y<=0]=0$），节省不少显存。

### Appendix

不同的神经网络层的参数量、前向与后向传播计算量分析可参考：[AIPerf](https://arxiv.org/pdf/2008.07141.pdf)

<img src="figures/fpoperation.png" alt="bp" style="zoom:80%;" /> <img src="figures/bpoperation.png" alt="bp" style="zoom:80%;" />

