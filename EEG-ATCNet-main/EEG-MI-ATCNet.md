## 写在前面的话

:blush:首先，之所以要介绍这篇论文所提出的**ATCNet是因为这个模型相较于经典的EEGNet、DeepConvNet等网络效果更好，可解释性更强**。"A"代表的是"attention"结构也就是注意力机制，”[TCNet](https://www.bbsmax.com/A/KE5QD1oyJL/)”代表的是前几年提出的一种模型结构，已经有[论文](EEG-TCNet: An Accurate Temporal Convolutional
Network for Embedded Motor-Imagery
Brain–Machine Interfaces)证明TCNet在EEG-MI任务重的优异特性，ATCNet相当于在此基础上加入了注意力机制。另外，**关于这篇文章的复现代码中包含EEGNet、DeepConvNet、ShallowConvNet等多种经典EEG-MI网络模型可供读者比较使用，各种模型代码均可运行**。

:blush:其次，ATCNet结构主要由三个模块构成：卷积块，注意力(AT)块，时间卷积块，并且包含一个滑动卷积结构，这个结构是上述TCN模型的一大特点。关于各个模块的细节在下文中进行细致介绍。

:blush:最后是数据集的问题，数据集使用的是经典的[BCI Competition IV-2a](http://www.bbci.de/competition/iv/#dataset2a)，读者可自行下载，需要强调的是：本代码复现使用的是.mat格式包含T/E两部分的代码。这是一个22 电极 EEG 运动图像数据集，有 9 个受试者和 2 个会话，每个受试者有 288 个四秒的想象运动试验。包括左手、右手、脚和舌头的动作。其官方详细介绍于[此](https://www.bbci.de/competition/iv/desc_2a.pdf)

###  摘要

:kissing:这是一种基于注意力的时间卷积网络（ATCNet）用于基于EEG的运动图像分类。

:satisfied:ATCNet模型利用多种技术以相对较少的参数提高MI分类的性能。ATCNet采用科学的机器学习来设计具有可解释和可解释特征的特定领域DL模型，多头自我关注来突出MI-EEG数据中最有价值的特征，时间卷积网络（TCN）来提取高级时间特征，以及基于卷积的滑动窗来有效地增强MI-EEG的数据。所提出的模型优于BCI竞赛IV-2a数据集中的现有技术，对于受试者相关模式和受试者无关模式的准确率分别为85.38%和70.97%。

**论文背景及工作意义**

:sweat_drops:背景上：最近，一种称为**时间卷积网络（TCN）**的新CNN变体专门设计用于时间序列建模和分类。在许多序列相关任务中，TCN优于其他递归网络，如LSTM和GRU。**与典型的神经网络相比，TCN可以随着参数数量的线性增加，以指数方式扩大感受野的大小，并且与RNN不同，它不会受到消失或爆炸梯度的影响。**最近的一些研究使用TCN架构对MI任务进行分类[22](https://www.bbsmax.com/A/KE5QD1oyJL/)，[23](Musallam Y K, AlFassam N I, Muhammad G, et al. Electroencephalography-based motor imagery classification using temporal convolutional network fusion[J]. Biomedical Signal Processing and Control, 2021, 69: 102826.)。Ingolfsson等人[22](https://www.bbsmax.com/A/KE5QD1oyJL/)提出了一种名为EEGTCN的TCN模型，该模型将TCN与众所周知的EEGNet架构相结合。[23](Musallam Y K, AlFassam N I, Muhammad G, et al. Electroencephalography-based motor imagery classification using temporal convolutional network fusion[J]. Biomedical Signal Processing and Control, 2021, 69: 102826.)中最近的一项研究试图使用特征融合技术改进EEG-TCN模型。我们的研究是对这些工作的持续贡献，它利用科学机器学习（SciML）和TCN架构的注意力机制。

:exclamation:需要特别说明的是：这篇论文利用到的两个模块Conv_block与TCN_block确实是和TCNet中的结构是一毛一样，只不过受到滑动卷积块的限制，在Conv_block模块中第二个平均池化的卷积核大小变为了（7,1）TCNet中是(8,1).其他的都是一样的。

:wave:意义上：**除了上述提到的这两个模块Conv_block与TCN_block之外，作者在本论文中的最大创新点在于使用注意力机制，并且使用了Convolutional-based sliding window (SW)也就是基于卷积的滑动窗口设计。尽管上述提到的这些都是有迹可循的，也就是说并不是作者独创的结构，但是把他们拼接起来作为一种模型，并且能够得到较高的准确率，这也是非常值得肯定的。**

:point_right:总体而言：在本文中，我们提出了一种基于注意力的时间卷积网络ATCNet来解码MI-EEG脑信号。这项研究利用SciML来解决特定领域的MI-EEG数据挑战，这产生了一个专门设计用于解码MI-EEG脑信号的鲁棒、可解释和可解释的DL模型。**所提出的DL模型分三个步骤处理MI-EEG数据：首先，使用传统层(指的是Conv_block)将MI-EEG信号编码为一系列高级时间表示，然后，使用注意力结构突出显示时间序列中最有价值的信息，最后，使用时间卷积层从突出显示的信息中提取高级时间特征。所提出的模型利用多头自关注和基于卷积的滑动窗口来提高MI分类的性能。**这项研究突出了以下贡献：

:walking:1.提出了一个高性能的ATCNet模型，它利用了TCN、SciML、注意力机制和基于卷积的滑动窗口的强大功能。

:runner:2.通过并行化过程和减少计算，使用卷积执行滑动窗口有助于增加MI数据并有效地提高准确性。

:running:3.自我注意有助于DL模型关注EEG数据中最有效的MI信息，多个头部有助于关注多个位置，从而产生多个注意力表征。

:couple:4.所提出的模型在BCI竞赛IV-2a（BCI-2a）数据集中取得了优异的结果[30]。

下面将分为几个部分对这篇论文进行一个详细的解读：

### 数据集部分

:ok_woman:这篇论文中只用了bci-2a的数据，把带有T(train)的作为训练集，带有E(evalution)的作为测试集，并且是独立于受试者，并不是说所有的数据集全部都放一块来训练，其实独立于受试者(针对每个受试者是有意义的)，**全部放在一起我个人感觉是没有意义的，因为我们不可能设计出一个适合于所有人的模型结构。**

:no_good:**这里的数据集的关键点在于利用trial里面的分隔点，也就是251,2251,,...,94758来进行来进行一个分割,也就是说针对每一个人的T/E数据，包括9个struct结构，而前三个struct结构是不用的，用到的是后面的6个struct结构。然后以trial里面的数据点251,2251,,...,94758作为分隔点，以window_length=1750作为数据的长度，举例来说，第一个实验的数据为251到251+1750，第二个是2251到2251+1750，以此类推直到最后一个94758到94758+1750。因为每个struct里面包含48个实验，因为每一个人的T/E总的实验个数为288个，然后数据的长度为1750，，通道数位22，因此数据的shape为(288,1750,22)。**这段话需要读者根据数据仔细理解，我个人也是想了非常长的时间才弄懂。

```python
def load_data(data_path, subject, training, all_trials=True):
  
    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6 * 48
    window_Length = 7 * 250

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0, a_trial.size):
            if (a_artifacts[trial] != 0 and not all_trials):
                continue
            data_return[NO_valid_trial, :, :] = np.transpose(
                a_X[int(a_trial[trial]):(int(a_trial[trial]) + window_Length), :22])
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1

    return data_return[0:NO_valid_trial, :, :], class_return[0:NO_valid_trial]

```

:person_frowning:然后在下面的get_data函数中设置了实际要取的数据长度也就是t1到t2，一共是1125，这里也就是论文中提到的1125*22，然后经过label的独热编码，数据的标准化归一化处理，最终得到结果：288,1125,22

```python
def get_data(path, subject, LOSO=False, isStandard=True):
    # Define dataset parameters
    fs = 250  # sampling rate
    t1 = int(1.5 * fs)  # start time_point
    t2 = int(6 * fs)  # end time_point
    T = t2 - t1  # length of the MI trial (samples or time_points)

    # Load and split the dataset into training and testing 
    if LOSO:
        # Loading and Dividing of the data set based on the 
        # 'Leave One Subject Out' (LOSO) evaluation approach. 
        X_train, y_train, X_test, y_test = load_data_LOSO(path, subject)
    else:
        path = path
        X_train, y_train = load_data(path, subject + 1, True)
        X_test, y_test = load_data(path, subject + 1, False)

    # Prepare training data    
    N_tr, N_ch, _ = X_train.shape
    X_train = X_train[:, :, t1:t2].reshape(N_tr, 1, N_ch, T)
    y_train_onehot = (y_train - 1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)
    # Prepare testing data 
    N_test, N_ch, _ = X_test.shape
    X_test = X_test[:, :, t1:t2].reshape(N_test, 1, N_ch, T)
    y_test_onehot = (y_test - 1).astype(int)
    y_test_onehot = to_categorical(y_test_onehot)

    # Standardize the data
    if (isStandard == True):
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot
```

### 网络模块部分详解

:nail_care:具体的论文的模块复述部分将在一个新的word附件中，这里只对网络的模型进行一个我自己的理解上的解释

***

#### 从总体上来说

:hourglass:从总体上来讲，模型大致可以这样理解：**ATCNet模型由三个主要块组成：卷积（CV）块、注意力（AT）块和时间卷积（TC）块，如图1所示。CV块通过三个卷积层（时间、信道深度和空间卷积）编码MI-EEG信号中的低层时空信息，其输出是具有更高级表示的时间序列。AT块然后使用多头自我关注（MSA）突出显示时间序列中最重要的信息。最后，TC块使用TCN提取时间序列中的高级时间特征，并将其馈送给具有SoftMax分类器的完全连接（FC）层。从CV块输出的时间序列可以被分割成多个滑动窗口，每个窗口分别被馈送到AT+TC。然后将所有窗口的输出连接起来并馈送到SoftMax分类器**。这有助于有效地扩充数据并提高准确性。ATCNet块的详细信息在以下小节中描述。另外，要强调的一点是实际输入到网络中的数据格式为：batch,1125,22,1也就是说高度和宽度为(1125,22),我们通常认为的在Conv2D卷积里面的通道特征维度是1

模型的整体流程：

<img src="https://cdn.jsdelivr.net/gh/lwlBCI/EEG-MI-ATCNet/images/image-20230205200002084.png" alt="image-20230205200002084" style="zoom:67%;" />

#### CV_BLOCK理解

<img src="https://cdn.jsdelivr.net/gh/lwlBCI/EEG-MI-ATCNet/images/image-20230205200032742.png" alt="image-20230205200032742" style="zoom:67%;" />

:watch:按照图中顺序和代码的内容来看，这部分：

1.首先是一个Conv2D卷积核大小为采样率的1/4也就四舍五入为64，即卷积核大小为(64,1)，步长默认为(1,1)并且这里的padding='same'，这意味着在输入为None，1125,22,1的情况下，输出的尺寸维度大小不会变仍然是1125*22，输出通道维度为F1=16，因此输出为None,1125,22,16,后跟BN层结构。

2.然后是一个Depthwise卷积层结构，卷积核大小为通道电极数目22，即卷积核大小为(1,22)，步长默认为(1,1)并且这里的depth_multiplier = D，这意味着在未指定输出通道情况下，输出通道数=输入 x D(也就是乘以2)，同样的输出的尺寸维度大小是1125*1，输出通道维度为32，因此输出为None,1125,1,32,后跟BN层结构+ELU激活。

3.然后是一个平均池化过程，池化卷积核大小为(8,1),这里尺寸维度变化了，变为了160*1,因此这里的输出为（None,160,1,32)，然后跟一个Dropout。

4.然后是一个Conv2D卷积(在上面提到的[参考22](https://www.bbsmax.com/A/KE5QD1oyJL/)中这里是Spatial Conv，我不清楚这里为啥要变成普通卷积)，卷积核大小为(16,1)，步长默认为(1,1)，输出通道数为32，同样这里的padding='same'，因此输出仍为（None,140,1,32)，后跟BN层结构+ELU激活。

5.最后是一个平均池化，池化卷积核大小为(7,1),这里尺寸维度变化了(在上面提到的[参考22](https://www.bbsmax.com/A/KE5QD1oyJL/)中这里是8），变为了20*22,因此这里的输出为（None,20,1,32)，然后跟一个Dropout。将最终的结果进行返回，但是在网络结构返回后，把（None,20,1,32)中的1去除了，也就是说输入到下一个block的结构为（None，20,32）。

:satellite:关于卷积的尺寸与池化的[尺寸计算](https://blog.csdn.net/qq_52254197/article/details/124951976),此模块附代码如下：

```python
def Conv_block(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1):
    """ Conv_block
    
        Notes
        -----
        This block is the same as EEGNet with SeparableConv2D replaced by Conv2D 
        The original code for this model is available at: https://github.com/vlawhern/arl-eegmodels
        See details at https://arxiv.org/abs/1611.08024
    """
    F2= F1*D  # 32
    block1 = Conv2D(F1, (kernLength, 1), padding = 'same',data_format='channels_last',use_bias = False)(input_layer)
    block1 = BatchNormalization(axis = -1)(block1)
    block2 = DepthwiseConv2D((1, in_chans), use_bias = False, 
                                    depth_multiplier = D,
                                    data_format='channels_last',
                                    depthwise_constraint = max_norm(1.))(block1)
    block2 = BatchNormalization(axis = -1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8,1),data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(F2, (16, 1),
                            data_format='channels_last',
                            use_bias = False, padding = 'same')(block2)
    block3 = BatchNormalization(axis = -1)(block3)
    block3 = Activation('elu')(block3)
    
    block3 = AveragePooling2D((poolSize,1),data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3
```

#### AT(注意力)结构

:loudspeaker:注意力模块中，这里用的是多头注意力机制也就是Multi heads attentions，并且用的是tf中自带的多头注意力函数，输入key_dim=8,num_heads=2,并且输入了两个x，代码如下：

```python
x = MultiHeadAttention(key_dim = key_dim, num_heads = num_heads, dropout = dropout)(x, x)
```

:lock:**其实我这里并不明白是怎么用的，特别是Q,K,V到底是怎么互相交互的？等到实际需要的时候再看吧。**另外还有一个点就是，这里在注意力机制这块也用到了捷径分支，所以就是要注意一下

![image-20230205200455190](https://cdn.jsdelivr.net/gh/lwlBCI/EEG-MI-ATCNet/images/image-20230205200455190.png)

捷径分支代码：

```python
mha_feature = Add()([input_feature, x])
```

#### TCN_Block结构

![image-20230205200608330](https://cdn.jsdelivr.net/gh/lwlBCI/EEG-MI-ATCNet/images/image-20230205200608330.png)

:postbox:上图是TCN结构的总括，包括两个TCN Residual Block，且每个TCN Residual Block内部都是残差结构。TCN结构其实早就有了，只不过这里是在EEG分类中用到了，实际上TCN结构的核心就是因果卷积与空洞(膨胀)卷积的结合，全称为一维时间卷积，推荐一篇详细解析TCN的[博客](https://www.cnblogs.com/zjuhaohaoxuexi/p/16813645.html)

:calling:CN的内部结构就是残差结构，如上图所示，先是一个因果、空洞卷积+BN、ELU，再来一个因果、空洞卷积+BN、ELU，然后采用捷径分支把原始的输入加上卷积后的。**而且特别是这里要强调的是，因果、空洞卷积并不会改变输入的尺寸，这是非常需要注意的一个点。另外还有一点就是，两个TCN在结合的时候，是一个一个来的，先经历第一个，然后通过Add添加捷径分支，然后是第二个，并且要非常注意的是第二个的空洞卷积膨胀系数为2，如果有第三个可能系数就是4了，系数的增加带来的是RFS也就是感受野的指数级递增**，这在博客里展示的很清楚，这里不再赘述。

#### 滑动卷积结构

:file_folder:滑动卷积结构比较新颖，我之前确实没见过，所以看了代码和查了一些资料之后我觉得这个滑动结构在这里的**作用是增加数据量的**，因为我们经过Conv_block获取了高级时间表示后的数据结构为：(None,20,32)这样的数据量确实比较少，所以呢，这里使用了滑动窗口利用循环将数据分为五个部分分别是：**(None,0:16,32)、(None,1:17,32)、(None,2:18,32)、(None,3:19,32)、(None,4:20,32)**,之所以是这么分的，是因为n_windows=5，所以这里分了五段，每段的感受野是16。

:file_folder:所以针对每个段(感受野为16)我们使用AT+TCN+4分类平坦层等进行处理，五个都处理完了以后进行tf.keras.layers.Average()，得到最终的输出结果，虽然在论文的图里画的是concat拼接，但是论文作者提供的源代码里用的是tf.keras.layers.Average()这也是我不懂的一个点。

:open_file_folder:所以写到这里，我们可以附上整个网络的流程代码：

```python
# 这部分代码是网络的整个流程
def ATCNet(n_classes, in_chans = 22, in_samples = 1125, n_windows = 3, attention = None, 
           eegn_F1 = 16, eegn_D = 2, eegn_kernelSize = 64, eegn_poolSize = 8, eegn_dropout=0.3, 
           tcn_depth = 2, tcn_kernelSize = 4, tcn_filters = 32, tcn_dropout = 0.3, 
           tcn_activation = 'elu', fuse = 'average'):
    """ ATCNet model from Altaheri et al 2022.
        See details at https://ieeexplore.ieee.org/abstract/document/9852687
    
        Notes
        -----
        The initial values in this model are based on the values identified by
        the authors
        
        References
        ----------
        .. H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed 
           attention temporal convolutional network for EEG-based motor imagery 
           classification," in IEEE Transactions on Industrial Informatics, 2022, 
           doi: 10.1109/TII.2022.3197419.
    """
    # 输入到模型的数据shape为(64,1,22,1125)
    input_1 = Input(shape = (1,in_chans, in_samples))   # TensorShape([None, 1, 22, 1125])
    input_2 = Permute((3,2,1))(input_1)  # 转为1125,22,1
    regRate=.25
    numFilters = eegn_F1 # 16
    F2 = numFilters*eegn_D # 32

    block1 = Conv_block(input_layer = input_2, F1 = eegn_F1, D = eegn_D, 
                        kernLength = eegn_kernelSize, poolSize = eegn_poolSize,
                        in_chans = in_chans, dropout = eegn_dropout)
    block1 = Lambda(lambda x: x[:,:,-1,:])(block1)  # 把其中的值为1的维度去除
     
    # Sliding window 
    sw_concat = []   # to store concatenated or averaged sliding window outputs
    for i in range(n_windows):
        st = i
        end = block1.shape[1]-n_windows+i+1
        block2 = block1[:, st:end, :]  # st:end的维度就是shape[1]，也就是Tc=20那个维度，st:end,一共是16个
        
        # Attention_model
        if attention is not None:
            block2 = attention_block(block2, attention)

        # Temporal convolutional network (TCN)
        block3 = TCN_block(input_layer = block2, input_dimension = F2, depth = tcn_depth,
                            kernel_size = tcn_kernelSize, filters = tcn_filters, 
                            dropout = tcn_dropout, activation = tcn_activation)
        # Get feature maps of the last sequence,获取最后一个序列的特征图,我不明白为什么只需要最后一个
        block3 = Lambda(lambda x: x[:,-1,:])(block3)
        
        # Outputs of sliding window: Average_after_dense or concatenate_then_dense
        if(fuse == 'average'):
            sw_concat.append(Dense(n_classes, kernel_constraint = max_norm(regRate))(block3))
        elif(fuse == 'concat'):
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])
                
    if(fuse == 'average'):
        if len(sw_concat) > 1: # more than one window

            sw_concat = tf.keras.layers.Average()(sw_concat[:])
        else: # one window (# windows = 1)
            sw_concat = sw_concat[0]
    elif(fuse == 'concat'):
        sw_concat = Dense(n_classes, kernel_constraint = max_norm(regRate))(sw_concat)
            
    
    softmax = Activation('softmax', name = 'softmax')(sw_concat)
    
    return Model(inputs = input_1, outputs = softmax)
```

#### 网络参数详细

![image-20230206110516454](https://cdn.jsdelivr.net/gh/lwlBCI/EEG-MI-ATCNet/images/image-20230206110516454.png)

#### 消融实验结果

![image-20230206110626522](https://cdn.jsdelivr.net/gh/lwlBCI/EEG-MI-ATCNet/images/image-20230206110626522.png)

:pushpin:**最终的独立于受试者的结果在文件夹的result中有，9个受试者的测试数据结果以混淆矩阵的形式给出**

### 如何运行

:blue_book:关于本文章的代码复现，在我个人的此仓库中已上传，可直接找到EEG-ATCNet-main文件夹中的main.py文件来运行，但要注意在运行此文件之前需要按照下面的信息更改main.py文件中的路径：

:notebook_with_decorative_cover:327行中的data_path改为使用者自己的BCI Competition IV-2a中的数据集路径

:ledger:329行改为使用者自己的保存结果的路径文件夹

:pencil2:修改完上述路径后，**可直接运行main.py文件(不需要运行preprocess.py)，代码默认执行ATCNet模型**。当然，您也可以修改run()函数中的334行代码中的train_conf,将'model':'ATCNet'进行修改，便可执行其他模型的运行。

### Cite

[1]Ingolfsson T M, Hersche M, Wang X, et al. EEG-TCNet: An accurate temporal convolutional network for embedded motor-imagery brain–machine interfaces[C]//2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC). IEEE, 2020: 2958-2965.

[2]Musallam Y K, AlFassam N I, Muhammad G, et al. Electroencephalography-based motor imagery classification using temporal convolutional network fusion[J]. Biomedical Signal Processing and Control, 2021, 69: 102826.

[3]Bai S, Kolter J Z, Koltun V. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling[J]. arXiv preprint arXiv:1803.01271, 2018.

[4]Altaheri H, Muhammad G, Alsulaiman M. Physics-Informed Attention Temporal Convolutional Network for EEG-Based Motor Imagery Classification[J]. IEEE Transactions on Industrial Informatics, 2022, 19(2): 2249-2258.