data dir: `/NASdata/AudioData/voxceleb/voxceleb1`

# CNN与声纹专题
## 基于ResNet的声纹
### ResNet

- torchvision中的resnet有什么问题？
- residual connection && BasicBlock && BottlenetBlock
- 参数估计 (conv 1x1的作用)
- batch norm / layer norm
- 变体与ResNet结构的理解加深
- 感受野3x3 2层 > 5*5 1层好

### 基于ResNet的声纹 `/mingback/linqj/voxceleb`

    model: Input -> ResNet -> statistics pooling -> Linear1 -> (Dropout) -> Linear2
    shape: T * D -> C * H * W -> 2C -> dim1 -> ---- -> dim2

- 输入特征：mfcc, fbank, mfcc+pitch ...
- 打包成batch时, 不同音频长度T不相等, 怎么处理: 比如 collate
- 调参: pytorch-template
- 如何保证网络可复现性
- 打分策略: cosine, PLDA

### 计算指标
EER, minDCF,...
vox1_train训练, vox1_test测试时EER低于4%即为合格

## TDNN x-vector
论文: https://www.danielpovey.com/files/2018_icassp_xvectors.pdf
TDNN == 一维CNN?
指标: EER 6%~7%即为合格
