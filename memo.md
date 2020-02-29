# TODO

- MNIST
- Mask – UAR 0.67
- 尝试使用不同的网络结构
    1. 基本： 参考torchvision里的各种实现，ResNet，DensetNet，Inception； 
    2. 进阶： 自己实现or找github，CNN-LSTM结构，一维CNN结构SineNet，LCNN，Attention，Transformer等等
- 尝试使用不同的特征
    1. 基本： 用现有的包提特征包括MFCC，LFCC，Fbank，CQT，Spectrogram
    2. 进阶：自己提特征，或者加入一些Kaldi提出的特征，比如说i-vector
- 尝试把特征embedding层提出来，接上后端的分类器，比如SVM
- 进阶：尝试不同的损失函数。分类，回归
https://www.isca-speech.org/archive/Interspeech_2019/
查看ComParE，ASVspoof Challenge的下面的文章

/mingback/wuhw/new_code/compare2020/mask_code/

Breathing，难度比较大，感兴趣的先把上面的做好
