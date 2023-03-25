import paddle
import paddle.nn as nn
#from torch.nn import init
import functools
#from torch.optim import lr_scheduler
import math


class Discriminator(nn.Layer):
    """
    判别器模型
    """

    def __init__(self, in_num_channel, embedding_num, out_num_channel=64, norm_layer=nn.BatchNorm2D, image_size=256):
        """
        判别器模型
        Parameters:
            input_nc (int)  -- 输入通道数
            ndf (int)       -- the number of filters in the first conv layer 输出通道数
            norm_layer      -- normalization layer
            embedding_num   用于计算类别损失，应该是生成字体的类别？
        """
        super(Discriminator, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2D
        else:
            use_bias = norm_layer != nn.BatchNorm2D

        # as tf implement, kernel_size = 5, use "SAME" padding, so we should use kw = 5 and padw = 2
        # kw = 4
        # padw = 1
        kw = 5
        padw = 2
        sequence = [
            nn.Conv2D(in_num_channel, out_num_channel, kernel_size=kw, stride=2, padding=padw),#
            nn.LeakyReLU(0.2, True)
        ]# 大小不变
        nf_mult = 1
        nf_mult_prev = 1
        # in tf implement, there are only 3 conv2d layers with stride=2.
        # for n in range(1, 4):
        for n in range(1, 3):  # gradually increase the number of filters
            in_num_channel=out_num_channel
            out_num_channel=out_num_channel*min(2 ** n, 8)
            # nf_mult_prev = nf_mult
            # nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2D(in_num_channel, out_num_channel, kernel_size=kw, stride=2, padding=padw, bias_attr=use_bias),
                norm_layer(out_num_channel),# BN层，需要再查一下资料，BN层可以避免梯度消失
                nn.LeakyReLU(0.2, True)
            ]# 大小减半
        # 网络逐步变宽
        in_num_channel=out_num_channel
        out_num_channel=in_num_channel*8
        # nf_mult_prev = nf_mult
        # nf_mult = 8
        sequence += [
            nn.Conv2D(in_num_channel, out_num_channel, kernel_size=kw, stride=1, padding=padw, bias_attr=use_bias),
            norm_layer(out_num_channel),
            nn.LeakyReLU(0.2, True)
        ]

        # Maybe useful? Experiment need to be done later.
        # output 1 channel prediction map
        sequence += [nn.Conv2D(out_num_channel, 1, kernel_size=kw, stride=1, padding=padw)]
        # 现在通道为1了，就是一张2D的图片了
        self.model = nn.Sequential(*sequence)
        # final_channels = ndf * nf_mult
        final_channels = 1
        # use stride of 2 conv2 layer 3 times, cal the image_size
        image_size = math.ceil(image_size / 2)# 下取整运算吗？
        image_size = math.ceil(image_size / 2)
        image_size = math.ceil(image_size / 2)
        # 524288 = 512(num_of_channels) * (w/2/2/2) * (h/2/2/2) = 2^19  (w=h=256)
        # 131072 = 512(num_of_channels) * (w/2/2/2) * (h/2/2/2) = 2^17  (w=h=128)
        final_features = final_channels * image_size * image_size # 这个应该是计算了出来最终输出图像的维度，然后完全线性化操作。
        self.binary = nn.Linear(final_features, 1)
        self.prob =nn.Sigmoid()
        self.catagory = nn.Linear(final_features, embedding_num)# 计算分类的损失，
        self.soft_max_catagory=nn.Softmax()
        

    def forward(self, input):
        """Standard forward."""
        # features = self.model(input).view(input.shape[0], -1)
        features = self.model(input)
        features = features.reshape((input.shape[0], -1))
        binary_logits = self.binary(features)
        catagory_logits = self.soft_max_catagory(self.catagory(features))
        return self.prob(binary_logits), catagory_logits
        #return binary_logits


if __name__ == '__main__':
    dNet=Discriminator(1,10,64,image_size=128)
    paddle.summary(dNet,(1,1,128,128))
    network_like="""
        ---------------------------------------------------------------------------
        Layer (type)       Input Shape          Output Shape         Param #    
        ===========================================================================
        Conv2D-1      [[1, 1, 128, 128]]    [1, 64, 64, 64]         1,664     
        LeakyReLU-1    [[1, 64, 64, 64]]     [1, 64, 64, 64]           0       
        Conv2D-2      [[1, 64, 64, 64]]     [1, 128, 32, 32]       204,800    
        BatchNorm2D-1   [[1, 128, 32, 32]]    [1, 128, 32, 32]         512      
        LeakyReLU-2    [[1, 128, 32, 32]]    [1, 128, 32, 32]          0       
        Conv2D-3      [[1, 128, 32, 32]]    [1, 512, 16, 16]      1,638,400   
        BatchNorm2D-2   [[1, 512, 16, 16]]    [1, 512, 16, 16]        2,048     
        LeakyReLU-3    [[1, 512, 16, 16]]    [1, 512, 16, 16]          0       
        Conv2D-4      [[1, 512, 16, 16]]   [1, 4096, 16, 16]     52,428,800   
        BatchNorm2D-3  [[1, 4096, 16, 16]]   [1, 4096, 16, 16]       16,384     
        LeakyReLU-4   [[1, 4096, 16, 16]]   [1, 4096, 16, 16]          0       
        Conv2D-5     [[1, 4096, 16, 16]]     [1, 1, 16, 16]        102,401    
        Linear-1          [[1, 256]]             [1, 1]              257      
        Linear-2          [[1, 256]]            [1, 10]             2,570     
        ===========================================================================
        Total params: 54,397,836
        Trainable params: 54,388,364
        Non-trainable params: 9,472
        ---------------------------------------------------------------------------
        Input size (MB): 0.06
        Forward/backward pass size (MB): 34.00
        Params size (MB): 207.51
        Estimated Total Size (MB): 241.58
        ---------------------------------------------------------------------------
        """