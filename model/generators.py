import paddle
import paddle.nn as nn

import functools



class UNetGenerator(nn.Layer):
    """
    基于Unet网络的生成器
    从pixe2pixe的论文和代码上看，只是类似
    """

    def __init__(self, input_nc=3, output_nc=3, num_downs=7, ngf=64, embedding_num=40, embedding_dim=128,
                 norm_layer=nn.BatchNorm2D, use_dropout=False):
        """
        Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UNetGenerator, self).__init__()
        # construct unet structure

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, embedding_dim=embedding_dim)  # add the innermost layer
        #网络的最底层模块，chanel已经比较宽了，是比较直接的BN+CNN+RELU的结构，这个时候通道已经非常多
        for _ in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer
        self.embedder = nn.Embedding(embedding_num, embedding_dim)# 这一个是对于分类的

    def forward(self, x, style_or_label=None):
        """Standard forward"""
        #if style_or_label is not None and 'LongTensor' in style_or_label.type:
        if style_or_label is not None and hasattr(style_or_label.type,"LOD_TENSOR"):
            return self.model(x, self.embedder(style_or_label))
        else:
            return self.model(x, style_or_label)


class UnetSkipConnectionBlock(nn.Layer):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
        看了pix2pix的实现代码才发现，不是完全参考的Unet，而是对uNet做了裁剪，所以这块还是直接复用代码吧
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, embedding_dim=128, norm_layer=nn.BatchNorm2D,
                 use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules=子模块
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        PIXE2PIXE的源代码中并没有embedding_dim，这个参数，应该是用来预测类型使用的。
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost #这块儿什么意思啊？
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2D
        else:
            use_bias = norm_layer == nn.InstanceNorm2D
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2D(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias_attr=use_bias)# 下采样了，宽度变一半
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)# 上下采样只是做了一次bn操作

        if outermost:
            upconv = nn.Conv2DTranspose(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)# 反卷积操作
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]

        elif innermost:
            upconv = nn.Conv2DTranspose(inner_nc + embedding_dim, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias_attr=use_bias)
            down = [downrelu, downconv]# 执行一次下采样，channel变大宽度变大
            up = [uprelu, upconv, upnorm]# 然后再执行一次上采样，上采样的情况

        else:
            upconv = nn.Conv2DTranspose(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias_attr=use_bias)# 之后再看一下装置卷积的内容
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up = up + [nn.Dropout(0.5)]

        self.submodule = submodule
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)

    def forward(self, x, style=None):
        if self.innermost:
            encode = self.down(x)
            if style is None:
                return encode
            enc = paddle.concat([style.reshape((style.shape[0],style.shape[1], 1, 1)), encode], 1)# 把样式的编码和encode结合到一起，然后在进行解码，所以就需要知道，style的通道数目，用来计算上采样的通道数
            dec = self.up(enc)
            # 最底下网络模块，把X和经过上采样和下采样的内容concat起来
            return paddle.concat([x, dec], 1), encode.reshape((x.shape[0], -1))
        elif self.outermost:
            enc = self.down(x)
            if style is None:
                return self.submodule(enc)
            sub, encode = self.submodule(enc, style)
            dec = self.up(sub)
            return dec, encode
        else:  # add skip connections
            enc = self.down(x)
            if style is None:
                return self.submodule(enc)
            sub, encode = self.submodule(enc, style)
            dec = self.up(sub)
            return paddle.concat([x, dec], 1), encode


if __name__ == '__main__':
    gNet=UNetGenerator()
    input_data = {'x1': paddle.rand([1, 3, 128, 128]),
              'x2': paddle.randint(low=0,high=5,shape=[1, 2])}
    #(lenet_multi_input, [(1, 1, 28, 28), (1, 400)], 
    #                                    dtypes=['float32', 'float32'])
    paddle.summary(gNet,[(1, 3, 128, 128)],dtypes=["float32"])
    paddle.summary(gNet,[(1, 3, 128, 128),(1,)],dtypes=["float32","int32"])