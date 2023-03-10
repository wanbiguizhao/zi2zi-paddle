import paddle
import paddle.nn as nn
from paddle.nn import initializer as init

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if gpu_ids:
        assert(paddle.fluid.is_compiled_with_cuda())
        net.to(gpu_ids[0])
        #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.Normal(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.XavierNormal(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    def init_func_paddle(layer:nn.Layer):  # define the initialization function
        # ???????????????????????????????????????????????????
        if  type(layer) in [nn.Conv2D,nn.Linear,nn.Conv2DTranspose ]:
            weight_attr = paddle.framework.ParamAttr(
                        name="linear_weight",
                        initializer=paddle.nn.initializer.Normal(mean=0.0, std=init_gain))

            bias_attr = paddle.framework.ParamAttr(
                name="linear_bias",
                initializer=paddle.nn.initializer.Constant(0.0))
            weight_attr.initializer(layer.weight)
            if hasattr(layer,"bias") and layer.bias is not None:
                bias_attr.initializer(layer.bias)
            ##layer.weight.set_value(weight_attr)
            #layer.bias.set_value(bias_attr)
            return
        elif type(layer)==nn.BatchNorm2D:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            weight_attr = paddle.framework.ParamAttr(
                        name="linear_weight",
                        initializer=paddle.nn.initializer.Normal(mean=1.0, std=init_gain))
            bias_attr = paddle.framework.ParamAttr(
                name="linear_bias",
                initializer=paddle.nn.initializer.Constant(0.0))
            weight_attr.initializer(layer.weight)
            bias_attr.initializer(layer.bias)
    print('initialize network with %s' % init_type)
    net.apply(init_func_paddle)  # apply the initialization function <init_func>