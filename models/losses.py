import paddle
import paddle.nn as nn


class CategoryLoss(nn.Layer):
    def __init__(self, category_num):
        super(CategoryLoss, self).__init__()
        emb = nn.Embedding(category_num, category_num)
        emb.weight.data = paddle.eye(category_num)
        self.emb = emb
        self.loss = nn.BCEWithLogitsLoss()# 因为不是2分类问题，所以要使用BCEloss

    def forward(self, category_logits, labels):
        target = self.emb(labels)
        return self.loss(category_logits, target)


class BinaryLoss(nn.Layer):
    def __init__(self, real):
        super(BinaryLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.real = real

    def forward(self, logits):
        if self.real:
            labels = paddle.ones((logits.shape[0], 1))
        else:
            labels = paddle.zeros((logits.shape[0], 1))
        # if logits.place.is_cuda: 这块儿代码的意思是，把数据从gpu内存中拷贝出来，然后在cpu计算，todo：以后再支持cpu运行吧
        #     labels = labels.cuda()
        return self.bce(logits, labels)
