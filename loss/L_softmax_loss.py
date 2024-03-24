import torch
import torch.nn as nn

class LSoftMaxLoss(nn.Module):
    def __init__(self, num_classes, batch_size =64,margin=0.5, scale=30,use_gpu=True):
        super(LSoftMaxLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.theta = nn.Parameter(torch.Tensor(batch_size, 768))
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.theta = nn.Parameter(torch.Tensor(batch_size, 768).cuda())
        else:
            self.theta = nn.Parameter(torch.Tensor(batch_size, 768))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, input, target,one_hot,distance_scale):
        batch_size = input.size(0)
        cos_theta = self.cosine_similarity(input, self.theta)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)  # 防止arccos中的NaN
        phi_theta = cos_theta - self.margin
        # one_hot = torch.zeros_like()
        # one_hot.scatter_(0, target.view(1, -1).expand_as(one_hot), 1)
        # import ipdb;ipdb.set_trace()
        # 增大某两类之间的距离
        
        phi_theta = phi_theta.unsqueeze(1).expand_as(one_hot)
        cos_theta = cos_theta.unsqueeze(1).expand_as(one_hot)
        # phi_theta[:, 0] -= distance_scale
        # phi_theta[:, 2] += distance_scale
        phi_theta[:, 6] -= distance_scale
        phi_theta[:, 9] += distance_scale
        output = (one_hot * phi_theta) + ((1.0 - one_hot) * cos_theta)
        # output = (phi_theta * one_hot) + (cos_theta * (1.0 - one_hot))
        output *= self.scale
        # import ipdb;ipdb.set_trace()


        log_probs = nn.functional.log_softmax(output, dim=1)
        # log_probs_class_grey = log_probs[:,0]
        # log_probs_class_dark_brown = log_probs[:,2]
        # log_probs_class_grey = log_probs[:,6]
        # log_probs_class_dark_brown = log_probs[:,9]
        loss = nn.functional.nll_loss(log_probs, target)
        # loss = nn.functional.nll_loss(log_probs_class_grey - log_probs_class_dark_brown,torch.zeros_like(target))

        return loss
    
if __name__ == '__main__':
    # 创建LSoftMaxLoss实例
    num_classes = 4
    margin = 0.5
    scale = 10
    lsoftmax_loss = LSoftMaxLoss(num_classes=4, batch_size=5,margin=0.5, scale=10,use_gpu=False)

    # 准备输入数据和目标标签
    input = torch.randn(5, 10)  # 输入数据，形状为 (batch_size, input_dim)
    target = torch.tensor([0, 1, 2, 3,0])  # 目标标签，形状为 (batch_size,)
    one_hot = nn.functional.one_hot(target, num_classes)

    print(one_hot)
    # import ipdb;ipdb.set_trace()

    # 调用LSoftMaxLoss的forward方法计算损失
    loss = lsoftmax_loss(input, target,one_hot,distance_scale= 3)

    # 打印损失值
    print(loss)