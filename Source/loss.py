import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd

class Losses(object):
    def __init__(self, classify_size_average=True, args_gpu=0):
        self.classify_size_average = classify_size_average
        self.args_gpu = args_gpu
        self.kernel_5 = torch.ones(2, 1, 5, 5).cuda(self.args_gpu)
        self.kernel_1_5 = torch.ones(1, 1, 5, 5).cuda(self.args_gpu)
        self.kernel = torch.ones(2, 1, 3, 3).cuda(self.args_gpu)
        self.kernel_1_3 = torch.ones(1, 1, 3, 3).cuda(self.args_gpu)
        # self.lambda_ = 0.1
        self.lambda_ = 0.5

        self.matrices_2 = torch.full((1024, 960), 2, dtype=torch.float).cuda(self.args_gpu)
        self.matrices_0 = torch.full((1024, 960), 0, dtype=torch.float).cuda(self.args_gpu)


    def loss_fn_v6v8_compareLSC(self, input_, target, outputs_classify, labels_classify, size_average=False):
        input = input_.clone()
        n, c, h, w = input_.size()

        n_fore = torch.sum(labels_classify)
        # n_fore = labels_classify[labels_classify == 1].size()[0]
        labels_classify_ = labels_classify.unsqueeze(1)
        l_c = labels_classify_.expand(n, c, h, w).float()
        # input[l_c == 0] = 0
        input = input * l_c

        i_t = target - input
        loss_l1 = torch.sum(torch.abs(i_t)) / (n_fore*2)
        
        l_c_outputs_classify = torch.nn.functional.sigmoid(outputs_classify).data.round().unsqueeze(1).expand(n, c, h, w)
        input = input_ * l_c_outputs_classify
        loss_CS = 1 + torch.sum(-((torch.sum(target*input, dim=1))/(torch.norm(target, dim=1)*torch.norm(input, dim=1)+1e-10))) / n_fore

        mask = F.conv2d(labels_classify_, self.kernel_1_3, padding=1)
        mask[mask < 9] = 0
        mask_9 = mask / 9
        loss_local = torch.sum(torch.abs(F.conv2d(i_t, self.kernel, padding=1, groups=2)*mask_9 - i_t*mask)) / (torch.sum(mask_9)*2)

        return loss_l1*0.1, loss_local*0.1, loss_CS*0.5
      
    def loss_fn_binary_cross_entropy_with_logits(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target, size_average=self.classify_size_average)

