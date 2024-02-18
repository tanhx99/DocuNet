import torch
import torch.nn as nn
import torch.nn.functional as F


def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e30
    y_pred_pos = y_pred - (1 - y_true) * 1e30
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros],dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros),dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


class balanced_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):

        loss = multilabel_categorical_crossentropy(labels,logits)
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = torch.zeros_like(logits[..., :1])
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output[:,1:].sum(1) == 0.).to(logits)

        return output


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output




class NCRLoss(nn.Module):
    def __init__(self, shift=0.0, isReg=True, eps=1e-8, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.isReg = isReg
        self.eps = eps
        self.reduction = reduction

    def compute_CE(self, x, y):
        """
        Adapted from "Asymmetric Loss For Multi-Label Classification"
        https://arxiv.org/abs/2009.14119
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Margin Shifting
        if self.shift is not None and self.shift > 0:
            xs_neg = (xs_neg + self.shift).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        return -loss.sum()

    def forward(self, logits, labels):
        # Logit margin for pre-defined relations 
        rel_margin = logits[:,1:] - logits[:,0].unsqueeze(1)
        loss = self.compute_CE(rel_margin.float(), labels[:,1:].float())
        
        if self.isReg: # Enable margin regularization
            # Logit margin for the none class label
            na_margin = logits[:,0] - logits[:,1:].mean(-1) 
            loss += self.compute_CE(na_margin.float(), labels[:,0].float())
        
        if self.reduction == "mean":
            loss /= labels.shape[0]

        return loss

    def get_label(self, logits, num_labels=-1):
        """Copied from https://github.com/wzhouad/ATLOP/blob/main/losses.py#L32"""
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output
