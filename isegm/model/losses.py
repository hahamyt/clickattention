import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from isegm.utils import misc
import math

# import visdom
# viz = visdom.Visdom()


class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        #print(pred.shape, label.shape)
        pred = pred.float()
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)


        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
        sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)



class DiversityLoss(nn.Module):
    def __init__(self):
        super(DiversityLoss, self).__init__()
        self.baseloss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
        self.click_loss = ClickLoss() #WFNL(alpha=0.5, gamma=2, w = 0.99)


    def forward(self, latent_preds, label, click_map):
        div_loss_lst = []
        click_loss = 0
        for i in range(latent_preds.shape[1]):
            single_pred = latent_preds[:,i,:,:].unsqueeze(1)
            single_loss = self.baseloss(single_pred,label)
            single_loss = single_loss.unsqueeze(-1)
            div_loss_lst.append(single_loss)
            click_loss += self.click_loss(single_pred,label,click_map)

        div_losses = torch.cat(div_loss_lst,1)
        div_loss_min = torch.min(div_losses,dim=1)[0]
        return div_loss_min.mean() + click_loss.mean()



class WFNL(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, w = 0.5, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(WFNL, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0
        self.w = w

    def forward(self, pred, label, weight = None):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if weight is not None:
            weight = weight * self.w + (1-self.w)
            loss = (loss * weight).sum() / (weight.sum() + self._eps)
        else:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
        sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)

class FocalLoss(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0,
                 ignore_label=-1):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            tsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(label.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss


class SoftIoU(nn.Module):
    def __init__(self, from_sigmoid=False, ignore_label=-1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label

        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)

        loss = 1.0 - torch.sum(pred * label * sample_weight, dim=(1, 2, 3)) \
            / (torch.sum(torch.max(pred, label) * sample_weight, dim=(1, 2, 3)) + 1e-8)

        return loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label
                     + torch.log(1. - pred + eps) * (1. - label))

        loss = self._weight * (loss * sample_weight)
        return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))


class WeightedSigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(WeightedSigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label, weight):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label
                     + torch.log(1. - pred + eps) * (1. - label))
        #weight = weight * 0.8 + 0.2
        loss = (weight * loss).sum() / weight.sum() 
        return loss #torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))




class ClickLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1, alpha = 0.99, beta = 0.01):
        super(ClickLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis
        self.alpha = alpha
        self.beta = beta


    def forward(self, pred, label, gaussian_maps = None):
        h_gt, w_gt = label.shape[-2],label.shape[-1]
        h_p, w_p = pred.shape[-2], pred.shape[-1]
        if h_gt != h_p or w_gt != w_p:
            pred  = F.interpolate(pred, size=label.size()[-2:],
                                                 mode='bilinear', align_corners=True)


        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label
                     + torch.log(1. - pred + eps) * (1. - label))

        loss = self._weight * (loss * sample_weight)
        weight_map = gaussian_maps.max(dim=1,keepdim = True)[0] * self.alpha + self.beta
        loss = (loss * weight_map).sum() / weight_map.sum()
        return loss



class AttWeightLoss(nn.Module):
    def __init__(self, attn_weight, scale=1):
        super(AttWeightLoss, self).__init__()
        self.critiation = nn.MSELoss(reduction='none')
        self.attn_weight = attn_weight
        self.scale = scale

    def forward(self, pred, gt, pos_click_len):
        # 筛选掉点击次数为1的图片
        mask = [x <= 1 for x in pos_click_len]

        gt64 = F.interpolate(gt, scale_factor=0.25 / self.scale)
        gt32 = F.interpolate(gt, scale_factor=0.125 / self.scale)
        gt16 = F.interpolate(gt, scale_factor=0.0625 / self.scale)
        gt8 = F.interpolate(gt, scale_factor=0.03125 / self.scale)
        gts = [gt64, gt32, gt16, gt8]
        gts = [gts[i] for i in range(4) if self.attn_weight[i]]

        loss = 0.0
        for i in range(len(pred)):
            tmp_loss = 0.0
            tmp_loss = tmp_loss + self.critiation(pred[i], gts[i])
            tmp_loss[mask, :, :] = 0.0 * tmp_loss[mask, :, :]     # 点击次数等于1次的不计算损失
            loss = loss + torch.mean(tmp_loss)

        return loss/len(gts)



class DiscriminateiveAffinityLoss(nn.Module):
    def __init__(self, class_num=None, loss_index=[False,False,False,False], scale=1):
        super(DiscriminateiveAffinityLoss, self).__init__()
        self.class_num = class_num
        self.scale = scale

        self.stages = [i for (i, v) in enumerate(loss_index) if v == True]  # 0,1,2,3
        self.L1 = nn.L1Loss(reduction='none')
        self.sizes = [128//scale, 64//scale, 32//scale, 16//scale]         # 对应512
        # self.sizes = [192//scale, 96//scale, 48//scale, 24//scale]            # 对应768
        # self.sizes = [64//scale, 32//scale, 16//scale, 8//scale]              # 对应256
        # self.sizes = [112//scale, 56//scale, 28//scale, 14//scale]            # 对应448

    def forward(self, seg, attns, attnWs, pos_click_len):
        # 筛选掉点击次数为1的图片, 次数为0的给他的损失设置为0
        mask = [x <= 1 for x in pos_click_len]

        seg = torch.sigmoid(seg)

        loss = 0.0

        for idx, i in enumerate(self.stages):
            loss_tmp = 0.0
            fore_seg = F.interpolate(seg, size=self.sizes[i], mode='bilinear', align_corners=True).flatten(2).permute(
                [0, 2, 1])  # 将预测结果插值到与Attention的尺寸一样

            back_seg = 1 - fore_seg
            attn = torch.stack(attns[i], dim=1)
            attn = attn.mean(dim=1)  # 层平均
            attn = attn.mean(dim=1)  # 头平均

            attnW = attnWs[idx].flatten(2).permute(0, 2, 1).detach()
            attn_sel_pos = torch.mul(attn, attnW)
            attn_sel_neg = torch.mul(attn, 1 - attnW)

            fore_seg_scale = F.interpolate(seg, size=int(math.sqrt(attn.shape[-1])), mode='bilinear', align_corners=True).flatten(
                2).permute([0, 2, 1])  # 将预测结果插值到与Attention的尺寸一样
            back_seg_scale = 1 - fore_seg_scale

            loss_tmp = loss_tmp + self.L1(torch.matmul(attn_sel_pos, fore_seg_scale), fore_seg)
            loss_tmp = loss_tmp + self.L1(torch.matmul(attn_sel_neg, back_seg_scale), back_seg)
            loss_tmp[mask, :, :] = 0.0 * loss_tmp[mask, :, :]
            loss  = loss + torch.mean(loss_tmp)

        # loss = loss / (2 * len(self.stages))

        return loss/len(self.stages)
