import torch
import math


########################################################################################################################
class BoxSimilarity(object):
    def __init__(self, iou_type="giou", coord_type="xyxy", eps=1e-9):
        self.iou_type = iou_type
        self.coord_type = coord_type
        self.eps = eps

    def __call__(self, box1, box2):
        """
        :param box1: [num,4] predicts
        :param box2:[num,4] targets
        :return:
        """
        box1_t = box1.T
        box2_t = box2.T

        if self.coord_type == "xyxy":
            b1_x1, b1_y1, b1_x2, b1_y2 = box1_t[0], box1_t[1], box1_t[2], box1_t[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2_t[0], box2_t[1], box2_t[2], box2_t[3]
        elif self.coord_type == "xywh":
            b1_x1, b1_x2 = box1_t[0] - box1_t[2] / 2., box1_t[0] + box1_t[2] / 2.
            b1_y1, b1_y2 = box1_t[1] - box1_t[3] / 2., box1_t[1] + box1_t[3] / 2.
            b2_x1, b2_x2 = box2_t[0] - box2_t[2] / 2., box2_t[0] + box2_t[2] / 2.
            b2_y1, b2_y2 = box2_t[1] - box2_t[3] / 2., box2_t[1] + box2_t[3] / 2.
        elif self.coord_type == "ltrb":
            b1_x1, b1_y1 = 0. - box1_t[0], 0. - box1_t[1]
            b1_x2, b1_y2 = 0. + box1_t[2], 0. + box1_t[3]
            b2_x1, b2_y1 = 0. - box2_t[0], 0. - box2_t[1]
            b2_x2, b2_y2 = 0. + box2_t[2], 0. + box2_t[3]
        else:
            raise NotImplementedError("coord_type only support xyxy, xywh,ltrb")
        inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                     (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union_area = w1 * h1 + w2 * h2 - inter_area + self.eps
        iou = inter_area / union_area
        if self.iou_type == "iou":
            return iou

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if self.iou_type == "giou":
            c_area = cw * ch + self.eps
            giou = iou - (c_area - union_area) / c_area
            return giou

        diagonal_dis = cw ** 2 + ch ** 2 + self.eps
        center_dis = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                      (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        if self.iou_type == 'diou':
            diou = iou - center_dis / diagonal_dis
            return diou

        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / ((1 + self.eps) - iou + v)

        if self.iou_type == "ciou":
            ciou = iou - (center_dis / diagonal_dis + v * alpha)
            return ciou

        raise NotImplementedError("iou_type only support iou,giou,diou,ciou")


class IOULoss(object):
    def __init__(self, iou_type="giou", coord_type="xyxy"):
        super(IOULoss, self).__init__()
        self.iou_type = iou_type
        self.box_similarity = BoxSimilarity(iou_type, coord_type)

    def __call__(self, predicts, targets):
        similarity = self.box_similarity(predicts, targets)
        if self.iou_type == "iou":
            return -similarity.log()
        else:
            return 1 - similarity

########################################################################################################################

class FocalLoss(object):
    def __init__(self, alpha=0.25, gamma=2.0, loss_weight=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def __call__(self, predicts, targets, weight, avg_factor):
        mask=torch.isnan(predicts.log())
        pos_loss=torch.FloatTensor(predicts.size()).type_as(predicts)
        pos_loss[mask]=torch.abs(targets[mask]-predicts[mask])
        neg_mask=torch.bitwise_not(mask)
        pos_loss[neg_mask]=-self.alpha * targets[neg_mask] * ((1 - predicts[neg_mask]) ** self.gamma) * (predicts[neg_mask].log())

        neg_loss = torch.FloatTensor(predicts.size()).type_as(predicts)
        mask = torch.isnan((1 - predicts).log())
        neg_mask=torch.bitwise_not(mask)
        neg_loss[mask]=torch.abs((1.-targets[mask])-(1.-predicts[mask]))
        neg_loss[neg_mask]=-(1 - self.alpha) * (1. - targets[neg_mask]) * (predicts[neg_mask] ** self.gamma) * ((1 - predicts[neg_mask]).log())

        loss = pos_loss + neg_loss
        loss *= weight.view(-1, 1)

        return self.loss_weight * loss.sum() / avg_factor


########################################################################################################################

class SmoothL1Loss(object):
    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def __call__(self, pred, target, weight, avg_factor):
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta, diff - 0.5 * self.beta)
        loss *= weight
        return self.loss_weight * loss.sum() / avg_factor



########################################################################################################################


class PointGenerator(object):

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_points(self, featmap_size, stride=16, device='cuda'):
        '''

        :param featmap_size:
        :param stride:
        :param device:
        :return:
        all_points(torch.tensor):   shape=[feat_h*feat_w,3]  3==>(x,y,stride)
                                    x y 原图尺度   stride (x,y)对应的stride
        '''
        feat_h, feat_w = featmap_size
        # shift_x shift_y 原图尺度
        shift_x = torch.arange(0., feat_w, device=device) * stride
        shift_y = torch.arange(0., feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)   # shape=[feat_h*feat_w]=[-1]
        stride = shift_x.new_full((shift_xx.shape[0], ), stride)  # shape=[-1]

        shifts = torch.stack([shift_xx, shift_yy, stride], dim=-1)
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        '''

        :param featmap_size:
        :param valid_size:
        :param device:
        :return:
        valid:  shape=[feat_h*feat_w,]
        '''
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid


########################################################################################################################


















