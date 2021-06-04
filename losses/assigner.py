import torch
from commons.boxs_utils import box_iou
from losses.utils import AssignResult


######################### first stage sample assigner ---------------------------------------------------------------------
class PointAssigner(object):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer indicating the ground truth index.

    - 0: negative sample, no assigned gt(0代表负样本)
    - positive integer: positive sample, index (1-based 从1开始的) of assigned gt (》=1 代表索引从1开始算的gt index)

    """

    def __init__(self, scale=4, pos_num=3):
        self.scale = scale
        self.pos_num = pos_num

    def assign(self, points, gt_bboxes, gt_labels=None):
        """Assign gt to points.

        This method assign a gt bbox to every points set, each points set will be assigned with  0, or a positive number.
        0 means negative sample, positive number is the index (1-based) of assigned gt.


        The assignment is done in following steps, the order matters.
        1. assign every points to 0
        2. A point is assigned to some gt bbox if
            (i) the point is within the k closest points to the gt bbox
            (ii) the distance between this point and the gt is smaller than other gt bboxes

        Args:
            points (Tensor): points to be assigned, shape(n, 3) while last dimension stands for (x, y, stride).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """

        if points.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')

        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(points_stride).int()  # [3, 4, 5, 6, 7]
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()
        num_gts, num_points = gt_bboxes.shape[0], points.shape[0]

        # assign gt box
        gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
        scale = self.scale   # 4
        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) + torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        # shape=[num_gt,] 元素取值[3,7]  表明gt_box应当由哪一层的points来预测
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        # stores the assigned gt index of each point
        assigned_gt_inds = points.new_zeros((num_points, ), dtype=torch.long)
        # stores the assigned gt dist (to this point) of each point
        assigned_gt_dist = points.new_full((num_points, ), float('inf'))
        points_range = torch.arange(points.shape[0])


        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]
            # get the index of points in this level
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]
            # get the points in this level
            lvl_points = points_xy[lvl_idx, :]
            # get the center point of gt
            gt_point = gt_bboxes_xy[[idx], :]
            gt_wh = gt_bboxes_wh[[idx], :]
            # compute the distance between gt center and all points in this level
            points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1) # shape=[lvl_points,1]
            # find the nearest k points to gt center in this level
            min_dist, min_dist_index = torch.topk(points_gt_dist, self.pos_num, largest=False)
            # the index of nearest k points to gt center in this level
            min_dist_points_index = points_index[min_dist_index]
            # The less_than_recorded_index stores the index of min_dist that is less then the assigned_gt_dist. Where
            # assigned_gt_dist stores the dist from previous assigned gt (if exist) to each point.
            less_than_recorded_index = min_dist < assigned_gt_dist[min_dist_points_index]
            # The min_dist_points_index stores the index of points satisfy:
            #   (1) it is k nearest to current gt center in this level.
            #   (2) it is closer to current gt center than other gt center.
            min_dist_points_index = min_dist_points_index[less_than_recorded_index]
            # assign the result
            assigned_gt_inds[min_dist_points_index] = idx + 1
            assigned_gt_dist[min_dist_points_index] = min_dist[less_than_recorded_index]

        '''
        对于point对应的target index,存在如下情况.   index=0  negative, index>=1 positive (gt_box的索引从1开始算， index-1为真实索引)
        对于point对应的label id,存在如下情况. label id取值范围[0,79],负样本对应的label_id为0,正样本对应的label_id为[0,79],所以正样本的
                                           label id要和target index结合起来看  
        '''

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_points, ))
            pos_inds = torch.nonzero(assigned_gt_inds>0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)





######################## refine stage sample assigner ---------------------------------------------------------------------------------


class MaxIoUAssigner(object):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a positive bbox.
                             Positive samples can have smaller IoU than pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same highest overlap with some gt to that gt.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=0.0,
                 gt_max_assign_all=True
                 ):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all

    def assign(self, bboxes, gt_bboxes, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor),
        each bbox will be assigned with -1, 0, or a positive number.
        -1 means don't care, 0 means negative sample, positive number is the index (1-based) of assigned gt.

        The assignment is done in following steps, the order matters.
        1. assign every bbox to -1(means not care)
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr, assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).  4==>x1,y1,x2,y2原图尺度
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """

        bboxes = bboxes[:, :4]
        overlaps = box_iou(gt_bboxes, bboxes)  # iou_matrix.shape=[num_gt,num_box]

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes, shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_zeros((num_bboxes,), dtype=torch.long)
            return AssignResult(
                            num_gts,
                            assigned_gt_inds,
                            max_overlaps,
                            labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        # shape=[num_box]
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        # shape=[num_gt]
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0]) & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1  # 1 based

        # 4. assign fg: for each gt, proposals with highest IoU
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1
        '''
        对于box对应的target index,存在如下情况.  index=-1 not care, index=0  negative, index>=1 positive (gt_box的索引从1开始算， index-1为真实索引)
        对于point对应的label id,存在如下情况. label id取值范围[0,79],负样本和not care对应的label_id为0,正样本对应的label_id为[0,79],所以正样本的
                                           label id要和target index结合起来看  
        '''

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes,))
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)




