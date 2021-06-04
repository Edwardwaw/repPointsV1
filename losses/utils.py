import torch


class AssignResult(object):
    """
    Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (int): the number of truth boxes considered when computing this assignment
        gt_inds (LongTensor): for each predicted box indicates the 1-based
                                index of the assigned truth box. 0 means unassigned and -1 means ignore.
        max_overlaps (FloatTensor): the iou between the predicted box and its assigned truth box.
        labels (None | LongTensor): If specified, for each predicted box indicates the category label of the assigned truth box.

    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels

    @property
    def num_preds(self):
        """
        Return the number of predictions in this assignment
        """
        return len(self.gt_inds)

    @property
    def info(self):
        """
        Returns a dictionary of info about the object
        """
        return {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
        }




#######################################################################################################################

class SamplingResult(object):
    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags):
        '''
        :param pos_inds: positive bbox index
        :param neg_inds: negative bbox index
        :param bboxes:
        :param gt_bboxes:
        :param assign_result:
        :param gt_flags: torch.zeros(bboxes.shape[0])
        '''
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1   #pos sample对应的gt box index(0 based here)

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]  # pos sample对应的gt box

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]   # pos sample对应的gt box的label id [0,79] 注意： 负样本的也是0
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])

    def to(self, device):
        """
        Change the device of the data inplace.
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, torch.Tensor):
                _dict[key] = value.to(device)
        return self


    @property
    def info(self):
        """
        Returns a dictionary of info about the object
        """
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_bboxes': self.pos_bboxes,
            'neg_bboxes': self.neg_bboxes,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }





########################################################################################################################


class PseudoSampler(object):
    def sample(self, assign_result, bboxes, gt_bboxes):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()  # positive proposal index
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()  # negative proposal index
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)  # shape=[num_proposals,]
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        return sampling_result























