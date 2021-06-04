from utils.repPoints import multi_apply
from losses.assigner import *
from losses.utils import PseudoSampler





def point_target(proposals_list,
                 valid_flag_list,
                 gt_bboxes_list,
                 batch_shapes,
                 cfg,
                 gt_labels_list=None,
                 unmap_outputs=True):
    """Compute corresponding GT box and classification targets for proposals.
       作proposal和gt 之间的匹配
    Args:
        points_list (list(points_one_img),len=bs):   points_one_img (list(points_one_level),len=num_of_fpn)
                                                     points_one_level(torch.tensor): shape=[-1,3] 3==>x,y,stride 原图尺度

        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        cfg (dict): train sample configs.

    Returns:
        tuple
    """
    num_imgs = len(batch_shapes)
    assert len(proposals_list) == len(valid_flag_list) == num_imgs

    # points number of each level, len=num_fpn
    num_level_proposals = [points.size(0) for points in proposals_list[0]]

    # concat all level points and flags to a single tensor
    # proposals_list(len=bs): list(proposals_one_img) proposals_one_img=[num_all_points,3]
    # valid_flag_list(len=bs): list(flags_one_img)    flags_one_img=[num_all_points,1]
    for i in range(num_imgs):
        assert len(proposals_list[i]) == len(valid_flag_list[i])
        proposals_list[i] = torch.cat(proposals_list[i])  # [num_all_points,3]
        valid_flag_list[i] = torch.cat(valid_flag_list[i])  # [num_all_points,1]

    # compute targets for each image
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]

    '''
    all_labels (list,len=bs):  list(labels)  labels(torch.tensor): shape=[num_all_points,1]  正样本位置为对应的gt label(注意: 取值范围[0,79])， 负样本和无效样本位置为0
    all_label_weights (list,len=bs):  list(label_weights)  label_weights(torch.tensor): shape=[num_all_points,1] 正样本位置为对应的pos_weight(=1 here)， 负样本位置为1， 无效样本位置为0
    all_bbox_gt (list,len=bs):  list(bbox_gt)  bbox_gt(torch.tensor): shape=[num_all_points,4] 正样本位置为对应的gt_box， 负样本和无效样本位置为0
    all_proposals (list,len=bs): list(proposals)  proposals(torch.tensor): shape=[num_all_points,3/4] 正样本位置为对应的proposal， 负样本和无效样本位置为0
    all_proposal_weights (list,len=bs):  list(proposals_weight)  proposals_weight(torch.tensor): shape=[num_all_points,4] 正样本位置为对应的1， 负样本和无效样本位置为0
    
    pos_inds_list (list,len=bs):  list(pos_inds)  positive proposal index
    neg_inds_list (list,len=bs):  list(neg_inds)  negative proposal index
    '''
    (all_labels, all_label_weights, all_bbox_gt, all_proposals, all_proposal_weights, pos_inds_list,neg_inds_list) = multi_apply(
                point_target_single,
                proposals_list,
                valid_flag_list,
                gt_bboxes_list,
                gt_labels_list,
                cfg=cfg,
                unmap_outputs=unmap_outputs)
    # no valid points
    if any([labels is None for labels in all_labels]):
        return None

    # sampled points of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])  # 正样本的个数
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])  # 副样本的个数

    labels_list = images_to_levels(all_labels, num_level_proposals)
    label_weights_list = images_to_levels(all_label_weights, num_level_proposals)
    bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
    proposals_list = images_to_levels(all_proposals, num_level_proposals)
    proposal_weights_list = images_to_levels(all_proposal_weights, num_level_proposals)

    '''
    labels_list (list,len=num_fpn_layer):  list(bs_lvl_labels)  bs_lvl_labels(torch.tensor): shape=[bs,num_lvl_proposals]  正样本位置为对应的gt label(注意: 取值范围[0,79])， 负样本和无效样本位置为0
    label_weights_list (list,len=num_fpn_layer):  list(bs_lvl_label_weights)  bs_lvl_label_weights(torch.tensor): shape=[bs,num_lvl_proposals]   正样本位置为对应的pos_weight(=1 here)， 负样本位置为1， 无效样本位置为0
    bbox_gt_list (list,len=num_fpn_layer):  list(bs_lvl_box_gt)  bs_lvl_box_gt(torch.tensor): shape=[bs,num_lvl_proposals,4]  正样本位置为对应的gt_box， 负样本和无效样本位置为0
    proposals_list (list,len=num_fpn_layer):  list(bs_lvl_proposals)  bs_lvl_proposals(torch.tensor): shape=[bs,num_lvl_proposals,4]  正样本位置为对应的proposal， 负样本和无效样本位置为0
    proposal_weights_list (list,len=num_fpn_layer):  list(bs_lvl_proposal_weights)  bs_lvl_proposal weights(torch.tensor): shape=[bs,num_lvl_proposals,4]   正样本位置为对应的1， 负样本和无效样本位置为0

    num_total_pos(int): num of positive proposals in the batch
    num_total_neg(int): num of negative proposals in the batch
    '''
    return (labels_list, label_weights_list, bbox_gt_list, proposals_list, proposal_weights_list, num_total_pos, num_total_neg)




def images_to_levels(target, num_level_grids):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]

    :return
    level_targets(list,len=num_fpn_layer):  list(bs_lvl_targets)  bs_lvl_targets(torch.tensor): shape=[bs,num_lvl_targets]
    """
    target = torch.stack(target, 0)  # shape=[bs,num_targets]
    level_targets = []
    start = 0
    for n in num_level_grids:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets





def build_assigner(cfg):
    if cfg['type'] == 'PointAssigner':
        return PointAssigner(scale=cfg['scale'],
                             pos_num=cfg['pos_num'])
    elif cfg['type'] == 'MaxIoUAssigner':
        return MaxIoUAssigner(pos_iou_thr=cfg['pos_iou_thr'],
                              neg_iou_thr=cfg['neg_iou_thr'],
                              min_pos_iou=cfg['min_pos_iou'],
                              gt_max_assign_all=True)
    else:
        NotImplementedError(cfg['type']+' is not supported now !')



def point_target_single(flat_proposals,
                        valid_flags,
                        gt_bboxes,
                        gt_labels,
                        cfg,
                        unmap_outputs=True):
    '''

    :param flat_proposals(torch.tensor):  shape=[num_all_proposal,3/4] 3==>(x,y,stride)  4==>(x1,y1,x2,y2)
    :param valid_flags(torch.tensor):     shape=[num_all_proposal,1]   =1 valid,  =0 unvalid
    :param gt_bboxes(torch.tensor):       shape=[num_gt,4]   4==>(x1,y1,x2,y2) 输入尺度
    :param gt_labels(torch.tensor):       shape=[num_gt,]
    :param cfg(dict):
    :param unmap_outputs(bool):  True
    :return:
    '''
    inside_flags = valid_flags
    if not inside_flags.any():
        return (None,) * 7
    # assign gt and sample proposals
    proposals = flat_proposals[inside_flags, :]


    bbox_assigner = build_assigner(cfg['assigner'])
    assign_result = bbox_assigner.assign(proposals, gt_bboxes, gt_labels)

    bbox_sampler = PseudoSampler()
    sampling_result = bbox_sampler.sample(assign_result, proposals, gt_bboxes)


    num_valid_proposals = proposals.shape[0]


    # bbox_gt pos_proposals 相同位置存放对应的gt proposal
    bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
    pos_proposals = torch.zeros_like(proposals)
    proposals_weights = proposals.new_zeros([num_valid_proposals, 4])

    # labels/label_weights.shape=[num_valid_proposals]
    labels = proposals.new_zeros(num_valid_proposals, dtype=torch.long)
    label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)

    pos_inds = sampling_result.pos_inds  # positive proposal index
    neg_inds = sampling_result.neg_inds  # negative proposal index

    '''
    bbox_gt: 正样本位置为对应的gt_box， 负样本位置为0
    pos_proposals： 正样本位置为对应的proposal， 负样本位置为0
    proposals_weights: 正样本位置为对应的1， 负样本位置为0
    
    labels:    正样本位置为对应的gt label(注意: 取值范围[0,79])， 负样本位置为0, not care为-1
    label_weights:   正样本位置为对应的pos_weight(=1 here)， 负样本位置为1, not care为0
    '''
    if len(pos_inds) > 0:
        pos_gt_bboxes = sampling_result.pos_gt_bboxes
        bbox_gt[pos_inds, :] = pos_gt_bboxes
        pos_proposals[pos_inds, :] = proposals[pos_inds, :]
        proposals_weights[pos_inds, :] = 1.0  # positive proposal位置weight=1
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg['pos_weight'] <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg['pos_weight']
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0


    '''
    bbox_gt: 正样本位置为对应的gt_box， 负样本和无效样本位置为0
    pos_proposals： 正样本位置为对应的proposal， 负样本和无效样本位置为0
    proposals_weights: 正样本位置为对应的1， 负样本和无效样本位置为0

    labels:    正样本位置为对应的gt label(注意: 取值范围[0,79])， 负样本和无效样本位置为0， not care为-1
    label_weights:   正样本位置为对应的pos_weight(=1 here)， 负样本位置为1， 无效样本位置为0, not care为0
    '''
    # map up to original set of proposals
    # 映射到未过滤掉无效prososal的变量中去
    if unmap_outputs:
        num_total_proposals = flat_proposals.size(0)
        labels = unmap(labels, num_total_proposals, inside_flags)
        label_weights = unmap(label_weights, num_total_proposals, inside_flags)
        bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
        pos_proposals = unmap(pos_proposals, num_total_proposals, inside_flags)
        proposals_weights = unmap(proposals_weights, num_total_proposals,inside_flags)

    return (labels, label_weights, bbox_gt, pos_proposals, proposals_weights, pos_inds, neg_inds)




def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count)
    """
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
