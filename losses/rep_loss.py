import numpy as np
import torch
from torch import nn
from losses.commons import *
from losses.point_target import point_target
from utils.repPoints import multi_apply



class RepLoss(object):
    def __init__(self,
                 num_class=80,
                 num_points=9,
                 point_strides=[8,16,32,64,128],
                 point_base_scale=4,
                 transform_method='moment',
                 moment_mul=0.01,
                 alpha=0.25,
                 gamma=2.0,
                 cls_weight=1.0,
                 beta_init=1.0/9.0,
                 loss_init_weight=0.5,
                 beta_refine=1.0/9.0,
                 loss_refine_weight=1.0
                 ):
        self.num_class=num_class
        self.num_points=num_points
        self.point_strides=point_strides
        self.point_base_scale = point_base_scale


        self.loss_cls=FocalLoss(alpha,gamma,cls_weight)
        self.loss_bbox_init=SmoothL1Loss(beta_init,loss_init_weight)
        self.loss_bbox_refine=SmoothL1Loss(beta_refine,loss_refine_weight)


        self.transform_method = transform_method
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul

        ### for label assign
        self.point_generators = [PointGenerator() for _ in self.point_strides]  # len=num_of_fpn



    def points2bbox(self, pts, y_first=True):
        """
        Converting the points set into bounding box.

        :param pts: the input points sets (fields), each points set (fields) is represented as 2n scalar. shape=[bs,num_points*2,h,w]
        :param y_first: if y_fisrt=True, the point set is represented as [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                        represented as [x1, y1, x2, y2 ... xn, yn].
        :return:
        each points set is converting to a bbox [x1, y1, x2, y2].
        bbox(torch.tensor): shape=[bs,4,h,w]    4==>x1,y1,x2,y2
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])  # shape=[bs,num_points,2,h,w]
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1, ...]  # shape=[bs,num_points,h,w]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0, ...]  # shape=[bs,num_points,h,w]

        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)  # shape=[bs,4,h,w]

        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]  # 仅取前四个预测
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)

        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                        self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ], dim=1)

        else:
            raise NotImplementedError
        return bbox




    def get_points(self, featmap_sizes, batch_shapes, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple],len=num_of_fpn): Multi-level feature map sizes.  (h,w)
            batch_shapes (list,len=bs): image size before batch padding   (w,h)

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(batch_shapes)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute points center for one time
        # list(points_one_level)  len=num_of_fpn
        # points_one_level(torch.tensor): shape=[-1,3]     -1=feat_h*feat*w    3=>(x,y,stride) 对应原图尺度
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(featmap_sizes[i], self.point_strides[i], device)
            multi_level_points.append(points)

        # points_list=list(multi_level_points)
        points_list = [[point.clone() for point in multi_level_points] for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_shape in enumerate(batch_shapes):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                w, h = img_shape  # image shape before batch padding
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w), device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list



    def centers_to_bboxes(self, point_list):
        """
        Get bboxes according to center points.
        Only used in MaxIOUAssigner.

        :param
        points_list (list(points_one_img),len=bs):   points_one_img (list(points_one_level),len=num_of_fpn)  points_one_level(torch.tensor): shape=[-1,3] 3==>x,y,stride 原图尺度
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5  # 4*stride/2
                bbox_shift = torch.Tensor([-scale, -scale, scale,
                                           scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat([point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)  # shape=[-1,4] 4==>(x,y,x,y)原图尺度
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list



    def offset_to_pts(self, center_list, pred_list):
        """
        Change from point offset to point coordinate.
        把预测的offset转换为对应的point坐标,坐标对应输入尺度

        center_list (list(points_one_img),len=bs):   points_one_img (list(points_one_level),len=num_of_fpn)  points_one_level(torch.tensor): shape=[-1,3] 3==>x,y,stride 原图尺度
        pred_list(len=num_fpn_layer, list(pts_pred_init)):   pts_pred_init(torch.tensor) shape=[bs,num_points*2,h,w] 在对应featuremap上的预测  2==>y,x

        :return
        pts_list(len=num_fpn, list(pts_lvl)):  pts_lvl(torch.tensor):  shape=[bs,h*w,num_points*2]  2==>x,y
        """
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                # 网格点/center point
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(1, self.num_points)  #  shape=[h*w,2*num_points]
                # 对应点的offset预测/ predicted offset
                pts_shift = pred_list[i_lvl][i_img]  # shape=[num_points*2,h,w]
                yx_pts_shift = pts_shift.permute(1, 2, 0).contiguous().view(-1, 2 * self.num_points)  #shape=[hw,2*num_points]

                y_pts_shift = yx_pts_shift[..., 0::2] # shape=[h*w,num_points]
                x_pts_shift = yx_pts_shift[..., 1::2] # shape=[h*w,num_points]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)  # shape=[h*w,num_points,2]  2==>x_shift,y_shift
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)  # shape=[h*w,2*num_point]
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center   # shape=[h*w,num_point*2]  原图尺度
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)  # shape=[bs,h*w,num_points*2]
            pts_list.append(pts_lvl)
        return pts_list



    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine,
                    labels, label_weights, bbox_gt_init, bbox_weights_init,
                    bbox_gt_refine, bbox_weights_refine,
                    stride, num_total_samples_init, num_total_samples_refine):
        '''
        模型预测值
        :param cls_scores(torch.tensor) shape=[bs,num_cls,h,w]
        :param pts_pred_init(torch.tensor):  shape=[bs,h*w,num_points*2] 2==>x,y  把对应的预测offset转为对应的输入尺度上的坐标点即repPoint
        :param pts_pred_refine(torch.tensor):  shape=[bs,h*w,num_points*2] 2==>x,y  把对应的预测offset转为对应的输入尺度上的坐标点即repPoint

        对应的label/target
        :param labels(torch.tensor): shape=[bs,num_lvl_proposals]   正样本元素取值[0,79]， 负样本和无效样本位置为0， not care为-1
        :param label_weights(torch.tensor): shape=[bs,num_lvl_proposals]  正样本位置为对应的pos_weight(=1 here)， 负样本位置为1， 无效样本位置为0, not care为0

        :param bbox_gt_init(torch.tensor): shape=[bs,num_lvl_proposals,4]  正样本元素取值=gt_box， 其他=0
        :param bbox_weights_init(torch.tensor): shape=[bs,num_lvl_proposals,4]   正样本元素取值=1， 其他=0

        :param bbox_gt_refine(torch.tensor): shape=[bs,num_lvl_proposals,4]  正样本元素取值=gt_box， 其他=0
        :param bbox_weights_refine(torch.tensor): shape=[bs,num_lvl_proposals,4]   正样本元素取值=1， 其他=0

        :return:
        '''

        ### regression loss --------------------------------------------------------------------------------------------
        # 1st stage
        bbox_gt_init = bbox_gt_init.contiguous().view(-1, 4)
        bbox_weights_init = bbox_weights_init.contiguous().view(-1, 4)
        bbox_pred_init = self.points2bbox(pts_pred_init.contiguous().view(-1, 2 * self.num_points),y_first=False)  # the predicted box in first stage
        pos_index_1=(bbox_weights_init[:,0]>0)

        # 2nd stage(refine stage)
        bbox_gt_refine = bbox_gt_refine.contiguous().view(-1, 4)
        bbox_weights_refine = bbox_weights_refine.contiguous().view(-1, 4)
        bbox_pred_refine = self.points2bbox(pts_pred_refine.contiguous().view(-1, 2 * self.num_points), y_first=False)
        pos_index_2=(bbox_weights_refine[:,0]>0)


        normalize_term = self.point_base_scale * stride
        loss_pts_init = self.loss_bbox_init(bbox_pred_init[pos_index_1] / normalize_term,
                                            bbox_gt_init[pos_index_1] / normalize_term,
                                            bbox_weights_init[pos_index_1],
                                            avg_factor=num_total_samples_init)
        loss_pts_refine = self.loss_bbox_refine(bbox_pred_refine[pos_index_2] / normalize_term,
                                                bbox_gt_refine[pos_index_2] / normalize_term,
                                                bbox_weights_refine[pos_index_2],
                                                avg_factor=num_total_samples_refine)

        ### classification loss ----------------------------------------------------------------------------------------
        labels = labels.contiguous().view(-1)
        label_weights = label_weights.contiguous().view(-1)
        cls_score = torch.sigmoid(cls_score.permute(0, 2, 3, 1).contiguous().view(-1, self.num_class))

        pos_flag = (bbox_weights_refine[:,0]>0)
        valid_flag = (label_weights>0)

        valid_label_weights=label_weights[valid_flag]  # 正样本1 负样本1
        valid_cls_scores=cls_score[valid_flag]  # shape=[num_valid,num_cls]

        cls_targets=torch.zeros_like(cls_score)
        cls_targets[pos_flag,labels[pos_flag]]=1.
        valid_cls_targets=cls_targets[valid_flag]
        # print('debug1',labels[pos_flag])
        # print('debug2',torch.nonzero(valid_cls_targets!=0,as_tuple=False)[:,1])


        loss_cls = self.loss_cls(valid_cls_scores,
                                 valid_cls_targets,
                                 valid_label_weights,
                                 avg_factor=num_total_samples_refine)

        return loss_cls, loss_pts_init, loss_pts_refine



    def __call__(self, cls_scores,
                 pts_preds_init,
                 pts_preds_refine,
                 batch_targets,
                 batch_shapes,
                 cfg):
        '''

        :param cls_scores(len=num_fpn_layer, list(cls_score)):   cls_score(torch.tensor) shape=[bs,num_cls,h,w]
        :param pts_preds_init(len=num_fpn_layer, list(pts_pred_init)):   pts_pred_init(torch.tensor) shape=[bs,num_points*2,h,w] 在对应featuremap上的预测  2==>y,x
        :param pts_preds_refine(len=num_fpn_layer, list(pts_pred_refine)):   pts_pred_refine(torch.tensor) shape=[bs,num_points*2,h,w] 在对应featuremap上的预测  2==>y,x
        :param batch_targets:   [N,6]  6==>(batch_id,label_id,x1,y1,x2,y2)  输入尺度
        :param batch_shapes:  list(img_shape_before_batchpadding)  (w,h)
        :param cfg: assigner_cfg
        :return:
        '''
        for i in range(len(cls_scores)):
            if cls_scores[i].dtype == torch.float16:
                cls_scores[i] = cls_scores[i].float()

        device = cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        gt_bboxes = list()
        gt_labels = list()
        for i in range(cls_scores[0].shape[0]):
            idx = batch_targets[:,0] == i
            if idx.sum() == 0:
                gt_bboxes.append(torch.zeros(0,4).type_as(batch_targets))
                gt_labels.append(None)
            else:
                gt_bboxes.append(batch_targets[idx,2:])
                gt_labels.append(batch_targets[idx,1].to(torch.long))
        assert len(featmap_sizes) == len(self.point_generators)


        # target for initial stage--------------------------------------------------------------------------------------------------------------------------------------------------
        # 返回有效(剔除batchpadding中的填充部分)的featuremap grid point坐标(对应原图尺度)及其stride
        # center_list (list(points_one_img),len=bs):   points_one_img (list(points_one_level),len=num_of_fpn)  points_one_level(torch.tensor): shape=[-1,3] 3==>x,y,stride 原图尺度
        # valid_flag_list (list(flags_one_img),len=bs):  flags_one_img (list(flags_one_level),len=num_of_fpn)   flags_one_level(torch.tensor): shape=[-1,]  =1 valid point, =0 unvalid point
        center_list, valid_flag_list = self.get_points(featmap_sizes, batch_shapes, device)

        # 把对应的预测offset转为对应的输入尺度上的坐标点即repPoint
        # pts_list(len=num_fpn, list(pts_lvl)): pts_lvl(torch.tensor):  shape=[bs,h*w,num_points*2] 2==>x,y
        pts_coordinate_preds_init = self.offset_to_pts(center_list, pts_preds_init)


        if cfg['init']['assigner']['type'] == 'PointAssigner':
            # Assign target for center list
            '''
            candidate_list (list(points_one_img),len=bs):   points_one_img (list(points_one_level),len=num_of_fpn)  
                                                                        points_one_level(torch.tensor): shape=[-1,3] 3==>x,y,stride 原图尺度
            '''
            candidate_list = center_list
        else:
            # transform center list to bbox list and assign target for bbox list
            '''
            candidate_list (list(boxes_one_img),len=bs):   boxes_one_img (list(boxes_one_level),len=num_of_fpn)  
                                                                       boxes_one_level(torch.tensor): shape=[-1,4] 4==>(x1,y1,x2,y2) 原图尺度
            '''
            bbox_list = self.centers_to_bboxes(center_list)
            candidate_list = bbox_list

        cls_reg_targets_init = point_target(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            batch_shapes,
            cfg['init'],
            gt_labels_list=gt_labels)


        '''
        注意第一次point regression并不关注类别
        bbox_gt_list_init(list,len=num_fpn_layer):  list(bs_lvl_box_gt)  bs_lvl_box_gt(torch.tensor): shape=[bs,num_lvl_proposals,4]  正样本位置为对应的gt_box， 负样本和无效样本位置为0
        candidate_list_init(list,len=num_fpn_layer):  list(bs_lvl_proposals)  bs_lvl_proposals(torch.tensor): shape=[bs,num_lvl_proposals,4]  正样本位置为对应的proposal， 负样本和无效样本位置为0
        bbox_weights_list_init (list,len=num_fpn_layer):  list(bs_lvl_proposal_weights)  bs_lvl_proposal weights(torch.tensor): shape=[bs,num_lvl_proposals,4]   正样本位置为对应的1， 负样本和无效样本位置为0
        num_total_pos_init(int): num of positive proposals in the batch
        num_total_neg_init(int): num of negative proposals in the batch
        '''
        (*_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init, num_total_pos_init, num_total_neg_init) = cls_reg_targets_init
        num_total_samples_init = num_total_pos_init


        ##### build target for refinement stage -----------------------------------------------------------------------------------------------
        # 返回有效点的坐标及其stride
        # center_list (list(points_one_img),len=bs):   points_one_img (list(points_one_level),len=num_of_fpn)  points_one_level(torch.tensor): shape=[-1,3] 3==>x,y,stride 原图尺度
        # valid_flag_list (list(flags_one_img),len=bs):  flags_one_img (list(flags_one_level),len=num_of_fpn)   flags_one_level(torch.tensor): shape=[-1,]  =1 valid point, =0 unvalid point
        center_list, valid_flag_list = self.get_points(featmap_sizes, batch_shapes, device)

        # 把对应的预测offset转为对应的输入尺度上的坐标点即repPoint
        # pts_list(len=num_fpn, list(pts_lvl)): pts_lvl(torch.tensor):  shape=[bs,h*w,num_points*2] 2==>x,y
        pts_coordinate_preds_refine = self.offset_to_pts(center_list, pts_preds_refine)


        bbox_list = []
        for i_img, center in enumerate(center_list):
            # center: (list(points_one_level),len=num_of_fpn)  points_one_level(torch.tensor): shape=[-1,3] 3==>x,y,stride 原图尺度
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):
                '''
                input: pts_preds_init[i_lvl].shape=[bs,num_points*2,h,w]
                return: bbox_preds_init (torch.tensor):  shape=[bs,4,h,w] 4==>x1,y1,x2,y2
                '''
                bbox_preds_init = self.points2bbox(pts_preds_init[i_lvl].detach())
                bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                bbox_center = torch.cat([center[i_lvl][:, :2], center[i_lvl][:, :2]],dim=1)  # shape=[num_points_one_level,4]
                # [num_points_level,4]+[num_points_one_level,4]=[num_points_one_level,4]  4==>x1,y1,x2,y2 原图尺度
                # 根据第一次的预测结果生成第二次refine所需要的anchor,为标签分配作准备
                bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).contiguous().view(-1, 4))
            bbox_list.append(bbox)  # list(box_per_img)  box_per_img=list(box_per_level)   box_per_level: [num_points_one_level,4]  4==>x,y,x,y原图尺度

        # 为第一次regression得出的结果分配真值
        cls_reg_targets_refine = point_target(
            bbox_list,
            valid_flag_list,
            gt_bboxes,
            batch_shapes,
            cfg['refine'],
            gt_labels_list=gt_labels)

        (labels_list, label_weights_list, bbox_gt_list_refine, candidate_list_refine, bbox_weights_list_refine, num_total_pos_refine, num_total_neg_refine) = cls_reg_targets_refine
        num_total_samples_refine = num_total_pos_refine

        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(self.loss_single,
                                                                     cls_scores,
                                                                     pts_coordinate_preds_init,
                                                                     pts_coordinate_preds_refine,
                                                                     labels_list,
                                                                     label_weights_list,
                                                                     bbox_gt_list_init,
                                                                     bbox_weights_list_init,
                                                                     bbox_gt_list_refine,
                                                                     bbox_weights_list_refine,
                                                                     self.point_strides,
                                                                     num_total_samples_init=num_total_samples_init,
                                                                     num_total_samples_refine=num_total_samples_refine)

        cls_loss = torch.sum(torch.stack(losses_cls))
        init_pts_loss=torch.sum(torch.stack(losses_pts_init))
        refine_pts_loss=torch.sum(torch.stack(losses_pts_refine))
        total_loss = cls_loss+init_pts_loss+refine_pts_loss
        # print(cls_loss,init_pts_loss,refine_pts_loss)

        # return cls_loss+init_pts_loss+refine_pts_loss,  torch.stack([cls_loss, init_pts_loss, refine_pts_loss]).detach(), num_total_pos_refine
        return total_loss, torch.stack([cls_loss, init_pts_loss, refine_pts_loss]).detach(), num_total_pos_refine


    ############# functions below is used for inference ----------------------------------------------------------------
    def get_bboxes(self,
                   cls_scores,
                   pts_preds_refine):
        '''
        注意： pts是在对应的featuremap尺度上作出的预测,即关于featuremap上某点(grid_x,grid_y)为原点时的对应尺度上的offset预测结果
        :param cls_scores(len=5): list(cls_out)  cls_out.shape=[bs,num_cls,h,w]
        :param pts_preds_refine(len=5):  list(pts_refine_out)  pts_init_out.shape=[bs,num_points*2,h,w]  2=>y,x
        :return:
        predict_list(len=num_of_fpn): list(predicts)  predicts(torch.Tensor):  shape=[bs,h*w,4+num_cls]
        '''

        assert len(cls_scores) == len(pts_preds_refine)
        num_levels = len(cls_scores)
        bs = cls_scores[0].shape[0]
        device=cls_scores[0].device
        # 生成在对应的特征层尺度上的box offset预测
        # len=num_of_fpn,list(box) box.shape=[bs,4,h,w]   4==>x1,y1,x2,y2
        bbox_preds_refine = [self.points2bbox(pts_pred_refine)
                             for pts_pred_refine in pts_preds_refine]

        '''
        len=num_of_fpn
        list(points_per_level),  points_per_level.shape=[feat_h*feat_w,3]  3==>(x,y,stride)
                                                              x y 对应原图尺度,  stride (x,y)对应的stride
        '''
        mlvl_points = [self.point_generators[i].grid_points(cls_scores[i].size()[-2:],self.point_strides[i],device)
                       for i in range(num_levels)]

        predict_list=[]

        for cls_score_per_level,box_pred_per_level,points_per_level,stride in \
                zip(cls_scores,bbox_preds_refine,mlvl_points,self.point_strides):
            '''
            cls_score_per_level(torch.Tensor): shape=[bs,num_cls,h,w]
            box_pred_per_level(torch.Tensor): shape=[bs,4,h,w]  生成在对应的特征层尺度上的box offset预测
            points_per_level(torch.Tensor): shape=[h*w,3]  3==>(x,y,stride)   xy对应原图尺度,  stride (x,y)对应的stride
            stride(int): 对应特征层的stride
            '''
            cls_score_per_level = cls_score_per_level.permute(0,2,3,1).contiguous()\
                .reshape(bs,-1,self.num_class).sigmoid()
            # shape=[bs,h*w,4]
            box_pred_per_level = box_pred_per_level.permute(0,2,3,1).contiguous().reshape(bs,-1, 4)
            # shape=[bs,h*w,3]
            points_per_level=points_per_level.expand(size=(bs,points_per_level.shape[0],points_per_level.shape[1]))
            # shape=[bs,h*w,4]
            box_pos_center_per_level = torch.cat([points_per_level[..., :2], points_per_level[..., :2]], dim=2)
            box_per_level = box_pred_per_level * stride + box_pos_center_per_level
            predict_out = torch.cat([box_per_level,cls_score_per_level],dim=-1)
            predict_list.append(predict_out)

        return predict_list


























