import torch
import math
import numpy as np
from torch import nn
from nets.common import FPN, CGR
# from torchvision.ops import DeformConv2d
from mmcv.ops import DeformConv2d
from utils.repPoints import multi_apply
from losses.rep_loss import RepLoss




def switch_backbones(bone_name):
    from nets.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, \
        resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
    if bone_name == "resnet18":
        return resnet18()
    elif bone_name == "resnet34":
        return resnet34()
    elif bone_name == "resnet50":
        return resnet50()
    elif bone_name == "resnet101":
        return resnet101()
    elif bone_name == "resnet152":
        return resnet152()
    elif bone_name == "resnext50_32x4d":
        return resnext50_32x4d()
    elif bone_name == "resnext101_32x8d":
        return resnext101_32x8d()
    elif bone_name == "wide_resnet50_2":
        return wide_resnet50_2()
    elif bone_name == "wide_resnet101_2":
        return wide_resnet101_2()
    else:
        raise NotImplementedError(bone_name)




class RepPointsHead(nn.Module):
    """
    RepPoint head.

    Args:
        num_cls(int): should not include background class
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        point_feat_channels (int): Number of channels of points features.
        stacked_convs (int): How many conv layers are used.
        gradient_mul (float): The multiplier to gradients from points refinement and recognition.
        point_base_scale (int): bbox scale for assigning labels.
        use_grid_points (bool): If we use bounding box representation,
                                the reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3, # the num of conv used before localization subnet and classification subnet
                 num_points=9,
                 gradient_mul=0.1,  # backpropagation 过程中梯度的乘积因子
                 point_base_scale=4,   # label assign时box的大小
                 use_grid_points=False,
                 center_init=True,
                ):
        super(RepPointsHead, self).__init__()
        self.in_channels = in_channels
        self.cls_out_channels = num_classes
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.num_points = num_points
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.use_grid_points = use_grid_points   # False
        self.center_init = center_init           # True

        # use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))  # 3
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)  # 1
        assert self.dcn_kernel * self.dcn_kernel == num_points, "The points number should be a square number."
        assert self.dcn_kernel % 2 == 1, "The points number should be an odd square number."

        dcn_base = np.arange(-self.dcn_pad,self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        # dcn_base_offset = [-1. -1. -1.  0. -1.  1.  0. -1.  0.  0.  0.  1.  1. -1.  1.  0.  1.  1.]
        # shape=(18,)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)   # shape=[1,18,1,1]

        self._init_layers()
        self.init_weights()



    def _init_layers(self):
        '''
        classification
        cls_output(torch.tensor):  shape=[bs,num_cls,h,w]

        regression
        attention: if use grid_point, pts_out_dim=4, else pts_out_dim=num_points*2(18 here)
        reg_output1(torch.tensor):  shape=[bs,pts_out_dim,h,w]
        reg_output2(torch.tensor):  shape=[bs,pts_out_dim,h,w]
        '''
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = []
        self.reg_convs = []
        # the first three convs before in localization/classification subnet
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(CGR(chn,self.feat_channels,3,1,1,False))
            self.reg_convs.append(CGR(chn,self.feat_channels,3,1,1,False))

        self.cls_convs=nn.Sequential(*self.cls_convs)
        self.reg_convs=nn.Sequential(*self.reg_convs)

        # if use reppoints, the output dim in localization subnet is equal to num_points*2.
        # if use box reprentation, the output dim in localization subnet is equal to 4
        pts_out_dim = 4 if self.use_grid_points else 2 * self.num_points

        # classification subnet
        self.reppoints_cls_conv = DeformConv2d(self.feat_channels,
                                               self.point_feat_channels,
                                               self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,self.cls_out_channels, 1, 1, 0)   # cls_out shape=[bs,num_cls,h,w]

        # 1st stage regression
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,self.point_feat_channels, 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)  # reg_out1 shape=[bs,pts_out_dim,h,w]

        # 2nd stage regression
        self.reppoints_pts_refine_conv = DeformConv2d(self.feat_channels,
                                                      self.point_feat_channels,
                                                      self.dcn_kernel, 1,
                                                      self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)  # reg_out2 shape=[bs,pts_out_dim,h,w]



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight,std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0.)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.reppoints_cls_out.bias, -math.log((1 - 0.01) / 0.01))



    def gen_grid_from_reg(self, reg, previous_boxes):
        """
        Base on the previous bboxes and regression values, we compute the regressed bboxes and
        generate the grids on the bboxes.

        :param reg: the regression(predicted) value to previous bboxes.  shape=[bs,4,h,w]
        :param previous_boxes: previous bboxes/ prior anchors.    shape=[1,4,1,1]   4==>(-2,-2,2,2)

        :return:
        注意grid_yx和regressed_bbox都是中心化的预测结果,即关于featuremap上某点(grid_x,grid_y)为原点时的对应尺度上的预测结果
        grid_yx(torch.tensor): shape=[bs,9*2,h,w]  根据预测的box得出的offset预测,
        regressed_bbox(torch.Tensor):  shape=[bs,4,h,w]  4==>(x1,y1,x2,y2)in corresponding featuremap size 解码出的对应featuremap尺度上的predicted_box
        """
        b, _, h, w = reg.shape
        # shape=[1,2,1,1]
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.
        bwh = (previous_boxes[:, 2:, ...] - previous_boxes[:, :2, ...]).clamp(min=1e-6)

        # decode predicted results
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])

        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]


        intervel = torch.linspace(0., 1., self.dcn_kernel).view(1, self.dcn_kernel, 1, 1).type_as(reg)   # shape=[1,3,1,1]  3==>(0,0.5,1)
        # shape: [bs,3,h,w]=>[bs,1,3,h,w]=>[bs,3,3,h,w]=>[bs,9,h,w]
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        # shape=[bs,9,2,h,w]=>[bs,9*2,h,w]
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)

        regressed_bbox = torch.cat([grid_left, grid_top, grid_left + grid_width, grid_top + grid_height], 1)
        return grid_yx, regressed_bbox



    def single_forward(self, x):
        '''
        :param x(torch.tensor): [bs,c,h,w] one of featuremaps in fpn layer

        :return:
        cls_output(torch.Tensor):  shape=[bs,num_cls,h,w]
        pts 是在对应的featuremap尺度上作出的预测,即关于featuremap上某点(grid_x,grid_y)为原点时的对应尺度上的offset预测结果
        pts_out_init(torch.Tensor):   shape=[bs,num_points*2,h,w]  2==>y,x
        pts_out_refine(torch.Tensor):  shape=[bs,num_points*2,h,w]  2==>y,x
        '''

        dcn_base_offset = self.dcn_base_offset.type_as(x)  # [1,18,1,1]


        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is from regular grid placed on a pre-defined bbox.
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = x.new_tensor([-scale, -scale, scale, scale]).view(1, 4, 1, 1)
        else:
            points_init = 0


        ############ the first three convblock -------------------------------------------------------------------------
        cls_feat=self.cls_convs(x)
        pts_feat=self.reg_convs(x)


        ###################################   regression 1 -------------------------------------------------------------

        pts_out_init = self.reppoints_pts_init_out(self.relu(self.reppoints_pts_init_conv(pts_feat)))
        # reg_out1, shape=[bs,4/num_points*2,h,w]

        # 解码生成第一次regression的在对应特征层尺度上的预测结果
        if self.use_grid_points:
            # pts_out_init.shape=[bs,num_points*2,h,w]
            # bbox_out_initshape=[bs,4,h,w]  4==>(x1,y1,x2,y2)in corresponding featuremap size
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(pts_out_init, bbox_init.detach())
        else:
            # shape=[bs,num_points*2,h,w]
            pts_out_init = pts_out_init + points_init



        ################################   regression 2 and classification ---------------------------------------------
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        cls_out = self.reppoints_cls_out(self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))  # cls_out

        pts_out_refine = self.reppoints_pts_refine_out(self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        if self.use_grid_points:
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()

        return cls_out, pts_out_init, pts_out_refine



    def forward(self,feats):
        '''
        :param feats(a list of featuremaps):  [p3,p4,p5,p6,p7]  pi.shape=[bs,c,h,w]
        :return:
        cls_out_list(len=5): list(cls_out)  cls_out.shape=[bs,num_cls,h,w]
        pts_init_out_list(len=5):  list(pts_init_out)  pts_init_out.shape=[bs,num_points*2,h,w]  2=>y,x
        pts_refine_out_list(len=5):  list(pts_refine_out)  pts_init_out.shape=[bs,num_points*2,h,w]  2=>y,x
        '''
        return multi_apply(self.single_forward,feats)




default_cfg = {
    # model
    "num_cls": 80,
    "point_strides": [8, 16, 32, 64, 128],
    "backbone": "resnet18",
    "fpn_channel": 256,
    "feat_channel" : 256,
    "point_feat_channel" : 256,
    "num_conv": 3,
    "num_point" : 9,
    "gradient_mul" : 0.1,
    "point_base_scale" : 4,
    "use_grid_points" : False,
    "center_init" : True,
    # loss
    "transform_method":'moment',
    "moment_mul":0.01,
    "alpha":0.25,
    "gamma":2.0,
    "cls_weight":1.0,
    "beta_init":1.0/9.0,
    "loss_init_weight":0.5,
    "beta_refine":1.0/9.0,
    "loss_refine_weight":1.0,
}





class RepPoints(nn.Module):
    def __init__(self, **kwargs):
        self.cfg = {**default_cfg, **kwargs}
        super(RepPoints, self).__init__()
        self.backbones = switch_backbones(self.cfg['backbone'])
        c3, c4, c5 = self.backbones.inner_channels
        self.neck = FPN(c3, c4, c5, self.cfg['fpn_channel'], bias=False)
        self.head = RepPointsHead( num_classes=self.cfg['num_cls'],
                                   in_channels=self.cfg['fpn_channel'],
                                   feat_channels=self.cfg['feat_channel'],
                                   point_feat_channels=self.cfg['point_feat_channel'],
                                   stacked_convs=self.cfg['num_conv'],
                                   num_points=self.cfg['num_point'],
                                   gradient_mul=self.cfg['gradient_mul'],
                                   point_base_scale=self.cfg['point_base_scale'],
                                   use_grid_points=self.cfg['use_grid_points'],
                                   center_init=self.cfg['center_init'],
                                  )

        self.loss = RepLoss(num_class=self.cfg['num_cls'],
                            num_points=self.cfg['num_point'],
                            point_strides=self.cfg['point_strides'],
                            point_base_scale=self.cfg['point_base_scale'],
                            transform_method=self.cfg['transform_method'],
                            moment_mul=self.cfg['moment_mul'],
                            alpha=self.cfg['alpha'],
                            gamma=self.cfg['gamma'],
                            cls_weight=self.cfg['cls_weight'],
                            beta_init=self.cfg['beta_init'],
                            loss_init_weight=self.cfg['loss_init_weight'],
                            beta_refine=self.cfg['beta_refine'],
                            loss_refine_weight=self.cfg['loss_refine_weight'])


    def forward(self, x, targets=None, shapes=None, assigner_cfg=None):
        c3, c4, c5 = self.backbones(x)
        p3, p4, p5, p6, p7 = self.neck([c3, c4, c5])

        cls_out_list,pts_out1_list,pts_out2_list = self.head([p3, p4, p5, p6, p7])

        if self.training:
            out=self.loss(cls_out_list,
                           pts_out1_list,
                           pts_out2_list,
                           targets,
                           shapes,
                           assigner_cfg)
        else:
            out=self.loss.get_bboxes(cls_out_list,pts_out2_list)

        return out





if __name__ == '__main__':
    from datasets.coco import COCODataSets
    from torch.utils.data.dataloader import DataLoader

    train_cfg = dict(
        init=dict(
            assigner=dict(type='PointAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))

    dataset = COCODataSets(img_root="/home/wangchao/public_dataset/coco/images/val2017",
                           annotation_path="/home/wangchao/public_dataset/coco/annotations/instances_val2017.json",
                           min_thresh=800,
                           max_thresh=1333,
                           use_crowd=False,
                           augments=True,
                           debug=False,
                           remove_blank=True
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=dataset.collect_fn)
    device=torch.device('cuda:0')
    net = RepPoints(backbone="resnet18")
    net = net.to(device)
    from torch.optim.sgd import SGD
    optim = SGD(net.parameters(), lr=1e-3)

    for imgs, targets, shapes in dataloader:
        imgs=imgs.to(device)
        targets=targets.to(device)
        loss,_,_=net(imgs, targets.to(device), shapes, train_cfg)
        print(loss)
        loss.backward()
        optim.step()



