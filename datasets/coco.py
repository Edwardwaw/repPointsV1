import os
import torch
from typing import List
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO
from commons.augmentations import *

coco_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
colors = [(67, 68, 113), (130, 45, 169), (2, 202, 130), (127, 111, 90), (92, 136, 113),
          (33, 250, 7), (238, 92, 104), (0, 151, 197), (134, 9, 145), (253, 181, 88),
          (246, 11, 137), (55, 72, 220), (136, 8, 253), (56, 73, 180), (85, 241, 53),
          (153, 207, 15), (187, 183, 180), (149, 32, 71), (92, 113, 184), (131, 7, 201),
          (56, 20, 219), (243, 201, 77), (13, 74, 96), (79, 14, 44), (195, 150, 66),
          (2, 249, 42), (195, 135, 43), (105, 70, 66), (120, 107, 116), (122, 241, 22),
          (17, 19, 179), (162, 185, 124), (31, 65, 117), (88, 200, 80), (232, 49, 154),
          (72, 1, 46), (59, 144, 187), (200, 193, 118), (123, 165, 219), (194, 84, 34),
          (91, 184, 108), (252, 64, 153), (251, 121, 27), (105, 93, 210), (89, 85, 81),
          (58, 12, 154), (81, 3, 50), (200, 40, 236), (155, 147, 180), (73, 29, 176),
          (193, 19, 175), (157, 225, 121), (128, 195, 235), (146, 251, 108), (13, 146, 186),
          (231, 118, 145), (253, 15, 105), (187, 149, 62), (121, 247, 158), (34, 8, 142),
          (83, 61, 48), (119, 218, 69), (197, 94, 130), (222, 176, 142), (21, 20, 77),
          (6, 42, 17), (136, 33, 156), (39, 252, 211), (52, 50, 40), (183, 115, 34),
          (107, 80, 164), (195, 215, 74), (7, 154, 135), (136, 35, 24), (131, 241, 125),
          (208, 99, 208), (5, 4, 129), (137, 156, 175), (29, 141, 67), (44, 20, 99)]

default_aug_cfg = {
    'hsv_h': 0.014,
    'hsv_s': 0.68,
    'hsv_v': 0.36,
    'degree': (-10, 10),
    'translate': 0,
    'shear': 0.0,
    'beta': (8, 8),
    'pad_val': (103, 116, 123),
}

rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]
cv.setNumThreads(0)


# noinspection PyTypeChecker
class COCODataSets(Dataset):
    def __init__(self, img_root, annotation_path,
                 min_thresh=800,
                 max_thresh=1333,
                 augments=True,
                 use_crowd=True,
                 debug=False,
                 remove_blank=True,
                 aug_cfg=None,
                 ):
        """
        :param img_root: ???????????????
        :param annotation_path: ?????????json??????????????????
        :param augments: ????????????????????????
        :param use_crowd: ????????????crowed?????????
        :param debug: debug??????(????????????)
        :param remove_blank: ????????????????????????????????????
        :param aug_cfg: ?????????????????????
        """
        super(COCODataSets, self).__init__()
        self.coco = COCO(annotation_path)
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.img_root = img_root
        self.use_crowd = use_crowd
        self.remove_blank = remove_blank
        self.augments = augments
        if aug_cfg is None:
            aug_cfg = default_aug_cfg
        self.aug_cfg = aug_cfg
        self.debug = debug
        self.empty_images_len = 0
        data_len = len(self.coco.imgs.keys())
        box_info_list = self.__load_data()
        self.box_info_list = box_info_list
        if len(box_info_list) != data_len:
            print("all data len:{:d} | valid data len:{:d}".format(data_len, len(box_info_list)))

        if self.debug:
            assert debug <= len(box_info_list), "not enough data to debug"
            print("debug")
            self.box_info_list = box_info_list[:debug]
        self.transform = None
        self.set_transform()
        self.batch_transform=BatchPadding(padding_value=self.aug_cfg['pad_val'],center_padding=False,size_divisible=32)

    def __load_data(self):
        box_info_list = list()
        for img_id in self.coco.imgs.keys():
            file_name = self.coco.imgs[img_id]['file_name']
            width, height = self.coco.imgs[img_id]['width'], self.coco.imgs[img_id]['height']
            file_path = os.path.join(self.img_root, file_name)
            if not os.path.exists(file_path):
                print("img {:s} is not exist".format(file_path))
                continue

            assert width > 1 and height > 1, "invalid width or heights"
            anns = self.coco.imgToAnns[img_id]
            label_list = list()
            for ann in anns:
                category_id, box, iscrowd = ann['category_id'], ann['bbox'], ann['iscrowd']
                label_id = coco_ids.index(category_id)
                assert label_id >= 0, 'error label_id'
                if not self.use_crowd and iscrowd == 1:
                    continue
                x1, y1 = box[:2]
                x2, y2 = x1 + box[2], y1 + box[3]
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    print("not a valid box ", box)
                    continue
                if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
                    print("warning box ", box)
                label_list.append((label_id, x1, y1, x2, y2))
            valid_box_len = len(label_list)
            # if valid_box_len == 0:
            #     box_info = BoxInfo(img_path=file_path, boxes=np.zeros((0, 4)), labels=np.zeros((0,)),
            #                        weights=np.ones(shape=(0,)))
            # else:
            #     label_info = np.array(label_list)
            #     box_info = BoxInfo(img_path=file_path, boxes=label_info[:, 1:], labels=label_info[:, 0],
            #                        weights=np.ones_like(label_info[:, 0]))
            # if self.remove_blank and valid_box_len == 0:
            #     self.empty_images_len += 1
            #     continue

            if valid_box_len == 0:
                self.empty_images_len += 1
                continue

            label_info = np.array(label_list).reshape(valid_box_len,-1)
            box_info = BoxInfo(img_path=file_path, boxes=label_info[:, 1:], labels=label_info[:, 0],weights=np.ones_like(label_info[:, 0]))

            box_info_list.append(box_info)
        return box_info_list

    def __getitem__(self, item):
        box_info = self.box_info_list[item].clone().load_img()
        box_info = self.transform(box_info)
        # assert box_info.img.dtype == np.uint8
        # import uuid
        # ret_img = box_info.draw_box(colors, coco_names)
        # file_name = str(uuid.uuid4()).replace("-", "")
        # cv.imwrite("{:s}.jpg".format(file_name), ret_img)
        return box_info


    def collect_fn(self, batch: List[BoxInfo]):
        '''

        :param batch:
        :return:
        images shape[bs,3,h,w]
        targets[bs,6] (bs_idx,label_idx,x1,y1,x2,y2) unnormalized
        batch_shapes(list,len=bs): list(item)  item=(w,h) before batch_padding
        '''
        imgs, batch_shapes, labels = self.batch_transform(batch)
        return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().float(), torch.from_numpy(labels).float(), batch_shapes

    def __len__(self):
        return len(self.box_info_list)

    def set_transform(self):
        color_gitter = OneOf(
            transforms=[
                Identity(),
                RandHSV(hgain=self.aug_cfg['hsv_h'],
                        vgain=self.aug_cfg['hsv_v'],
                        sgain=self.aug_cfg['hsv_s']),
                RandBlur().reset(p=0.5),
                RandNoise().reset(p=0.5)
            ]
        )
        basic_transform = Compose(
            transforms=[
                color_gitter,
                RandScaleMinMax(self.min_thresh,self.max_thresh),
                # RandCrop(min_thresh=0.6, max_thresh=1.0).reset(p=0.2),
                # RandScaleToMax(max_threshes=[size for size in range(self.min_thresh,self.max_thresh,32)]),
                # RandScaleToMax(max_threshes=[self.max_thresh]),
                # RandPerspective(degree=self.aug_cfg['degree'], scale=(0.8, 1.2))
            ]
        )

        mosaic = MosaicWrapper(candidate_box_info=self.box_info_list,
                               sizes=[size for size in range(self.min_thresh,self.max_thresh,32)],
                               color_gitter=color_gitter)

        mix_up = MixUpWrapper(candidate_box_info=self.box_info_list,
                              beta=self.aug_cfg['beta'],
                              color_gitter=Compose(
                                  transforms=[
                                      color_gitter,
                                      RandScaleMinMax(min_thresh=self.max_thresh,
                                                      max_thresh=self.max_thresh),
                                  ]
                              ))

        augment_transform = Compose(
            transforms=[
                OneOf(transforms=[
                    (1.0, basic_transform),
                    (0.0, mix_up),
                    (0.0, mosaic)
                ]),
                # color_gitter,
                LRFlip().reset(p=0.5)
            ]
        )
        std_transform = RandScaleToMax(max_threshes=[self.max_thresh])

        if self.augments:
            self.transform = augment_transform
        else:
            self.transform = std_transform



if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    dataset = COCODataSets(img_root="/home/wangchao/public_dataset/coco/images/val2017",
                           annotation_path="/home/wangchao/public_dataset/coco/annotations/instances_val2017.json",
                           use_crowd=False,
                           augments=True,
                           debug=False,
                           remove_blank=True
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=dataset.collect_fn)

    import time
    no_images = 0
    t1=time.time()
    for img_tensor, target_tensor, _ in dataloader:
        print(target_tensor.shape,target_tensor[:,1].max(),target_tensor[:,1].min())
        # print(img_tensor.shape, target_tensor.shape)
        # if target_tensor.shape[0]==0:
        #     no_images+=1
    t2=time.time()
    print('used time is ',(t2-t1))
    print('no image is ',no_images)