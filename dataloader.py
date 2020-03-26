import json
import glob
import torch
import random
import numpy as np
from itertools import chain
from types import SimpleNamespace
from numpy import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageStat


from utils.anchor import get_anchors_numpy
from log.logger import logger
import config as cfg


class VideoBuffer:
    """ Custom class to store single video """

    def __init__(self, title, images, gt, sample_range):
        self.title = title
        self.images = images
        self.gt = gt
        self.sample_range = sample_range

    def __len__(self):
        return len(self.images)


class TrainingDataset(Dataset):

    def __init__(self, videos, n_per_epoch):
        self.videos = videos
        self.n_videos = len(self.videos)
        self.n_per_epoch = n_per_epoch
        self.transforms = self.get_transforms()
        self.anchors_corners, self.anchors_centersizes = get_anchors_numpy()

    def __getitem__(self, index):
        return self.single_sample(hard_negative=random.random() < cfg.HARDNEG_PROB)

    def __len__(self):
        return self.n_per_epoch

    def single_sample(self, hard_negative=False):
        """
        Implement extracting data for training
        :param hard_negative:
        :return: gt in format (x1, y1, x2, y2), ...
        """

        template_gt, detection_gt, template_img, detection_img = \
            self.sample_hardnegative() if hard_negative else self.sample()

        if cfg.BGR:
            template_img = self.rgb_to_bgr(template_img)
            detection_img = self.rgb_to_bgr(detection_img)

        template_A, detection_A = self.get_A(template_gt), self.get_A(detection_gt)
        template_center, detection_center = self.get_center(template_gt), self.get_center(detection_gt)
        template_indent, detection_indent = \
            self.sample_Acoef(cfg.TEMPLATE_BASE_A, cfg.TEMPLATE_SCALE) * template_A,\
            self.sample_Acoef(cfg.DETECTION_BASE_A, cfg.DETECTION_SCALE) * detection_A
        template_shift, detection_shift = \
            template_indent * self.sample_shift(cfg.TEMPLATE_SHIFT) / cfg.D_TEMPLATE, \
            detection_indent * self.sample_shift(cfg.DETECTION_SHIFT) / cfg.D_DETECTION

        template_img = self.crop_template(template_img, template_center,
                                          template_indent, template_shift)

        detection_gt = self.resize_gt(detection_gt, detection_center,
                                      detection_indent, detection_shift)

        detection_img = self.crop_detection(detection_img, detection_center,
                                            detection_indent, detection_shift)

        # TODO More checks
        # template_img_1 = self.transforms.to_pil(template_img)
        # template_img_1.show()
        #
        # from PIL import ImageDraw
        # detection_img_1 = self.transforms.to_pil(detection_img)
        # d = ImageDraw.Draw(detection_img_1)
        # d.rectangle(tuple(detection_gt), outline=1)
        # detection_img_1.show()

        # get dists and correct gt parametrization
        overlaps = self.get_iou(detection_gt)

        # generate positive anchors
        pos_anchors = np.array([])\
            if hard_negative \
            else self.sample_anchors(overlaps, lambda x: x >= cfg.THRESHOLD_HIGH, cfg.POS_NUM)

        # generate negative anchors
        neg_anchors = self.sample_anchors(overlaps, lambda x: x >= cfg.THRESHOLD_HARDNEG, cfg.HARDNEG_NUM)\
            if hard_negative \
            else self.sample_anchors(overlaps, lambda x: x <= cfg.THRESHOLD_LOW, cfg.NEG_NUM)

        # norm
        if cfg.IS_NORM:
            template_img = self.transforms.norm(template_img)
            detection_img = self.transforms.norm(detection_img)

        # norm + unsqueeze along batch dimension
        return template_img.unsqueeze(0), \
            detection_img.unsqueeze(0), \
            torch.from_numpy(detection_gt).unsqueeze(0), \
            torch.from_numpy(pos_anchors).long(), \
            torch.from_numpy(neg_anchors).long()

    def sample(self):
        """ Extract data from single video for normal train """
        # choose specific video
        idx = np.random.randint(self.n_videos)
        video = self.videos[idx]

        # choose template/detection indexes
        n_frames = len(video)
        template_idx = np.random.randint(n_frames)

        shift = np.random.randint(- video.sample_range, video.sample_range + 1)
        detection_idx = np.clip(template_idx + shift, a_min=0, a_max=n_frames - 1)

        return video.gt[template_idx], \
            video.gt[detection_idx], \
            Image.open(video.images[template_idx]).convert("RGB"), \
            Image.open(video.images[detection_idx]).convert("RGB")

    def sample_hardnegative(self):
        """ Extract data from 2 different videos for hard-negative train """
        # choose 2 different video with non-similar names
        idx1, idx2 = np.random.choice(self.n_videos, size=2, replace=False)
        video1, video2 = self.videos[idx1], self.videos[idx2]
        # choose template/detection indexes
        n_frames1, n_frames2 = len(video1), len(video2)
        template_idx, detection_idx = np.random.randint(n_frames1), np.random.randint(n_frames2)

        return video1.gt[template_idx], \
            video2.gt[detection_idx], \
            Image.open(video1.images[template_idx]).convert("RGB"), \
            Image.open(video2.images[detection_idx]).convert("RGB")

    def crop_template(self, img, center, indent, shift):
        """ Crop and resize image """
        center -= shift
        img = self.crop_treat_bounds(img, center, indent)
        img = self.transforms.resize_template(img)
        img = self.transforms.to_tensor(img)
        return img

    def crop_detection(self, img, center, indent, shift):
        """ Crop and resize image """
        center -= shift
        img = self.crop_treat_bounds(img, center, indent)
        img = self.transforms.resize_detection(img)
        img = self.transforms.to_tensor(img)
        return img

    def crop_treat_bounds(self, img, center, indent):
        """ Crop image filling bounds with mean pixels """
        def get_boundaries(img, center, indent):
            leftup = np.clip(center - indent, a_min=0, a_max=None)
            rightdown = np.clip(center + indent, a_min=None, a_max=img.size)
            shift = leftup - center + indent
            return np.array([*leftup, *rightdown]), shift
        boundaries, shift = get_boundaries(img, center, indent)
        pure_img = img.crop(boundaries)
        mean_pixels = self.get_mean_pixels(pure_img)
        blank = Image.new('RGB', (int(2 * indent), int(2 * indent)), mean_pixels)
        shift_x, shift_y = shift
        blank.paste(pure_img, (int(shift_x), int(shift_y)))
        return blank

    def get_iou(self, detection_gt):
        """ Calculate gt-anchors overlaps
        :param detection_gt: np.shape([4,])
        :return:
            overlaps | np.shape([n_anchors,])
        """
        x1, y1, x2, y2 = detection_gt
        ax1, ay1, ax2, ay2 = self.anchors_corners.T
        xx1 = np.maximum(ax1, x1)
        yy1 = np.maximum(ay1, y1)
        xx2 = np.minimum(ax2, x2)
        yy2 = np.minimum(ay2, y2)
        ww = np.maximum(0, xx2 - xx1)
        hh = np.maximum(0, yy2 - yy1)
        area = (x2 - x1) * (y2 - y1)
        anchor_area = (ax2 - ax1) * (ay2 - ay1)
        inter = ww * hh
        iou = inter / (area + anchor_area - inter)
        return iou

    @staticmethod
    def sample_Acoef(base, scale):
        """ Sample random resize A-coef """
        return base + np.random.uniform(-scale, scale)

    @staticmethod
    def get_mean_pixels(img):
        """ Calculate mean pixel for given PIL image """
        return tuple([int(x) for x in ImageStat.Stat(img).mean])

    @staticmethod
    def sample_shift(max_translation):
        """ Sample random translation for shifting detection center """
        return np.random.uniform(- max_translation, max_translation, 2)

    @staticmethod
    def sample_anchors(dists, condition, n_max):
        """ Sample (<= n_max) anchors by some condition on overlaps
        :param dists: gt-anchors iou | np.shape([n_anchors,])
        :param condition: condition on overlaps | boolean func
        :param n_max: max number of chosen anchors | int
        :return:
            np.shape([ <= n_max,])
        """
        idxs = np.where(condition(dists))[0]
        if len(idxs) == 0:
            return np.array([])
        else:
            return np.random.choice(idxs, min(len(idxs), n_max), replace=False)

    @staticmethod
    def resize_gt(gt, center, indent, shift):
        """ Resize groundtruth """
        gt = gt.reshape([2, 2]) - center + indent
        gt += shift
        gt = cfg.D_DETECTION * gt / (2 * indent)
        return gt.flatten()

    @staticmethod
    def get_center(gt):
        """ Calculate center by box coordinates """
        return (gt[0:2] + gt[2:4]) / 2

    @staticmethod
    def get_A(gt):
        """ Calculate heuristic indent for crop procedure """
        w, h = gt[2:4] - gt[0:2]
        p = (w + h) / 2
        return np.sqrt((w + p) * (h + p))

    @staticmethod
    def get_transforms():
        """ Obtain train transforms """
        return SimpleNamespace(
            to_tensor=transforms.ToTensor(),
            norm=transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            resize_template=transforms.Resize((cfg.D_TEMPLATE, cfg.D_TEMPLATE)),
            resize_detection=transforms.Resize((cfg.D_DETECTION, cfg.D_DETECTION)),
            to_pil=transforms.ToPILImage(),
        )

    @staticmethod
    def rgb_to_bgr(img):
        """ img: PIL RGB Image
            returns: PIL BGR Image """
        arr = np.array(img)
        arr = arr[:, :, ::-1]
        return Image.fromarray(arr)


def get_train_dataloader(n_per_epoch, batch_size, num_workers):
    """ Func to get dataloader for train """

    RAW_DATA_LOADER = {
        'COCO': load_raw_coco,
        # TODO More datasets
    }
    videos_list = list(chain.from_iterable([RAW_DATA_LOADER[dataset]() for dataset in cfg.TRAIN_DATASETS]))

    def collate_fn(batch_data):
        templates, detections, gts, pos, neg = zip(*batch_data)
        return torch.cat(templates), torch.cat(detections), torch.cat(gts), pos, neg

    return DataLoader(dataset=TrainingDataset(videos_list, n_per_epoch),
                      batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)


def load_raw_coco():
    """ Load COCO dataset to uniform format: List of `VideoBuffer`s """
    logger("Loading COCO dataset..")

    def load_split(split, anno_file):
        assert anno_file.endswith('.json')
        assert split in ['train2017', 'val2017']
        # load paths and anno
        with open(str(cfg.DATA_PATH / 'coco' / 'annotations' / anno_file)) as file:
            anno_file = json.load(file)
        images = {v.rsplit('/', 1)[-1]: v for v in
                  glob.glob(str(cfg.DATA_PATH / 'coco' / split / '*.jpg'))}
        # path2img, bbox_list
        id2data = {}
        for elem in anno_file['images']:
            id2data[elem['id']] = (images[elem['file_name']], [])
        for elem in anno_file['annotations']:
            if len(elem['bbox']) == 4:
                x1, y1, w, h = elem['bbox']
                id2data[elem['image_id']][1].append((x1, y1, x1 + w, y1 + h))
        # delete empty images
        del anno_file, images
        id2data = {k: v for k, v in id2data.items() if v[1]}
        # create buffers
        return [VideoBuffer(
            title=k, images=[path] * len(bbox),
            gt=np.array(bbox), sample_range=cfg.COCO_SAMPLE_RANGE
        ) for k, (path, bbox) in id2data.items()]

    return load_split('train2017', 'instances_train2017.json') + load_split('val2017', 'instances_val2017.json')


if __name__ == "__main__":
    """ quick test """

    dl = get_train_dataloader(n_per_epoch=12, batch_size=3, num_workers=0)
    for e in dl:
        print(e)
        break
    exit(0)
