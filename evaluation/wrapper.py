import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F

from utils.anchor import Anchors
from utils.crops import get_axis_aligned_bbox, corner_wh_bbox, get_scaled_bbox, corner2center
import config as cfg


class SiamRpnEvalTracker:
    def __init__(self, model, device):
        self.window = self.get_window()
        self.model = model
        self.model.eval()
        self.device = device

    def _init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate channel average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        w_z = self.size[0] + cfg.TRACK_CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK_CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.D_TEMPLATE,
                                    s_z, self.channel_average)

        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list): (x, y, w, h)
        """
        w_z = self.size[0] + cfg.TRACK_CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK_CONTEXT_AMOUNT * np.sum(self.size)

        s_z = np.sqrt(w_z * h_z)
        s_x = s_z * (cfg.D_DETECTION / cfg.D_TEMPLATE)

        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.D_DETECTION,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs[0])
        pred_bbox = self._convert_bbox(outputs[1], Anchors.centersizes)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        scale_z = cfg.D_TEMPLATE / s_z
        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK_PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK_WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK_WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK_LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        best_score = score[best_idx]

        return best_score, bbox

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            original_sz: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2

        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1

        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1

        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))

        if cfg.IS_NORM:
            im_patch = self.normalize(im_patch)

        else:
            im_patch = im_patch.transpose(2, 0, 1)
            im_patch = im_patch.astype(np.float32)
            im_patch = torch.from_numpy(im_patch)

        im_patch = im_patch.unsqueeze(0)
        im_patch = im_patch.to(self.device)

        return im_patch

    @staticmethod
    def get_window():
        score_size = (cfg.D_DETECTION - cfg.D_TEMPLATE) // cfg.ANCHOR_STRIDE + 1 + cfg.TRACK_BASE_SIZE
        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), cfg.ANCHOR_NUM)
        return window

    @staticmethod
    def _convert_bbox(delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    @staticmethod
    def _convert_score(score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    @staticmethod
    def _bbox_clip(cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    @staticmethod
    def normalize(x):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return transform(x)


class TrackerEvalWrapper(SiamRpnEvalTracker):
    def __init__(self, model, device, model_name):
        super(TrackerEvalWrapper, self).__init__(model, device)
        self.name = model_name
        self.is_deterministic = True

    def init(self, image, bbox):
        image = np.array(image)

        if cfg.BGR:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if len(bbox) == 4:
            bbox = corner_wh_bbox(bbox)

        cxy_wh = get_axis_aligned_bbox(np.array(bbox))

        scaled_bbox = get_scaled_bbox(cxy_wh)

        self._init(image, scaled_bbox)

    def update(self, image):
        """
        :param image:
        :return: bbox: (cx, cy, w, h)
        """
        image = np.array(image)

        if cfg.BGR:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        score, bbox = self.track(image)

        bbox = corner2center(bbox)

        return bbox


if __name__ == "__main__":
    """ quick test """
