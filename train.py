import numpy as np
from torch import Tensor
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from utils.utils import is_valid_number, get_lr, get_gradnorm
import config as cfg

LOSS_PARTS = ['overall', 'clsneg', 'clspos', 'reg']
MODEL_PARTS = ['backbone', 'chest', 'neck', 'head']


def train(model, optimizer, dataloader, scheduler, criter, device, writer):

    loss_values = np.zeros(4)
    model.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader.dataset), leave=False)
    for idx_batch, batch in pbar:

        templates, detections, gts, pos_anchors, neg_anchors = \
            [x.to(device) if isinstance(x, Tensor)
             else list(map(lambda e: e.to(device), x)) for x in batch]

        cls_outputs, loc_outputs = model(templates, detections)

        gts = gts.float()

        loss, values = criter(cls_outputs, loc_outputs, gts, pos_anchors, neg_anchors)

        loss_values += values
        pbar.set_description(
            " | ".join([f"{prt}:{val:.2f}" for prt, val in zip(LOSS_PARTS, values)])
        )

        if is_valid_number(loss.data.item()):
            loss.backward()

            for prt, val in zip(LOSS_PARTS, values):
                writer.add_scalar(f'Train/{prt}_loss', val, writer.train_step)
            for group, model_part in enumerate(MODEL_PARTS):
                writer.add_scalar(f"Train/gradnorm_{model_part}", get_gradnorm(optimizer, group), writer.train_step)
            for group, model_part in enumerate(MODEL_PARTS):
                writer.add_scalar(f"Train/lr_{model_part}", get_lr(optimizer, group), writer.train_step)

            clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        writer.train_step += 1

        scheduler.step()

    return model, optimizer, scheduler, writer
