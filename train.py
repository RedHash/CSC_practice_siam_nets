import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from utils.utils import is_valid_number, get_lr, get_gradnorm
import config as cfg

LOSS_PARTS = ['overall', 'clsneg', 'clspos', 'reg']
MODEL_PARTS = ['backbone', 'chest', 'neck', 'head']


def train(model, optimizer, dataloader, scheduler, criter, device, writer, args):

    loss_values = np.zeros(4)
    model.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for idx_batch, batch in pbar:

        templates, detections, gts, pos_anchors, neg_anchors = \
            [x.to(device) if isinstance(x, Tensor)
             else list(map(lambda e: e.to(device), x)) for x in batch]

        cls_outputs, loc_outputs = model(templates, detections)

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
            if not device == torch.device('cpu'):
                writer.add_scalar(f"Train/gpu memory", torch.cuda.memory_allocated(device), writer.train_step)
            writer.train_step += 1

            if (idx_batch + 1) % args.accumulation_interval == 0 or (idx_batch + 1) == len(dataloader):
                clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

    scheduler.step()

    return model, optimizer, scheduler, writer
