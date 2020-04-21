import argparse
import torch
from tensorboardX import SummaryWriter
from torch.optim.adamw import AdamW

from train import train
from eval import evaluate
from dataloader import get_train_dataloader
from model.tracker import SiamTracker
from log.logger import logger
from utils.loss import TrackingLoss
from utils.scheduler import build_scheduler
from utils.utils import init_seed, count_parameters, download_backbone_weights, str2maybeNone, DummyWriter
import config as cfg


def get_args():

    parser = argparse.ArgumentParser(description='test')

    parser.add_argument("-mode", choices=['trainval', 'test'], default="trainval", type=str)
    parser.add_argument("-model_name", default='resnet50-pysot', type=str)

    parser.add_argument("-batch_size", default=8, type=int)
    parser.add_argument("-accumulation_interval", default=4, type=int)
    parser.add_argument("-n_per_epoch", default=600000, type=int)

    parser.add_argument("-save_filename", default='siam.pth', type=str2maybeNone)
    parser.add_argument("-load_filename", default='None', type=str2maybeNone)
    parser.add_argument("-tb_tag", default='last_experiment', type=str2maybeNone)

    args = parser.parse_args()

    assert args.model_name in ['resnet50-pysot', 'resnet50-imagenet'] \
        or args.model_name.startswith('efficientnet') \
        or args.model_name.startswith('efficientdet'),\
        f'Wrong model name: {args.model_name}!'
    cfg.MODEL_HOLDER.choose_model(args.model_name)

    return args


def load_model(load_filename):
    model = SiamTracker()
    if load_filename is not None:
        model.load(load_filename)
    else:
        logger("Loading backbone weights...")
        if not cfg.MODEL_HOLDER.BACKBONE_WEIGHTS.exists():
            download_backbone_weights(cfg.MODEL_HOLDER.BACKBONE_TYPE, cfg.MODEL_HOLDER.BACKBONE_WEIGHTS)
        model.load_backbone(cfg.MODEL_HOLDER.BACKBONE_WEIGHTS)
        model.initialize_weights()
    logger(f"Model trainable parameters: {count_parameters(model)}")
    model.froze_backbone()
    model.unfroze_trainable()
    return model


def build_tools(model, train_backbone=False, last_epoch=0):
    optimizer = AdamW([
        {'params': model.backbone_trainable if train_backbone else [], 'initial_lr': cfg.BACKBONE_LR},
        {'params': model.chest_trainable, 'initial_lr': cfg.BASE_LR},
        {'params': model.neck_trainable, 'initial_lr': cfg.BASE_LR},
        {'params': model.head_trainable, 'initial_lr': cfg.BASE_LR},
    ], weight_decay=cfg.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, last_epoch=last_epoch)
    criter = TrackingLoss(cfg.CLS_WEIGHT, cfg.LOC_WEIGHT)
    return optimizer, scheduler, criter


def setup_gpu(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


def setup_multi_gpu(model):
    device = torch.device("cuda:0")
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model, device


def setup_writer(tb_tag, args):
    writer = DummyWriter() if args.tb_tag is None else SummaryWriter(logdir=cfg.TB_LOGDIR / tb_tag)
    writer.add_text("Hyperparams", '<br />'.join([f"{k}: {v}" for k, v in args.__dict__.items()]))
    writer.train_step, writer.eval_step = 0, 0
    return writer


def main(args):

    logger("Setup tensorboard writer...")
    writer = setup_writer(args.tb_tag, args)
    logger("Loading model...")
    model = load_model(args.load_filename)
    logger("Setup gpu...")
    model, device = setup_gpu(model)
    logger("Setup tools...")
    optimizer, scheduler, criter = build_tools(model, last_epoch=model.load_epoch)

    if args.mode == 'trainval':

        logger("Load train dataloader...")
        train_loader = get_train_dataloader(args.n_per_epoch, args.batch_size, cfg.NUM_WORKERS)

        for epoch in range(model.load_epoch, cfg.EPOCHS):

            # unfroze strategy
            if epoch > cfg.UNFROZE_EPOCH:
                model.unfroze_backbone()
                optimizer, scheduler, criter = \
                    build_tools(model, train_backbone=True, last_epoch=epoch)

            logger(f"{epoch} epoch training...")
            model, optimizer, scheduler, writer = train(
                model, optimizer, train_loader, scheduler, criter, device, writer, args)

            if not epoch % cfg.VALID_INTERVAL and epoch >= cfg.VALID_DELAY:
                logger(f"{epoch} epoch validation...")
                model, writer = evaluate(model, device, writer, args.save_filename, args, epoch)

    elif args.mode == 'test':

        logger("Testing...")
        evaluate(model, device, writer, args.save_filename, args)

    writer = logger.write(writer)
    writer.close()


if __name__ == '__main__':
    init_seed(seed=cfg.SEED)
    main(get_args())
