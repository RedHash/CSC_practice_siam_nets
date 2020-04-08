from evaluation.got10k.experiments import ExperimentVOT
from evaluation.wrapper import TrackerEvalWrapper
from model.tracker import SiamTracker
import argparse
import torch

import config as cfg
from config import ModelHolder

METRICS = ['accuracy', 'robustness', 'speed_fps']


def evaluate(model, device, writer, save_filename, args, epoch=0, visualize=False):

    model.eval()
    tracker = TrackerEvalWrapper(model, device, args.model_name)

    if cfg.EVAL_KWARGS['dataset_name'] == 'VOT':
        experiment = ExperimentVOT(cfg.EVAL_KWARGS['root_dir'],
                                   version=cfg.EVAL_KWARGS['version'],
                                   download=cfg.EVAL_KWARGS['download'])
        experiment.run(tracker, visualize=visualize)
        report = experiment.report([tracker.name])
        metrics = report[args.model_name]
    else:
        # TODO: Possibly add other datasets
        raise NotImplementedError('Other evaluation datasets not supported yet; Use VOT')

    # metrics is a dict {'accuracy: .., 'robustness': .., 'speed_fps': ..}
    for m in METRICS:
        writer.add_scalar(f"Eval/{m}", metrics[m], writer.eval_step)
    if save_filename is not None:
        model.save(save_filename, epoch)
    writer.eval_step += 1

    return tracker.model, writer


if __name__ == "__main__":
    """ Test on VOT  """

    parser = argparse.ArgumentParser(description='Tracking evaluation')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--visualize', type=bool, default=True)
    args = parser.parse_args()

    siam_tracker = SiamTracker()
    holder = ModelHolder()
    holder.choose_model('resnet50-pysot')
    siam_tracker.load_state_dict(torch.load(args.model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    siam_tracker = siam_tracker.to(device)

    evaluate(model=siam_tracker, device=device, writer=None, visualize=args.visualize)
