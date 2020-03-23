import argparse

from evaluation.got10k.experiments import ExperimentVOT
from evaluation.wrapper import TrackerEvalWrapper
import config as cfg

METRICS = ['accuracy', 'robustness', 'speed_fps']


def evaluate(model, device, writer, visualize=False):

    model.eval()
    tracker = TrackerEvalWrapper(model, device)

    if cfg.EVAL_KWARGS['dataset_name'] == 'VOT':
        experiment = ExperimentVOT(cfg.EVAL_KWARGS['root_dir'],
                                   version=cfg.EVAL_KWARGS['version'],
                                   download=cfg.EVAL_KWARGS['download'])
        experiment.run(tracker, visualize=visualize)
        report = experiment.report([tracker.name])

        metrics = report[cfg.MODEL_NAME]

    else:
        # TODO: Possibly add other datasets
        raise NotImplementedError('Other evaluation datasets not supported yet; Use VOT')

    # Log to TensorBoard
    # metrics is a dict {'accuracy: .., 'robustness': .., 'speed_fps': ..}
    for m in METRICS:
        writer.add_scalar(f"Eval/{m}", metrics[m], writer.eval_step)

    metrics_improved = False

    # Assuming that model's properties keep the best values:
    if model.accuracy < metrics['accuracy']:
        model.metrics[0] = metrics['accuracy']
        metrics_improved = True

    if model.robustness > metrics['robustness']:
        model.metrics[1] = metrics['robustness']
        metrics_improved = True

    if model.speed_fps < metrics['speed_fps']:
        model.metrics[2] = metrics['speed_fps']
        metrics_improved = True

    # save weights if any metric improved
    if metrics_improved:
        stats = "".join('{}:{}'.format(k, v) for k, v in metrics.items())
        model.save(stats + '.pth')

    writer.eval_step += 1

    return tracker.model, writer


if __name__ == "__main__":
    """ quick test """

    parser = argparse.ArgumentParser(description='Tracking evaluation')
    parser.add_argument('--dataset_name', type=str, default='VOT')
    parser.add_argument('--version', type=int, default=2016)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--download', type=bool, default=False)
    parser.add_argument('--visualize', type=bool, default=True)
    args = parser.parse_args()

    evaluate(data_path=args.data_path, model_path=args.model_path, visualize=args.visualize,
             download=args.download, version=args.version)
