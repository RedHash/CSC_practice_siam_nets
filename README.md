# Object Tracking

## Setup tensorboard on 6006 port

```bash
bash logs/tb.sh logs/tb_runs 6006
```

## Nvidia-smi

```bash
watch -n0.1 nvidia-smi
```

## Download Data (approx 10 Mb/sec)

```bash
pip install gsutil
gsutil -m cp -r gs://vot-proj/coco/* data/coco/
```

