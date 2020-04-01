# Object Tracking

## Setup tensorboard on 6006 port

```bash
bash logs/tb.sh logs/tb_runs 6006
```

## Download Data (approx 10 Mb/sec)

```bash
pip install gsutil
gsutil -m cp -r gs://vot-proj/coco/* data/coco/
```

## Run

```bash
python -m main \
    -mode trainval \
    -model_name resnet50-pysot \
    -batch_size 16 \
    -accumulation_interval 4 \
    -n_per_epoch 100000 \
    -save_filename siam \
    -tb_tag pysot
```

## Other

# Nvidia-smi
```bash
watch -n0.1 nvidia-smi
```