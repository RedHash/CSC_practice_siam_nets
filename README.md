# Object Tracking

## Setup tensorboard on 6006 port

```bash
bash logs/tb.sh logs/tb_runs 6006
```

## Download Data (approx 10 Mb/sec)

### Google cloud

```bash
pip install gsutil
gsutil -m cp -r gs://vot-proj/coco/* data/coco/
```

### Model Zoo
| Model Name | backbone | #params | EAO | 
| :----------: | :--------: | :-----------: | :--------: |
| SiamRPN++  | ResNet50 PySot | 53.95M | TBD | 
| SiamRPN++  | ResNet50 ImageNet | 53.95M | TBD | 
| EfficientSiamRPN  | EfficientNet-B0 | 5.79M | TBD | 
| EfficientSiamRPN  | EfficientNet-B1 | 8.30M | TBD | 
| EfficientSiamRPN  | EfficientNet-B2 | 9.62M | TBD | 
| EfficientSiamRPN  | EfficientNet-B3 | 12.74M | TBD | 
| EfficientSiamRPN  | EfficientNet-B4 | 19.85M | TBD | 
| Siamese EfficientDet | EfficientNet-B0 | TBD | TBD | 
| Siamese EfficientDet  | EfficientNet-B1 | TBD | TBD | 
| Siamese EfficientDet  | EfficientNet-B2 | TBD | TBD | 
| Siamese EfficientDet  | EfficientNet-B3 | TBD | TBD | 
| Siamese EfficientDet  | EfficientNet-B4 | TBD | TBD | 


### Download Coco

```bash
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip && mv train2017 data/coco && rm train2017.zip

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && mv val2017 data/coco && rm val2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && mv annotations data/coco && rm annotations_trainval2017.zip
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