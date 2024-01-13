# Deep learning template

## Environment
- Pytorch

### Default Environmental installation
```shell script
pip install -r requirements.txt
```

## Train
```shell script
python main.py --config ./config/bird.yml --gpus 0 --batch_size 16 --epochs 1001 --output ./output/bird
```

### Resume training process
If your training process is interrupted unexpectedly, set resume_epoch and resume_model_path in main.py to resume training.


## Val
### Tensorboard
Our code supports automate FID evaluation during training, the results are stored in TensorBoard files under `output_dir/logs`. You can change the test interval by changing test_interval in the YAML file.
 
 ```shell script
 tensorboard --logdir=./output/bird/logs --port 8166
```

## code structure
- [main.py](main.py): Entry file
- [src](src): Related code
    - [train.py](src/train.py): trainer
    - [dataset.py](src/dataset.py): Dataset related code
    - [losses.py](src/losses.py): Custom loss code
    - [utils.py](src/utils.py): Tool code
    - [models](src/models): Model-dependent code
- [requirements.txt](requirements.txt): Default environment
- [config](configs): Configuration file directory
- output: Default output directory
- [README](README.md): English README
- [README_zh](README_zh.md): 中文版README