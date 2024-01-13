# 深度学习模版

## 环境
- Pytorch

### 默认环境安装
```shell script
pip install -r requirements.txt
```

## 训练
```shell script
python main.py --config ./config/bird.yml --gpus 0 --batch_size 16 --epochs 1001 --output ./output/bird
```

### 恢复训练进程
If your training process is interrupted unexpectedly, set resume_epoch and resume_model_path in main.py to resume training.


## 验证
### Tensorboard
Our code supports automate FID evaluation during training, the results are stored in TensorBoard files under `output_dir/logs`. You can change the test interval by changing test_interval in the YAML file.
 
 ```shell script
 tensorboard --logdir=./output/bird/logs --port 8166
```

## 代码结构
- [main.py](main.py): 入口文件
- [src](src): 相关代码
    - [train.py](src/train.py): 训练器
    - [dataset.py](src/dataset.py): 数据集相关代码
    - [losses.py](src/losses.py): 自定义loss相关代码
    - [utils.py](src/utils.py): 工具类代码
    - [models](src/models): 模型相关代码
- [requirements.txt](requirements.txt): 默认环境
- [config](configs): 配置文件目录
- output: 默认输出目录
- [README](README.md): English README
- [README_zh](README_zh.md): 中文版README