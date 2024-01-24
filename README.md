# ST-SSL: Spatio-Temporal Self-Supervised Learning for Traffic Prediction 


引用于下述论文

* J. Ji, J. Wang, C. Huang, et al. "Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction". in Thirty-Seventh AAAI Conference on Artificial Intelligence, 2023. 

The homepage of J. Ji is available at [here](https://echo-ji.github.io/academicpages/).

## 实验环境需求

python3.8配置下面包: 
```
numpy==1.21.2
pandas==1.3.5
PyYAML==6.0
torch==1.10.1
```


## 初始代码地址
```
https://github.com/Echo-Ji/ST-SSL
```

## 自己修改增加的预训练的执行顺序
```
先执行 python pre_train.py 
随后执行 python new_trainer.py
```