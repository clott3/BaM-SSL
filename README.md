BaM-SSL
---------------------------------------------------------------
<p align="center">
  <img width="900" alt="image" src="https://github.com/clott3/BaM-SSL/assets/55004415/b73e0cd9-2c88-4ccf-ae7d-1a2d2f0e8480">
</p>

Official PyTorch implementation of [Mitigating Confirmation Bias in Semi-supervised Learning via Efficient Bayesian Model Averaging](https://openreview.net/forum?id=PRrKOaDQtQ&).

## Training Code
We recommend training on CIFAR-100 for the largest stable gains (results on CIFAR-10 vary for different seeds). Current training code only supports training on a single GPU. 

### UDA Baseline:
```python train_bayes_cifar.py --dataset=cifar100 --num_labeled=400 --uda ```

### BaM-UDA (ours):
```python train_bayes_cifar.py --dataset=cifar100 --num_labeled=400 --uda --uda_T=0.9 --bayes ```

### FixMatch Baseline:
```python train_bayes_cifar.py --dataset=cifar100 --num_labeled=400 ```

### BaM-FM (ours):
```python train_bayes_cifar.py --dataset=cifar100 --num_labeled=400 --bayes ```

-----

For CIFAR-10 BaM versions, add `--final_quan=0.95 --bayes_lr=0.005` for best results. 
