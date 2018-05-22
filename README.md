# BON
Code for Boundary Optimizing Network (BON): https://arxiv.org/abs/1801.02642

The code uses the IRIS dataset as an example. To use the code for another dataset, use an appropriate D network (e.g. DenseNet for images) and G network (a large enough network to generate data points). 

Example of running the code with 100 generators:
```
python -i bon++.py --nbG 100 --w_g 2.0 --w_exp 1.025 --lr 0.001 --lr_g 0.0001 --saving 50 --niter 1001
```
