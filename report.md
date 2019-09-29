<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>

# MLP Report

Name: Xiangyun Ding (丁相允)

Student ID: 2016011361

## Overview

In the forward propagation, given input $\mathbf{x}=[x_1,x_2,...,x_n]$, the output of Softmax layer is $\mathbf{y}=[y_1,y_2,...,y_n]$, where we have:
$$y_i=\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$$

In the backward propagation, we've known $\frac{\partial E}{\partial y_i}$, and we want to calculate $\frac{\partial E}{\partial x_i}$. So we should use the chain rule:
$$\frac{\partial E}{\partial x_i}=\sum_{j=1}^n\frac{\partial E}{\partial y_j}\times\frac{\partial y_j}{\partial x_i}$$

If $j\neq i$, then we have:
$$\frac{\partial y_j}{\partial x_i}=\frac{\partial \frac{e^{x_j}}{\sum_{j=1}^n e^{x_j}}}{\partial x_i}=e^{x_i}\times-\frac{e^{x^j}}{(\sum_{j=1}^n e^{x_j})^2}=-y_i\times y_j$$

If $j=i$, then we have:
$$\frac{\partial y_j}{\partial x_i}=\frac{\partial\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}}{\partial x_i}=\frac{(\sum_{j=1}^n e^{x_j})\times e^{x_i}-e^{x_i}\times e^{x_i}}{(\sum_{j=1}^n e^{x_j})^2}=y_i\times(1-y_i)$$

So we can simply use the result of $j\neq i$, then add $y_i$ to the answer. Finally we have:
$$\frac{\partial E}{\partial x_i}=\sum_{j=1}^n\frac{\partial E}{\partial y_j}\times-y_i\times y_j+\frac{\partial E}{\partial y_i}\times y_i$$

This is the forward and backward propagation equation for Softmax layer.

## Experiments

Basic Results:

|ID|Network Structure|Loss Function|Learning Rate|Train Time(s)|Best Accuracy(%)|
|----|----|----|----|----|----|----|
|0|Linear($784\times128$)->Relu->Linear($128\times10$)->Softmax|EuclideanLoss|0.1|179|96.7|
|1|Linear($784\times128$)->Sigmoid->Linear($128\times10$)->Softmax|EuclideanLoss|0.1|431|95.1|
|2|Linear($784\times128$)->Relu->Linear($128\times10$)->Softmax|CrossEntropyLoss|0.1|84.3|97.6|
|3|Linear($784\times128$)->Sigmoid->Linear($128\times10$)->Softmax|CrossEntropyLoss|0.1|326|97.4|
|4|Linear($784\times128$)->Relu->Linear($128\times128$)->Relu->Linear($128\times10$)->Softmax|EuclideanLoss|0.1|162|96.5|
|5|Linear($784\times128$)->Relu->Linear($128\times128$)->Relu->Linear($128\times10$)->Softmax|CrossEntropyLoss|0.1|101|98.1|

Loss-Accuracy Figures:
|ID|Figure|
|----|----|
|0|<img src="codes/images/0.png" height=300>|
|1|<img src="codes/images/1.png" height=300>|
|2|<img src="codes/images/2.png" height=300>|
|3|<img src="codes/images/3.png" height=300>|
|4|<img src="codes/images/4.png" height=300>|
|5|<img src="codes/images/5.png" height=300>|

For higher resolution images, please visit directory `codes/images`.
