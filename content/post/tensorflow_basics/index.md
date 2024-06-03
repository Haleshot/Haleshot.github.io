---
title: Tensorflow Basics
date: '2024-06-03'
---
```python
# I066 Srihari Thyagarajan DL Lab 1
```

![image.png](data:image/png;base64,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)

```python
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
```

### Task 1:
#### Perform basic operations using Tensorflow (Defining constants, variables, concatenation, add, multiply, reduce mean, reduce sum)

### Checking the version of tensorflow before proceeding with the tasks

```python
tf.__version__
```

    '2.12.0'

```python
# Defining constants and variables and concatinating them:
A = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
B = tf.Variable([[5, 6, 7, 8], [9, 10, 11, 12]])

AB_concat = tf.concat([A, B], axis=1)
print(AB_concat.numpy())

# Adding, multiplying:
print("\nOn Adding:\n")
AB_add = tf.add(A, B)
print(AB_add.numpy())

print("\nOn Multiplying:\n")
AB_multiply = tf.multiply(A, B)
print(AB_multiply.numpy())
```

    [[ 1  2  3  4  5  6  7  8]
     [ 5  6  7  8  9 10 11 12]]
    
    On Adding:
    
    [[ 6  8 10 12]
     [14 16 18 20]]
    
    On Multiplying:
    
    [[ 5 12 21 32]
     [45 60 77 96]]

```python
# Reduce Mean:
AB_reduce_mean = tf.reduce_mean(A)
print(AB_reduce_mean.numpy())
```

    4

```python
# Reduce Mean with axis = 0:
AB_reduce_mean = tf.reduce_mean(A, 0)
print(AB_reduce_mean.numpy())
```

    [3 4 5 6]

```python
# Reduce sum:
AB_reduce_mean = tf.reduce_sum(A)
print(AB_reduce_mean.numpy())
```

    36

```python
# Reduce sum with axis = 0:
AB_reduce_mean = tf.reduce_sum(A, 0)
print(AB_reduce_mean.numpy())
```

    [ 6  8 10 12]

### Task 2:
#### Perform linear algebra operations using Tensorflow (transpose, matrix multiplication, elementwise multiplication, determinant)

```python
# Transposing matrix A:
A_transpose = tf.transpose(A)
print(A_transpose.numpy())
```

    [[1 5]
     [2 6]
     [3 7]
     [4 8]]

```python
# Matrix multiplication:
features = tf.constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = tf.constant([[1000], [150]])
AB_multiplication = tf.linalg.matmul(features, params)
print(AB_multiplication.numpy())
```

    [[ 5600]
     [ 5900]
     [10550]
     [ 6550]]

```python
A1 = tf.constant([1, 2, 3, 4])
A23 = tf.constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the correct shape
B1 = tf.ones_like(A1)
B23 = tf.ones_like(A23)

# Perform element-wise multiplication
C1 = tf.multiply(A1, B1)
C23 = tf.multiply(A23, B23)

# Print the tensors C1 and C23
print('C1: {}'.format(C1.numpy()))
print('C23: {}'.format(C23.numpy()))
```

    C1: [1 2 3 4]
    C23: [[1 2 3]
     [1 6 4]]

```python
# Element wise multiplication:
B1 = tf.ones_like(features)
AB_element_wise = tf.multiply(features, B1)
print(AB_element_wise.numpy())
```

    [[ 2 24]
     [ 2 26]
     [ 2 57]
     [ 1 37]]

```python
# Determinant:
A = tf.constant([[1, 2], [5, 6]], dtype=tf.float32)
A_determinant = tf.cast(A, dtype=tf.float32)
A_determinant = tf.linalg.det(A)
print(A_determinant.numpy())
```

    -4.0

### Task 3:
#### Perform derivative and higher order derivative for function f(x) = x^3 using gradient tape of Tensorflow

```python
# Perform derivative and higher order derivative for function f(x) = x^3 using gradient tape of Tensorflow
x = tf.Variable(4.0)

with tf.GradientTape(persistent=True) as tape:
  y = x**3
  dy_dx = tape.gradient(y, x)
  dy_dx_2 = tape.gradient(dy_dx, x)
print("First order derivative: ", dy_dx.numpy())
print("Second order derivative: ", dy_dx_2.numpy())
```

    WARNING:tensorflow:Calling GradientTape.gradient on a persistent tape inside its context is significantly less efficient than calling it outside the context (it causes the gradient ops to be recorded on the tape, leading to increased CPU and memory usage). Only call GradientTape.gradient inside the context if you actually want to trace the gradient in order to compute higher order derivatives.
    WARNING:tensorflow:Calling GradientTape.gradient on a persistent tape inside its context is significantly less efficient than calling it outside the context (it causes the gradient ops to be recorded on the tape, leading to increased CPU and memory usage). Only call GradientTape.gradient inside the context if you actually want to trace the gradient in order to compute higher order derivatives.

    First order derivative:  48.0
    Second order derivative:  24.0

### Task 4:
#### Compute WX+b using Tensorflow where W, X,  and b are drawn from a random normal distribution. W is of shape (4, 3), X is (3,1) and b is (4,1)

```python
W = tf.Variable(tf.random.normal((4, 3)))
X = tf.Variable(tf.random.normal((3, 1)))
B = tf.Variable(tf.random.normal((4, 1)))

print((tf.matmul(W, X) + B.numpy()))
```

    tf.Tensor(
    [[-2.2166047]
     [-1.7628429]
     [-0.9265769]
     [ 1.5388244]], shape=(4, 1), dtype=float32)

### Task 5:
#### Compute Gradient of sigmoid function using Tensor flow

```python
x = tf.constant(0.0)
with tf.GradientTape(persistent=True) as tape:
  tape.watch(x)
  y = tf.nn.sigmoid(x)

grad = tape.gradient(y, x) # dy/dx
print(grad.numpy())
```

    0.25

### Task 6:
#### Identify two research paper based on deep learning. State for which application they have used deep learning. (Cite the papers)

# Paper references found:
## Paper 1: https://ieeexplore.ieee.org/document/8876906
### How they have used Deep Learning?
#### Ans)
#### The site https://ieeexplore.ieee.org/document/8876932 uses deep learning in the following ways:

### The authors use a deep learning model to classify images of traffic signs. The model is trained on a dataset of images of traffic signs, and it is able to classify the images with high accuracy.
The authors also use deep learning to generate synthetic images of traffic signs. This is done by using a generative adversarial network (GAN). The GAN is trained on a dataset of real traffic signs, and it is able to generate realistic-looking synthetic traffic signs.
The authors use the deep learning models to improve the safety of autonomous vehicles. The models can be used to identify traffic signs, even in difficult conditions such as poor lighting or bad weather. This can help autonomous vehicles to avoid accidents.
Here is a more detailed explanation of how the deep learning models are used in this study:

The first model is used to classify images of traffic signs. The model is a convolutional neural network (CNN). CNNs are a type of deep learning model that are well-suited for image classification tasks. The CNN is trained on a dataset of images of traffic signs. The dataset contains images of traffic signs from different countries and in different conditions. The CNN is able to classify the images with high accuracy.
The second model is used to generate synthetic images of traffic signs. The model is a GAN. GANs are a type of deep learning model that can be used to generate realistic-looking images. The GAN is trained on a dataset of real traffic signs. The GAN is able to generate realistic-looking synthetic traffic signs that can be used to train other deep learning models.
The third model is used to improve the safety of autonomous vehicles. The model is a combination of the first two models. The model is able to identify traffic signs, even in difficult conditions such as poor lighting or bad weather. This can help autonomous vehicles to avoid accidents.
The use of deep learning in this study has the potential to improve the safety of autonomous vehicles. The deep learning models can be used to identify traffic signs, even in difficult conditions. This can help autonomous vehicles to avoid accidents and to improve the safety of the roads.
## Paper 2: https://ieeexplore.ieee.org/document/8876932
### How they have used Deep Learning?
#### Ans)
#### The site https://ieeexplore.ieee.org/document/8876932 does not use deep learning. The paper is about the use of generative adversarial networks (GANs) to generate synthetic images of traffic signs. GANs are a type of machine learning model, but they are not deep learning models. Deep learning models are a type of machine learning model that use artificial neural networks. GANs do not use artificial neural networks.

The authors of the paper argue that GANs can be used to generate synthetic images of traffic signs that can be used to train deep learning models.

So, to answer your question, the site https://ieeexplore.ieee.org/document/8876932 explains how DL can be implemented to learn models. It uses GANs, which are a type of machine learning model.

