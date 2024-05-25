---
title: Discrete Fourier Transform
date: '2024-05-24'
---
```python
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 # You can also use PIL as alternative.
from google.colab.patches import cv2_imshow
import math
```

# Task

```python
# Defining an array:
x = np.array([1, 2, 1, 2])
```

## Formulae used to implement in user defined functions:
<Insert later>

### Obtaining Twiddle factor matrix for Forward DFT:

```python
N = 4
l = np.zeros((4, 4), dtype = complex) # We get the values in cosθ - jsinθ due to which we need the array to be of complex datatypes.
for i in range(N):
  for j in range(N):
    real_part = int(math.cos(2 * math.pi * i * j/N)) 
    img_part = int(math.sin(2 * math.pi * i * j/N)) * -1
    l[i, j] = complex(real_part, img_part)
print(l)
```

    [[ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
     [ 1.+0.j  0.-1.j -1.+0.j  0.+1.j]
     [ 1.+0.j -1.+0.j  1.+0.j -1.+0.j]
     [ 1.+0.j  0.+1.j -1.+0.j  0.-1.j]]

### Getting DFT for above DFT twiddle factor matrix:

```python
A = np.dot(l, np.transpose(x))
print(A)
```

    [ 6.+0.j  0.+0.j -2.+0.j  0.+0.j]

### Obtaining Twiddle factor matrix for Inverse DFT:

```python
N = 4
inv_l = np.zeros((4, 4), dtype = complex) # We get the values in cosθ - jsinθ due to which we need the array to be of complex datatypes.
for i in range(N):
  for j in range(N):
    real_part = int(math.cos(2 * math.pi * i * j/N)) 
    img_part = int(math.sin(2 * math.pi * i * j/N))
    inv_l[i, j] = complex(real_part, img_part)
print(inv_l)
```

    [[ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
     [ 1.+0.j  0.+1.j -1.+0.j  0.-1.j]
     [ 1.+0.j -1.+0.j  1.+0.j -1.+0.j]
     [ 1.+0.j  0.-1.j -1.+0.j  0.+1.j]]

### Getting Invserse DFT for above Inverse DFT twiddle factor matrix:

```python
B = np.dot(inv_l, A)/N
print(B)
```

    [1.+0.j 2.+0.j 1.+0.j 2.+0.j]

## Comparing the above values of user defined algorithms to find DFT and FFT with the built in Transform functions of Numpy

```python
P = np.fft.fft(x) # Obtaining the Forward Fourier Transform for the given sequence using numpy.fft.fft
Q = np.fft.ifft(x) # Obtaining the Inverse Fourier Transform for the given sequence using numpy.fft.ifft
print(P)
```

    [ 6.+0.j  0.+0.j -2.+0.j  0.+0.j]

```python
print(Q)
```

    [ 1.5+0.j  0. +0.j -0.5+0.j  0. +0.j]

# Hence from the given cell references, we see that the built in function answer and the one defined in Numpy match.
## User Defined answers in cells:

1.   [Forward Fourier Transform](#scrollTo=HR667Q06jNSu&line=1&uniqifier=1)
2.   [Inverse Fourier Transform](#scrollTo=dU9hJQ5LkOti&line=2&uniqifier=1)

## Built in functions answers cells
1.   [Forward Fourier Transform](#scrollTo=OtdVaoAak0Me&line=3&uniqifier=1)
2.   [Inverse Fourier Transform](#scrollTo=kb9QVQXwmkCb&line=1&uniqifier=1)

# From the above experiment, I learnt the following:
1.	Obtaining the twiddle factor matrix .
2.	To find the DFT and IDFT of a given sequence using twiddle factor matrix.  
3.	To compute the DFT and IDFT using matrix method/user defined and built in fft and ifft function.
4. Implement Forward and inverse DFT on the given sequence.
