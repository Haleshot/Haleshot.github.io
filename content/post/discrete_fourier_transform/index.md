---
title: Discrete Fourier Transform
subtitle: Experimenting with DFT, IDFT and matrices!
summary: Experimenting with DFT, IDFT and matrices!
date: '2023-03-07T00:00:00Z'
lastmod: '2024-05-25T00:00:00Z'
draft: false
featured: true
image:
  caption: 'Formula used'
  focal_point: ''
  placement: 2
  preview_only: false
authors:
  - admin
tags:
  - Signal Processing
  - Fourier Transform
  - Matrices
  - Image Processing
categories:
  - Programming
  - Mathematics
  - Academic
---

# Discrete Fourier Transform (DFT) in Python

## Aim

The aim of this project is to obtain the twiddle factor matrix and perform the following tasks using it:

- Find the DFT and IDFT of the sequence [1, 2, 2, 1] using the twiddle factor matrix.
- Compute the DFT using matrix method and the FFT function.
- Observe and comment on the execution time required for each of the above methods.

## Table of Contents

- [Aim](#aim)
- [Software](#software)
- [Prerequisite](#prerequisite)
- [Outcome](#outcome)
- [Theory](#theory)
  - [Discrete Fourier Transform (DFT)](#discrete-fourier-transform-dft)
- [Algorithm](#algorithm)

## Software

This project is implemented using Python.

## Prerequisite

To understand and work with this project, you should be familiar with the following concepts:

| Sr. No | Concepts                     |
| ------ | ---------------------------- |
| 1.     | Discrete Fourier Transform   |

## Outcome

After successful completion of this experiment, students will be able to:

- Implement inverse and forward DFT on the given sequence.
- Can be found [here](https://github.com/Haleshot/Signal_Image_Processing/blob/main/Discrete_Fourier_Transform/Discrete_Fourier_Transform.ipynb).

## Theory

### Discrete Fourier Transform (DFT)

The Discrete Fourier Transform (DFT) is used for performing frequency analysis of discrete time signals. DFT provides a discrete frequency domain representation, whereas other transforms are continuous in the frequency domain.

The N-point DFT of a discrete time signal x[n] is given by the equation:

X(k) = Σ(x(n) * W<sub>N</sub><sup>nk</sup>)

Where N is chosen such that N ≥ L (length of x[n]).
The inverse DFT allows us to recover the sequence x[n] from the frequency samples.

The twiddle factor is given by:

W<sub>N</sub><sup>nk</sup> = e<sup>(-jθ)</sup> = cos(θ) - j sin(θ)

Where θ = (2πnk) / N, n = 0 to N-1, and k = 0 to N-1.

### Algorithm

The algorithm for performing the DFT using the twiddle factor matrix is as follows:

1. Assign a value for the time period, N.
2. Assign the frequency and sampling frequency.
3. Give the sampling period rate, n.
4. Define the sine function: sin(2π(f/fs)n).
5. Find the DFT using the built-in function `fft()`.
6. Find the DFT using the formula mentioned above.
7. Plot the input function.
8. Plot the DFT (absolute value) when the built-in function is used.
9. Plot the DFT (absolute value) when the DFT formula is used.
10. Display the output.


## Formulae used to implement in user defined functions

### DFT

![png](DFT.png)

### Inverse DFT

![png](IDFT.png)

### Obtaining Twiddle factor matrix for Forward DFT

![png](Twiddle_Factor_Matrix.png)

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

### Getting DFT for above DFT twiddle factor matrix

```python
A = np.dot(l, np.transpose(x))
print(A)
```

    [ 6.+0.j  0.+0.j -2.+0.j  0.+0.j]

### Obtaining Twiddle factor matrix for Inverse DFT

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

### Getting Invserse DFT for above Inverse DFT twiddle factor matrix

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


# Conclusion:

1. Obtaining the twiddle factor matrix .
2. To find the DFT and IDFT of a given sequence using twiddle factor matrix.  
3. To compute the DFT and IDFT using matrix method/user defined and built in fft and ifft function.
4. Implement Forward and inverse DFT on the given sequence.
