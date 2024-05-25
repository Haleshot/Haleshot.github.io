---
title: Morphological Operations
subtitle: Enhancing Fingerprint Images with Morphological Techniques
summary: Applying morphological operations for noise removal in fingerprint images
date: '2023-03-16T00:00:00Z'
lastmod: '2024-05-25T00:00:00Z'
draft: false
featured: true
image:
 caption: 'Program output'
 focal_point: ''
 placement: 2
 preview_only: false
authors:
 - admin
tags:
 - Image Processing
 - Morphological Operations
 - Fingerprint Enhancement
 - Noise Removal
categories:
 - Programming
 - Computer Vision
 - Biometrics
 - Academic
---

# Morphological Operations for Noise Removal

## Aim

The aim of this project is to apply a suitable sequence of morphological operations on a given noisy fingerprint test image in order to obtain a noise-free image.

## Table of Contents

- [Aim](#aim)
- [Software](#software)
- [Prerequisite](#prerequisite)
- [Outcome](#outcome)
- [Theory](#theory)

## Software

This project is implemented using Python.

## Prerequisite

To understand and work with this project, you should be familiar with the following concepts:

| Sr. No | Concepts                               |
| ------ | -------------------------------------- |
| 1.     | Morphological operations (dilation, erosion, opening, and closing) |

## Outcome

After successfully completing this experiment, students will be able to:

- Implement erosion, dilation, opening, and closing operations to remove noise from an image.
- Can be found [here](https://github.com/Haleshot/Signal_Image_Processing/blob/main/Morphological_Operations/Morphological_Operations.ipynb).

## Theory

### Dilation

Dilation is a morphological operation that expands the boundaries of objects in an image. It is defined as follows:

A ⊕ B = { Z | [(B ⊖ Z) ∩ A] ∈ A }

In the above equation, A represents the image, and B is the structuring element. The symbol ⊕ denotes dilation. The operation involves taking the reflection of B about its origin and shifting it by Z. Dilation of A with B yields a set of all displacements, Z, such that the overlap of (B ⊖ Z) and A contains at least one element.

### Erosion

Erosion is a morphological operation that erodes or shrinks the boundaries of objects in an image. It is defined as follows:

A ⊖ B = { Z | (B ⊖ Z) ∈ A }

In the above equation, A represents the image, and B is the structuring element. The symbol ⊖ denotes erosion. The operation involves taking the reflection of B about its origin and shifting it by Z. Erosion of A by B yields a set of all points where B, translated (shifted by Z), is entirely contained within A. Erosion reduces the number of pixels from the object boundary.

### Opening

Morphological opening of an image is performed by first applying erosion followed by dilation:

A ∘ B = OPEN(A, B) = D(E(A))

In the above equation, A represents the image, B is the structuring element, D represents the dilation operation, and E represents the erosion operation. The symbol ∘ denotes opening. Opening is useful for removing noise and fine details from the image while preserving the overall shape of objects.

### Closing

Morphological closing of an image is performed by first applying dilation followed by erosion:

A ∙ B = CLOSE(A, B) = E(D(A))

In the above equation, A represents the image, B is the structuring element, D represents the dilation operation, and E represents the erosion operation. The symbol ∙ denotes closing. Closing is useful for filling small holes and gaps in objects while preserving their overall shape.

To remove noise from the given image, it is recommended to apply opening and closing operations in the correct order. This helps in reducing the noise and improving the overall quality of the image.


```python
# import libraries
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

# Morphology Application:

## Importing the image:

```python
img = cv2.imread(r"F:\New_Laptop_Documents\NMIMS_College_Docs\2nd_Year\2nd Semester\SIP\Practicals\Experiment_9\Fig0911(a)(noisy_fingerprint).tif", 0)
```

```python
# Since we demonstrated the effects of user defined functions with built in functions of cv2 and they yielded the same result,
# we now create built in functions for erosion and dilation to use them for opening and closing of the image
```

```python
# We first erode the image and then dilate it on the image obtained after erosion. This process is called opening.
# Then we again dilate the image which we obtained after dilation and then erode the newly obtained image. This process is called closing.
```

```python
# We define inbuilt functions of erosion and dilation. This is also done to initialize the size of the structuring element.
```

```python
struct_element_size = int(input("Enter Structuring Element Size : "))
SE= np.ones((struct_element_size, struct_element_size))
constant = (struct_element_size - 1)//2
```

    Enter Structuring Element Size : 15

```python
# Defining a function using Built in Function for erosion
def erosion(img, SE):
  img_erode = cv2.erode(img, SE, 1)
  return img_erode

# Defining a function using Built in Function for dilation
def dilation(img, SE):
  img_dilate = cv2.dilate(img, SE, 1)
  return img_dilate
```

```python
# Defining the structuring element
SE = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
```

## Opening the image:

```python
# Erode the image for  the first time:
Erosion_First_Time = erosion(img, SE)

# Dilate the eroded image.
Dilating_First_Time = dilation(Erosion_First_Time, SE)
```

```python
# Dilate the opened image followed by ersoion.
Dilating_Second_Time = dilation(Dilating_First_Time, SE)
Erosion_Second_Time = erosion(Dilating_Second_Time, SE)
```

```python
plt.figure(figsize = (15, 15))

plt.subplot(3, 2, 1)
plt.imshow(img, cmap = "gray")
plt.title("Original")

plt.subplot(3, 2, 2)
plt.title("Erosion First Time")
plt.imshow(Erosion_First_Time, cmap = "gray")

plt.subplot(3, 2, 3)
plt.title("Dilating First Time")
plt.imshow(Dilating_First_Time, cmap = "gray")

plt.subplot(3, 2, 4)
plt.title("Dilating Second Time")
plt.imshow(Dilating_Second_Time, cmap = "gray")

plt.subplot(3, 2, 5)
plt.title("Erosion Second Time")
plt.imshow(Erosion_Second_Time, cmap = "gray")
```

    <matplotlib.image.AxesImage at 0x2d1f53eb5e0>

    
![png](output_13_1.png)
    

```python
radii = ["Original", "Erosion First Time", "Dilating First Time", "Dilating Second Time", "Erosion Second Time"]
images = [img, Erosion_First_Time, Dilating_First_Time, Dilating_Second_Time, Erosion_Second_Time]
plt.figure(figsize = (15, 15))
for i in range(len(radii)):    
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i], cmap = "gray", vmin = 0, vmax = 255)
    plt.title("{}".format(radii[i]))
    plt.xticks([])
    plt.yticks([])
```

    
![png](output_14_0.png)
    

```python
radii = ["Original", "Opening the Image", "Closing the Image"]
images = [img, Dilating_First_Time, Erosion_Second_Time]
plt.figure(figsize = (15, 15))
for i in range(len(radii)):    
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap = "gray", vmin = 0, vmax = 255)
    plt.title("{}".format(radii[i]))
    plt.xticks([])
    plt.yticks([])
```

    
![png](output_15_0.png)
    

