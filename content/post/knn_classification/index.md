---
title: Knn Classification
subtitle: Exploring the K-Nearest Neighbors Algorithm for Classification
summary: Implementing the KNN algorithm for classification tasks using Python
date: '2023-02-16T00:00:00Z'
lastmod: '2024-05-26T00:00:00Z'
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
 - Machine Learning
 - Classification
 - KNN
 - K-Nearest Neighbors
 - Python
 - Pandas
 - NumPy
 - Matplotlib
 - Seaborn
categories:
 - Programming
 - Data Science
 - Algorithms
 - Academic
---

# K-Nearest Neighbor (KNN) Algorithm for Classification

This README file provides instructions and information for implementing the KNN algorithm for classification. The experiment requires prior knowledge of Python programming and the following libraries: Pandas, NumPy, Matplotlib, and Seaborn.

## Table of Contents
1. [Aim](#aim)
2. [Prerequisite](#prerequisite)
3. [Outcome](#outcome)
4. [Theory](#theory)
5. [Task 1: Implementing KNN Algorithm](#task-1)
6. [Task 2: Operations on Social_Network_Ads.csv Dataset](#task-2)
7. [Task 3: Implementing KNN Algorithm on Sklearn Dataset](#task-3)

<a name="aim"></a>
## Aim
The aim of this project is to implement the KNN algorithm for classification.

<a name="prerequisite"></a>
## Prerequisite
To successfully complete this experiment, you should have knowledge of Python programming and the following libraries: Pandas, NumPy, Matplotlib, and Seaborn.

<a name="outcome"></a>
## Outcome
After successfully completing this experiment, you will be able to:
1. Implement the KNN technique for classification.
2. Can be found [here](https://github.com/Haleshot/AI-ML/blob/master/KNN_Classification/KNN_Classification.ipynb).

<a name="theory"></a>
## Theory
### K-Nearest Neighbor (KNN)
KNN is one of the simplest machine learning algorithms based on the supervised learning technique. It assumes similarity between new data and available data and assigns the new case to the category that is most similar to the available categories. KNN can be used to classify new data into suitable categories.

The steps involved in KNN are as follows:
1. Step 1: Select the number K of neighbors.
2. Step 2: Calculate the Euclidean distance of K number of neighbors.
3. Step 3: Take the K nearest neighbors based on the calculated Euclidean distance.
4. Step 4: Count the number of data points in each category among these K neighbors.
5. Step 5: Assign the new data points to the category for which the number of neighbors is maximum.
6. Step 6: Model is ready!

To determine a good value of K:
- It is generally determined experimentally.
- Initialize K with 1 and use the test set for classifier validation (accuracy).
- Increment the value of K and repeat the procedure.
- Choose the value of K that shows the minimum error.

Feature transformation is an important aspect of KNN. It involves normalizing or standardizing the data to fall within a smaller or common range. This helps prevent attributes with initially large ranges from outweighing attributes with initially smaller ranges.

Tasks are provided to help you understand and implement the KNN algorithm.

<a name="task-1"></a>
## Task 1: Implementing KNN Algorithm
Perform the following tasks to implement the KNN algorithm:

For the test data (1,1), determine the class using:
- 3-KNN
- 5-KNN
- 7-KNN

Use the provided dataset as the training dataset. Write your own user-defined function to implement KNN.

<a name="task-2"></a>
## Task 2: Operations on Social_Network_Ads.csv Dataset
Perform the following tasks on the given `Social_Network_Ads.csv` dataset:

1. Upload the dataset and store it in a Pandas DataFrame.
2. Explore the dataset using the `head()`, `describe()`, and `size` commands.
3. Identify the input features and output feature.
4. Identify the total number of classes in the output feature.
5. Remove columns that are not useful for classification.
6. Convert categorical columns into numeric columns.
7. Apply scalar transformation.

<a name="task-3"></a>
## Task 3: Implementing KNN Algorithm on Sklearn Dataset
Perform the following tasks to implement the KNN algorithm on the given dataset from the Sklearn library with K=5:

1. Split the dataset into train and test sets.
2. Fit the KNN model on the train dataset.
3. Identify the class for the test dataset.
4. Print the confusion matrix.
5. Print the accuracy score.
6. Write your inference based on the results.

Follow the instructions provided for each task and analyze the results to gain a better understanding of the KNN algorithm.

```python
# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
```

```python
# Eucledian Distance Function:
x2 = 1
y2 = 1
def Eucledian(x1, y1):
  result = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
  return result
```

# Task 1:

```python
x = [[-1, 1], [0, 1], [0, 2], [1, -1], [1, 0], [1, 2], [2, 2], [2, 3]]
```

```python
print(x)
```

    [[-1, 1], [0, 1], [0, 2], [1, -1], [1, 0], [1, 2], [2, 2], [2, 3]]

```python
Euc_list = []
for i in x:
  a, b = i
  temp = Eucledian(a, b)
  Euc_list.append(temp)
```

```python
print(Euc_list)
```

    [2.0, 1.0, 1.4142135623730951, 2.0, 1.0, 1.0, 1.4142135623730951, 2.23606797749979]

```python
c_label = ["N", "P", "N", "N", "P", "P", "P", "N", "P"]
```

```python
dataset_given = [[-1, 1, "N"], [0, 1, "P"], [0, 2, "N"], [1, -1, "N"], [1, 0, "P"], [1, 2, "P"], [2, 2, "N"], [2, 2, "P"]]
```

```python
mapped_list = []
for i in range(len(Euc_list)):
  mapped_list.append([Euc_list[i], dataset_given[i][2]])
```

```python
print(mapped_list)
mapped_list.sort(key = lambda mapped_list: mapped_list[0])
```

    [[1.0, 'P'], [1.0, 'P'], [1.0, 'P'], [1.4142135623730951, 'N'], [1.4142135623730951, 'N'], [2.0, 'N'], [2.0, 'N'], [2.23606797749979, 'P']]

```python
def knn(k, data):
  N_Counter, P_Counter = 0, 0
  print("For k value = ", k)
  for i in range(k):
    x = data[i][1]
    if x == "P":
      P_Counter += 1
    else:
      N_Counter += 1
  if N_Counter > P_Counter:
    print("N")
  else:
    print("P")
```

```python
knn(3, mapped_list)
knn(5, mapped_list)
knn(7, mapped_list)
```

    For k value =  3
    P
    For k value =  5
    P
    For k value =  7
    N

```python
# Doing a similar technique for a dataset which is given.
```

```python
df = pd.read_csv("/content/test_knn.csv")
```

```python
x2, y2 = 2, 2
```

```python
c_label = ["0", "0", "0", "0", "0", "1", "1", "1", "1", "1"]
```

```python
a = df["a"]
b = df["b"]
c = df["c"]
dataset_list = []
for i in range(len(df)):
  data = [a[i], b[i], c[i]]
  dataset_list.append(data)
```

```python
print(dataset_list)
```

    [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0], [3.396561688, 4.400293529, 0], [1.38807019, 1.850220317, 0], [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1], [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1], [8.675418651, -0.242068655, 1], [7.673756466, 3.508563011, 1]]

```python
Euc_list = []
for i in dataset_list:
  a, b, c = i
  temp = Eucledian(a, b)
  Euc_list.append(temp)
print(Euc_list)
```

    [0.9556058716129691, 0.6456285171156554, 2.777011626505853, 0.6299936871161619, 1.463861332756508, 5.678519763639245, 3.3336195608705412, 4.927917437576228, 7.041880858112566, 5.870883646746391]

To do

```python
mapped_list = []
c = df["c"]
for i in range(len(Euc_list)):
  mapped_list.append([Euc_list[i], c[i]])

mapped_list.sort(key = lambda mapped_list: mapped_list[0])
print(mapped_list)
```

    [[0.6299936871161619, 0], [0.6456285171156554, 0], [0.9556058716129691, 0], [1.463861332756508, 0], [2.777011626505853, 0], [3.3336195608705412, 1], [4.927917437576228, 1], [5.678519763639245, 1], [5.870883646746391, 1], [7.041880858112566, 1]]

```python
def knn(k, data):
  N_Counter, P_Counter = 0, 0
  print("For k value = ", k)
  for i in range(k):
    x = data[i][1]
    if x == 1:
      P_Counter += 1
    else:
      N_Counter += 1
  if N_Counter > P_Counter:
    print("0")
  else:
    print("1")
```

```python
knn(3, mapped_list)
knn(5, mapped_list)
knn(7, mapped_list)
```

    For k value =  3
    0
    For k value =  5
    0
    For k value =  7
    0

# Task 2:
![image](Task2.png)

```python
df = pd.read_csv("/content/Social_Network_Ads.csv")
```

## EDA:

```python
df
```

  <div id="df-5f72628a-2cf8-497c-8aa8-0d054815b2b3">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>EstimatedSalary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15624510</td>
      <td>Male</td>
      <td>19</td>
      <td>19000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15810944</td>
      <td>Male</td>
      <td>35</td>
      <td>20000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15668575</td>
      <td>Female</td>
      <td>26</td>
      <td>43000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15603246</td>
      <td>Female</td>
      <td>27</td>
      <td>57000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15804002</td>
      <td>Male</td>
      <td>19</td>
      <td>76000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>395</th>
      <td>15691863</td>
      <td>Female</td>
      <td>46</td>
      <td>41000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>396</th>
      <td>15706071</td>
      <td>Male</td>
      <td>51</td>
      <td>23000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>15654296</td>
      <td>Female</td>
      <td>50</td>
      <td>20000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>398</th>
      <td>15755018</td>
      <td>Male</td>
      <td>36</td>
      <td>33000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>399</th>
      <td>15594041</td>
      <td>Female</td>
      <td>49</td>
      <td>36000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 5 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5f72628a-2cf8-497c-8aa8-0d054815b2b3')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-5f72628a-2cf8-497c-8aa8-0d054815b2b3 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5f72628a-2cf8-497c-8aa8-0d054815b2b3');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

```python
df.head
```

    <bound method NDFrame.head of       User ID  Gender  Age  EstimatedSalary  Purchased
    0    15624510    Male   19            19000          0
    1    15810944    Male   35            20000          0
    2    15668575  Female   26            43000          0
    3    15603246  Female   27            57000          0
    4    15804002    Male   19            76000          0
    ..        ...     ...  ...              ...        ...
    395  15691863  Female   46            41000          1
    396  15706071    Male   51            23000          1
    397  15654296  Female   50            20000          1
    398  15755018    Male   36            33000          0
    399  15594041  Female   49            36000          1
    
    [400 rows x 5 columns]>

```python
df.describe()
```

  <div id="df-396001a2-881b-4623-80c4-95b6230a03a8">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User ID</th>
      <th>Age</th>
      <th>EstimatedSalary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.000000e+02</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.569154e+07</td>
      <td>37.655000</td>
      <td>69742.500000</td>
      <td>0.357500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.165832e+04</td>
      <td>10.482877</td>
      <td>34096.960282</td>
      <td>0.479864</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.556669e+07</td>
      <td>18.000000</td>
      <td>15000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.562676e+07</td>
      <td>29.750000</td>
      <td>43000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.569434e+07</td>
      <td>37.000000</td>
      <td>70000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.575036e+07</td>
      <td>46.000000</td>
      <td>88000.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.581524e+07</td>
      <td>60.000000</td>
      <td>150000.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-396001a2-881b-4623-80c4-95b6230a03a8')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-396001a2-881b-4623-80c4-95b6230a03a8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-396001a2-881b-4623-80c4-95b6230a03a8');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

```python
df.shape
```

    (400, 5)

```python
df.size
```

    2000

### Binary classification dataset since Purchased is a label category containing 0s and 1s only (0 if the User hasn't purchased, 1 if the user has purchased).

### The values shown below tell us how many samples have 0 and 1 Purchase values respectively.

```python
df["Purchased"].value_counts()
```

    0    257
    1    143
    Name: Purchased, dtype: int64

### Gender Age and Estimated Salary can be considered as Input features.
### Purchased Column can be considered to be the Output feature.

## We'll now use Label Encoding to convert the Gender categorical feature to Numerical feature.

```python
df2 = df
df2
```

  <div id="df-0d07654f-634b-4cff-a4b4-1cd64df25513">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>EstimatedSalary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15624510</td>
      <td>Male</td>
      <td>19</td>
      <td>19000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15810944</td>
      <td>Male</td>
      <td>35</td>
      <td>20000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15668575</td>
      <td>Female</td>
      <td>26</td>
      <td>43000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15603246</td>
      <td>Female</td>
      <td>27</td>
      <td>57000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15804002</td>
      <td>Male</td>
      <td>19</td>
      <td>76000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>395</th>
      <td>15691863</td>
      <td>Female</td>
      <td>46</td>
      <td>41000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>396</th>
      <td>15706071</td>
      <td>Male</td>
      <td>51</td>
      <td>23000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>15654296</td>
      <td>Female</td>
      <td>50</td>
      <td>20000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>398</th>
      <td>15755018</td>
      <td>Male</td>
      <td>36</td>
      <td>33000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>399</th>
      <td>15594041</td>
      <td>Female</td>
      <td>49</td>
      <td>36000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 5 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0d07654f-634b-4cff-a4b4-1cd64df25513')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0d07654f-634b-4cff-a4b4-1cd64df25513 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0d07654f-634b-4cff-a4b4-1cd64df25513');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

```python
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df2['Gender'] = label_encoder.fit_transform(df2['Gender'])
```

```python
df2
```

  <div id="df-9adf9c37-3e03-4bb4-a5f8-cc98a8362de0">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>EstimatedSalary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15624510</td>
      <td>1</td>
      <td>19</td>
      <td>19000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15810944</td>
      <td>1</td>
      <td>35</td>
      <td>20000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15668575</td>
      <td>0</td>
      <td>26</td>
      <td>43000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15603246</td>
      <td>0</td>
      <td>27</td>
      <td>57000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15804002</td>
      <td>1</td>
      <td>19</td>
      <td>76000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>395</th>
      <td>15691863</td>
      <td>0</td>
      <td>46</td>
      <td>41000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>396</th>
      <td>15706071</td>
      <td>1</td>
      <td>51</td>
      <td>23000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>15654296</td>
      <td>0</td>
      <td>50</td>
      <td>20000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>398</th>
      <td>15755018</td>
      <td>1</td>
      <td>36</td>
      <td>33000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>399</th>
      <td>15594041</td>
      <td>0</td>
      <td>49</td>
      <td>36000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 5 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9adf9c37-3e03-4bb4-a5f8-cc98a8362de0')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9adf9c37-3e03-4bb4-a5f8-cc98a8362de0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9adf9c37-3e03-4bb4-a5f8-cc98a8362de0');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

```python
df2 = df2.drop(['User ID'], axis = 1)
df2
```

  <div id="df-9d6b2658-ebdf-4efa-969e-3fda0a6a782b">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>EstimatedSalary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>19</td>
      <td>19000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>35</td>
      <td>20000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>26</td>
      <td>43000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>27</td>
      <td>57000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>19</td>
      <td>76000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>395</th>
      <td>0</td>
      <td>46</td>
      <td>41000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>396</th>
      <td>1</td>
      <td>51</td>
      <td>23000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>0</td>
      <td>50</td>
      <td>20000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>398</th>
      <td>1</td>
      <td>36</td>
      <td>33000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>399</th>
      <td>0</td>
      <td>49</td>
      <td>36000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 4 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9d6b2658-ebdf-4efa-969e-3fda0a6a782b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9d6b2658-ebdf-4efa-969e-3fda0a6a782b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9d6b2658-ebdf-4efa-969e-3fda0a6a782b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

# Task 3:
![image](Task3.png)


```python
from sklearn.model_selection import train_test_split
x = df.iloc[:, [1, 2, 3]].values # Stores the value of input features on which we'll train the Model
print(x)
```

    [[    1    19 19000]
     [    1    35 20000]
     [    0    26 43000]
     ...
     [    0    50 20000]
     [    1    36 33000]
     [    0    49 36000]]

```python
y = df2.iloc[:, -1].values # Stores the value of output feature which we'll predict based on the model value.
print(y)
```

    [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
     0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0
     0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 0 1 1 0 0 0 1 0 0 0 1 0 1
     1 1 0 0 1 1 0 1 1 0 1 1 0 1 0 0 0 1 1 0 1 1 0 1 0 1 0 1 0 0 1 1 0 1 0 0 1
     1 0 1 1 0 1 1 0 0 1 0 0 1 1 1 1 1 0 1 1 1 1 0 1 1 0 1 0 1 0 1 1 1 1 0 0 0
     1 1 0 1 1 1 1 1 0 0 0 1 1 0 0 1 0 1 0 1 1 0 1 0 1 1 0 1 1 0 0 0 1 1 0 1 0
     0 1 0 1 0 0 1 1 0 0 1 1 0 1 1 0 0 1 0 1 0 1 1 1 0 1 0 1 1 1 0 1 1 1 1 0 1
     1 1 0 1 0 1 0 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 0 1]

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# We define test_size of the 400 samples to be 0.2 = 20%
# We also define the random_state as 0 for a constant value as different samples will be picked up
# for train and test everytime.
```

```python
x_train.shape
```

    (320, 3)

```python
x_test.shape
```

    (80, 3)

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
```

```python
x_train
```

    array([[ 1.02532046e+00,  1.92295008e+00,  2.14601566e+00],
           [-9.75304830e-01,  2.02016082e+00,  3.78719297e-01],
           [-9.75304830e-01, -1.38221530e+00, -4.32498705e-01],
           [-9.75304830e-01, -1.18779381e+00, -1.01194013e+00],
           [-9.75304830e-01,  1.92295008e+00, -9.25023920e-01],
           [-9.75304830e-01,  3.67578135e-01,  2.91803083e-01],
           [-9.75304830e-01,  1.73156642e-01,  1.46942725e-01],
           [ 1.02532046e+00,  2.02016082e+00,  1.74040666e+00],
           [-9.75304830e-01,  7.56421121e-01, -8.38107706e-01],
           [-9.75304830e-01,  2.70367388e-01, -2.87638347e-01],
           [ 1.02532046e+00,  3.67578135e-01, -1.71750061e-01],
           [-9.75304830e-01, -1.18475597e-01,  2.20395980e+00],
           [-9.75304830e-01, -1.47942605e+00, -6.35303205e-01],
           [ 1.02532046e+00, -1.28500455e+00, -1.06988428e+00],
           [ 1.02532046e+00, -1.38221530e+00,  4.07691369e-01],
           [-9.75304830e-01, -1.09058306e+00,  7.55356227e-01],
           [ 1.02532046e+00, -1.47942605e+00, -2.00722133e-01],
           [ 1.02532046e+00,  9.50842613e-01, -1.06988428e+00],
           [ 1.02532046e+00,  9.50842613e-01,  5.81523798e-01],
           [ 1.02532046e+00,  3.67578135e-01,  9.87132798e-01],
           [-9.75304830e-01,  5.61999628e-01, -8.96051849e-01],
           [ 1.02532046e+00, -6.04529329e-01,  1.45068594e+00],
           [-9.75304830e-01, -2.12648508e-02, -5.77359062e-01],
           [-9.75304830e-01, -6.04529329e-01,  1.88526701e+00],
           [-9.75304830e-01,  1.33968560e+00, -1.41754914e+00],
           [-9.75304830e-01,  1.43689635e+00,  9.87132798e-01],
           [ 1.02532046e+00,  7.59458956e-02, -8.09135634e-01],
           [ 1.02532046e+00, -2.12648508e-02, -2.58666276e-01],
           [-9.75304830e-01, -2.15686344e-01, -5.77359062e-01],
           [-9.75304830e-01, -2.15686344e-01, -2.00722133e-01],
           [-9.75304830e-01, -3.12897090e-01, -1.30166085e+00],
           [ 1.02532046e+00, -3.12897090e-01, -5.77359062e-01],
           [ 1.02532046e+00,  3.67578135e-01,  8.89985821e-02],
           [-9.75304830e-01,  8.53631867e-01, -6.06331134e-01],
           [-9.75304830e-01,  2.02016082e+00, -1.18577256e+00],
           [ 1.02532046e+00,  1.04805336e+00, -1.42777990e-01],
           [-9.75304830e-01,  6.59210374e-01,  1.76937873e+00],
           [-9.75304830e-01, -7.01740076e-01,  5.52551726e-01],
           [-9.75304830e-01,  7.56421121e-01,  3.49747226e-01],
           [-9.75304830e-01,  8.53631867e-01, -5.48386991e-01],
           [ 1.02532046e+00, -1.18779381e+00, -1.59138156e+00],
           [ 1.02532046e+00,  2.11737157e+00,  9.29188655e-01],
           [-9.75304830e-01, -2.12648508e-02,  1.21890937e+00],
           [ 1.02532046e+00,  1.73156642e-01,  1.07404901e+00],
           [-9.75304830e-01,  3.67578135e-01, -4.90442848e-01],
           [ 1.02532046e+00, -3.12897090e-01, -3.16610419e-01],
           [ 1.02532046e+00,  9.50842613e-01, -8.38107706e-01],
           [-9.75304830e-01,  9.50842613e-01,  1.85629494e+00],
           [-9.75304830e-01, -2.12648508e-02,  1.24788144e+00],
           [ 1.02532046e+00, -8.96161568e-01,  2.26190394e+00],
           [-9.75304830e-01, -1.18779381e+00, -1.59138156e+00],
           [ 1.02532046e+00,  2.11737157e+00, -8.09135634e-01],
           [ 1.02532046e+00, -1.38221530e+00, -1.47549328e+00],
           [ 1.02532046e+00,  3.67578135e-01,  2.29087602e+00],
           [-9.75304830e-01,  7.56421121e-01,  7.55356227e-01],
           [ 1.02532046e+00, -9.93372315e-01, -3.16610419e-01],
           [ 1.02532046e+00,  7.59458956e-02,  7.55356227e-01],
           [ 1.02532046e+00, -9.93372315e-01,  5.52551726e-01],
           [ 1.02532046e+00,  2.70367388e-01,  6.00265106e-02],
           [ 1.02532046e+00,  6.59210374e-01, -1.27268878e+00],
           [-9.75304830e-01, -5.07318583e-01, -2.68897039e-02],
           [-9.75304830e-01, -1.77105829e+00,  3.49747226e-01],
           [ 1.02532046e+00, -7.01740076e-01,  1.17970654e-01],
           [ 1.02532046e+00,  3.67578135e-01,  2.91803083e-01],
           [-9.75304830e-01, -3.12897090e-01,  6.00265106e-02],
           [-9.75304830e-01, -5.07318583e-01,  2.29087602e+00],
           [ 1.02532046e+00,  1.73156642e-01,  3.10544391e-02],
           [-9.75304830e-01,  1.24247485e+00,  2.20395980e+00],
           [ 1.02532046e+00,  7.56421121e-01,  2.62831011e-01],
           [ 1.02532046e+00, -3.12897090e-01,  1.46942725e-01],
           [ 1.02532046e+00, -2.12648508e-02, -5.48386991e-01],
           [-9.75304830e-01, -2.15686344e-01,  1.46942725e-01],
           [-9.75304830e-01, -1.18475597e-01,  2.33858940e-01],
           [ 1.02532046e+00, -2.12648508e-02, -2.58666276e-01],
           [-9.75304830e-01,  2.11737157e+00,  1.10302108e+00],
           [-9.75304830e-01, -1.77105829e+00,  3.49747226e-01],
           [ 1.02532046e+00,  1.82573933e+00,  1.17970654e-01],
           [ 1.02532046e+00,  3.67578135e-01, -1.42777990e-01],
           [ 1.02532046e+00, -1.18779381e+00,  2.91803083e-01],
           [ 1.02532046e+00,  7.56421121e-01,  1.36376973e+00],
           [ 1.02532046e+00, -3.12897090e-01, -2.58666276e-01],
           [-9.75304830e-01, -1.67384754e+00, -5.58617754e-02],
           [-9.75304830e-01, -9.93372315e-01, -7.51191491e-01],
           [ 1.02532046e+00,  2.70367388e-01,  4.94607583e-01],
           [-9.75304830e-01, -1.18475597e-01, -1.06988428e+00],
           [ 1.02532046e+00, -1.09058306e+00,  5.81523798e-01],
           [ 1.02532046e+00,  7.59458956e-02, -8.09135634e-01],
           [ 1.02532046e+00, -9.93372315e-01,  1.53760216e+00],
           [-9.75304830e-01, -7.01740076e-01,  1.39274180e+00],
           [ 1.02532046e+00, -1.28500455e+00,  4.94607583e-01],
           [-9.75304830e-01, -3.12897090e-01,  3.10544391e-02],
           [ 1.02532046e+00, -1.18475597e-01,  2.08236764e-03],
           [ 1.02532046e+00, -3.12897090e-01, -8.96051849e-01],
           [ 1.02532046e+00,  8.53631867e-01, -1.35960499e+00],
           [-9.75304830e-01, -3.12897090e-01,  2.23293187e+00],
           [-9.75304830e-01,  9.50842613e-01,  1.97218323e+00],
           [ 1.02532046e+00, -1.18779381e+00,  4.65635512e-01],
           [ 1.02532046e+00, -1.28500455e+00,  2.62831011e-01],
           [-9.75304830e-01,  1.33968560e+00,  1.97218323e+00],
           [ 1.02532046e+00,  1.24247485e+00, -1.35960499e+00],
           [-9.75304830e-01, -3.12897090e-01, -2.87638347e-01],
           [-9.75304830e-01, -5.07318583e-01,  1.24788144e+00],
           [ 1.02532046e+00, -7.98950822e-01,  1.07404901e+00],
           [ 1.02532046e+00,  9.50842613e-01, -1.06988428e+00],
           [-9.75304830e-01,  2.70367388e-01,  2.91803083e-01],
           [-9.75304830e-01,  9.50842613e-01,  7.55356227e-01],
           [ 1.02532046e+00, -7.01740076e-01, -1.50446535e+00],
           [-9.75304830e-01, -7.01740076e-01,  3.10544391e-02],
           [ 1.02532046e+00,  4.64788881e-01,  1.71143459e+00],
           [-9.75304830e-01,  2.02016082e+00,  1.75914797e-01],
           [-9.75304830e-01, -1.96547978e+00, -7.51191491e-01],
           [ 1.02532046e+00, -2.15686344e-01,  1.39274180e+00],
           [-9.75304830e-01,  3.67578135e-01,  5.81523798e-01],
           [-9.75304830e-01,  8.53631867e-01, -1.15680049e+00],
           [-9.75304830e-01, -1.18779381e+00, -7.80163563e-01],
           [ 1.02532046e+00,  1.73156642e-01,  2.33858940e-01],
           [ 1.02532046e+00,  7.56421121e-01, -3.16610419e-01],
           [-9.75304830e-01,  2.02016082e+00, -8.09135634e-01],
           [-9.75304830e-01,  7.56421121e-01,  1.17970654e-01],
           [ 1.02532046e+00, -3.12897090e-01,  6.10495869e-01],
           [-9.75304830e-01, -9.93372315e-01, -3.16610419e-01],
           [ 1.02532046e+00,  1.73156642e-01, -3.74554562e-01],
           [ 1.02532046e+00,  2.02016082e+00,  2.11704359e+00],
           [-9.75304830e-01,  1.82573933e+00, -1.27268878e+00],
           [-9.75304830e-01,  1.33968560e+00, -9.25023920e-01],
           [-9.75304830e-01,  8.53631867e-01,  1.24788144e+00],
           [-9.75304830e-01,  1.43689635e+00,  2.11704359e+00],
           [ 1.02532046e+00, -3.12897090e-01, -1.24371671e+00],
           [-9.75304830e-01,  1.92295008e+00,  9.00216584e-01],
           [-9.75304830e-01,  6.59210374e-01, -7.22219420e-01],
           [-9.75304830e-01, -1.47942605e+00,  3.49747226e-01],
           [ 1.02532046e+00,  7.56421121e-01, -1.35960499e+00],
           [ 1.02532046e+00,  3.67578135e-01, -1.42777990e-01],
           [-9.75304830e-01, -9.93372315e-01,  4.07691369e-01],
           [ 1.02532046e+00, -2.12648508e-02, -3.16610419e-01],
           [-9.75304830e-01, -1.18779381e+00,  4.07691369e-01],
           [-9.75304830e-01, -8.96161568e-01, -1.21474464e+00],
           [-9.75304830e-01, -1.18475597e-01,  3.10544391e-02],
           [-9.75304830e-01, -1.57663679e+00, -4.32498705e-01],
           [-9.75304830e-01,  9.50842613e-01, -1.01194013e+00],
           [ 1.02532046e+00,  1.04805336e+00, -1.21474464e+00],
           [-9.75304830e-01, -2.12648508e-02, -1.42777990e-01],
           [-9.75304830e-01, -1.09058306e+00, -1.53343742e+00],
           [ 1.02532046e+00,  7.56421121e-01, -1.21474464e+00],
           [ 1.02532046e+00,  9.50842613e-01,  2.05909944e+00],
           [-9.75304830e-01, -1.18779381e+00, -1.53343742e+00],
           [-9.75304830e-01, -3.12897090e-01,  7.84328298e-01],
           [-9.75304830e-01,  7.59458956e-02, -3.16610419e-01],
           [-9.75304830e-01, -1.38221530e+00, -1.24371671e+00],
           [ 1.02532046e+00, -6.04529329e-01, -1.50446535e+00],
           [ 1.02532046e+00,  7.56421121e-01,  5.23579655e-01],
           [ 1.02532046e+00, -3.12897090e-01, -3.45582490e-01],
           [ 1.02532046e+00,  1.72852858e+00, -2.87638347e-01],
           [ 1.02532046e+00,  8.53631867e-01, -1.04091221e+00],
           [-9.75304830e-01,  1.73156642e-01,  6.00265106e-02],
           [ 1.02532046e+00, -6.04529329e-01,  8.71244512e-01],
           [-9.75304830e-01, -1.86826903e+00, -1.41754914e+00],
           [ 1.02532046e+00, -1.28500455e+00,  5.81523798e-01],
           [ 1.02532046e+00, -3.12897090e-01,  5.23579655e-01],
           [ 1.02532046e+00, -9.93372315e-01, -1.09885635e+00],
           [-9.75304830e-01,  1.14526411e+00, -1.44652121e+00],
           [ 1.02532046e+00,  1.73156642e-01, -3.16610419e-01],
           [-9.75304830e-01,  1.14526411e+00, -7.51191491e-01],
           [ 1.02532046e+00, -3.12897090e-01,  6.00265106e-02],
           [-9.75304830e-01,  1.73156642e-01,  2.08807152e+00],
           [-9.75304830e-01,  7.56421121e-01, -1.09885635e+00],
           [-9.75304830e-01,  7.59458956e-02,  3.10544391e-02],
           [ 1.02532046e+00, -1.77105829e+00,  1.17970654e-01],
           [ 1.02532046e+00, -8.96161568e-01,  1.46942725e-01],
           [ 1.02532046e+00, -7.01740076e-01,  1.75914797e-01],
           [ 1.02532046e+00,  8.53631867e-01, -1.30166085e+00],
           [ 1.02532046e+00,  1.73156642e-01, -2.58666276e-01],
           [ 1.02532046e+00, -4.10107836e-01,  1.21890937e+00],
           [-9.75304830e-01, -2.12648508e-02,  2.91803083e-01],
           [-9.75304830e-01,  3.67578135e-01,  1.46942725e-01],
           [-9.75304830e-01,  8.53631867e-01, -6.64275277e-01],
           [-9.75304830e-01,  7.59458956e-02,  1.46942725e-01],
           [ 1.02532046e+00, -1.86826903e+00, -1.30166085e+00],
           [-9.75304830e-01, -1.18475597e-01,  2.91803083e-01],
           [ 1.02532046e+00, -2.15686344e-01, -2.87638347e-01],
           [ 1.02532046e+00,  2.70367388e-01, -5.19414919e-01],
           [ 1.02532046e+00, -2.15686344e-01,  1.59554630e+00],
           [-9.75304830e-01,  9.50842613e-01, -1.18577256e+00],
           [-9.75304830e-01, -2.15686344e-01,  1.62451837e+00],
           [-9.75304830e-01,  1.24247485e+00,  1.85629494e+00],
           [-9.75304830e-01, -1.09058306e+00, -3.74554562e-01],
           [ 1.02532046e+00, -2.12648508e-02,  3.10544391e-02],
           [-9.75304830e-01,  7.59458956e-02, -2.58666276e-01],
           [-9.75304830e-01, -1.57663679e+00, -1.24371671e+00],
           [-9.75304830e-01, -5.07318583e-01, -2.87638347e-01],
           [ 1.02532046e+00,  9.50842613e-01,  1.17970654e-01],
           [-9.75304830e-01,  1.92295008e+00, -1.35960499e+00],
           [ 1.02532046e+00,  1.43689635e+00,  6.00265106e-02],
           [-9.75304830e-01, -6.04529329e-01,  1.36376973e+00],
           [ 1.02532046e+00,  1.53410709e+00,  2.08236764e-03],
           [ 1.02532046e+00, -7.98950822e-01,  2.91803083e-01],
           [-9.75304830e-01,  1.92295008e+00,  7.26384155e-01],
           [-9.75304830e-01, -1.18779381e+00, -5.19414919e-01],
           [ 1.02532046e+00,  6.59210374e-01,  2.62831011e-01],
           [ 1.02532046e+00, -1.38221530e+00, -4.32498705e-01],
           [ 1.02532046e+00,  1.73156642e-01,  1.46942725e-01],
           [-9.75304830e-01, -5.07318583e-01, -1.21474464e+00],
           [-9.75304830e-01,  5.61999628e-01,  2.00115530e+00],
           [ 1.02532046e+00, -1.57663679e+00, -1.50446535e+00],
           [-9.75304830e-01, -5.07318583e-01, -5.48386991e-01],
           [-9.75304830e-01,  4.64788881e-01,  1.82732287e+00],
           [-9.75304830e-01, -1.38221530e+00, -1.09885635e+00],
           [-9.75304830e-01,  7.56421121e-01, -1.38857706e+00],
           [ 1.02532046e+00, -3.12897090e-01, -4.32498705e-01],
           [-9.75304830e-01,  1.53410709e+00,  9.87132798e-01],
           [-9.75304830e-01,  9.50842613e-01,  1.42171387e+00],
           [ 1.02532046e+00, -3.12897090e-01, -4.90442848e-01],
           [ 1.02532046e+00, -1.18475597e-01,  2.14601566e+00],
           [-9.75304830e-01, -1.47942605e+00, -1.13805918e-01],
           [-9.75304830e-01, -1.18475597e-01,  1.94321116e+00],
           [ 1.02532046e+00, -7.01740076e-01, -3.45582490e-01],
           [-9.75304830e-01, -5.07318583e-01, -8.38107706e-01],
           [-9.75304830e-01,  6.59210374e-01, -1.38857706e+00],
           [ 1.02532046e+00, -7.98950822e-01, -1.59138156e+00],
           [ 1.02532046e+00, -1.86826903e+00, -1.47549328e+00],
           [ 1.02532046e+00,  1.04805336e+00,  1.17970654e-01],
           [ 1.02532046e+00,  7.59458956e-02,  1.50863009e+00],
           [ 1.02532046e+00, -3.12897090e-01,  8.89985821e-02],
           [ 1.02532046e+00,  7.59458956e-02,  3.10544391e-02],
           [ 1.02532046e+00, -1.38221530e+00, -1.35960499e+00],
           [-9.75304830e-01,  2.70367388e-01,  6.00265106e-02],
           [-9.75304830e-01, -8.96161568e-01,  3.78719297e-01],
           [-9.75304830e-01,  1.53410709e+00, -1.27268878e+00],
           [-9.75304830e-01, -3.12897090e-01, -7.51191491e-01],
           [ 1.02532046e+00, -1.18475597e-01,  1.46942725e-01],
           [-9.75304830e-01, -8.96161568e-01, -6.64275277e-01],
           [-9.75304830e-01, -7.01740076e-01, -5.58617754e-02],
           [ 1.02532046e+00,  3.67578135e-01, -4.61470776e-01],
           [ 1.02532046e+00, -7.98950822e-01,  1.88526701e+00],
           [-9.75304830e-01,  1.33968560e+00,  1.27685351e+00],
           [-9.75304830e-01,  1.14526411e+00, -9.82968063e-01],
           [ 1.02532046e+00,  1.72852858e+00,  1.82732287e+00],
           [ 1.02532046e+00, -8.96161568e-01, -2.58666276e-01],
           [ 1.02532046e+00, -7.98950822e-01,  5.52551726e-01],
           [ 1.02532046e+00, -1.18779381e+00, -1.56240949e+00],
           [ 1.02532046e+00, -5.07318583e-01, -1.12782842e+00],
           [-9.75304830e-01,  2.70367388e-01,  6.00265106e-02],
           [ 1.02532046e+00, -2.15686344e-01, -1.06988428e+00],
           [-9.75304830e-01,  1.63131784e+00,  1.59554630e+00],
           [-9.75304830e-01,  9.50842613e-01,  1.76937873e+00],
           [-9.75304830e-01,  2.70367388e-01,  3.10544391e-02],
           [-9.75304830e-01, -7.98950822e-01, -2.29694204e-01],
           [ 1.02532046e+00, -1.18475597e-01,  6.00265106e-02],
           [-9.75304830e-01,  2.70367388e-01, -2.00722133e-01],
           [-9.75304830e-01,  1.92295008e+00, -6.64275277e-01],
           [-9.75304830e-01, -7.98950822e-01,  1.33479766e+00],
           [ 1.02532046e+00, -1.77105829e+00, -6.06331134e-01],
           [ 1.02532046e+00, -1.18475597e-01,  1.17970654e-01],
           [ 1.02532046e+00,  2.70367388e-01, -3.16610419e-01],
           [ 1.02532046e+00,  1.04805336e+00,  5.52551726e-01],
           [ 1.02532046e+00, -9.93372315e-01,  2.62831011e-01],
           [-9.75304830e-01,  1.43689635e+00,  3.49747226e-01],
           [ 1.02532046e+00,  1.73156642e-01, -3.74554562e-01],
           [ 1.02532046e+00,  2.11737157e+00, -1.04091221e+00],
           [ 1.02532046e+00, -3.12897090e-01,  1.10302108e+00],
           [ 1.02532046e+00, -1.67384754e+00,  6.00265106e-02],
           [ 1.02532046e+00, -2.12648508e-02,  3.10544391e-02],
           [ 1.02532046e+00,  7.59458956e-02,  1.04507694e+00],
           [-9.75304830e-01, -1.18475597e-01, -3.74554562e-01],
           [-9.75304830e-01, -1.18779381e+00,  6.00265106e-02],
           [-9.75304830e-01, -3.12897090e-01, -1.35960499e+00],
           [-9.75304830e-01,  1.53410709e+00,  1.10302108e+00],
           [ 1.02532046e+00, -7.98950822e-01, -1.53343742e+00],
           [ 1.02532046e+00,  7.59458956e-02,  1.85629494e+00],
           [ 1.02532046e+00, -8.96161568e-01, -7.80163563e-01],
           [ 1.02532046e+00, -5.07318583e-01, -7.80163563e-01],
           [ 1.02532046e+00, -3.12897090e-01, -9.25023920e-01],
           [ 1.02532046e+00,  2.70367388e-01, -7.22219420e-01],
           [-9.75304830e-01,  2.70367388e-01,  6.00265106e-02],
           [-9.75304830e-01,  7.59458956e-02,  1.85629494e+00],
           [-9.75304830e-01, -1.09058306e+00,  1.94321116e+00],
           [-9.75304830e-01, -1.67384754e+00, -1.56240949e+00],
           [ 1.02532046e+00, -1.18779381e+00, -1.09885635e+00],
           [ 1.02532046e+00, -7.01740076e-01, -1.13805918e-01],
           [-9.75304830e-01,  7.59458956e-02,  8.89985821e-02],
           [ 1.02532046e+00,  2.70367388e-01,  2.62831011e-01],
           [-9.75304830e-01,  8.53631867e-01, -5.77359062e-01],
           [-9.75304830e-01,  2.70367388e-01, -1.15680049e+00],
           [-9.75304830e-01, -1.18475597e-01,  6.68440012e-01],
           [-9.75304830e-01,  2.11737157e+00, -6.93247348e-01],
           [ 1.02532046e+00, -1.28500455e+00, -1.38857706e+00],
           [-9.75304830e-01, -9.93372315e-01, -9.53995992e-01],
           [-9.75304830e-01, -2.12648508e-02, -4.32498705e-01],
           [-9.75304830e-01, -2.15686344e-01, -4.61470776e-01],
           [-9.75304830e-01, -1.77105829e+00, -9.82968063e-01],
           [-9.75304830e-01,  1.72852858e+00,  9.87132798e-01],
           [ 1.02532046e+00,  1.73156642e-01, -3.74554562e-01],
           [-9.75304830e-01,  3.67578135e-01,  1.10302108e+00],
           [-9.75304830e-01, -1.77105829e+00, -1.35960499e+00],
           [ 1.02532046e+00,  1.73156642e-01, -1.42777990e-01],
           [ 1.02532046e+00,  8.53631867e-01, -1.44652121e+00],
           [-9.75304830e-01, -1.96547978e+00,  4.65635512e-01],
           [ 1.02532046e+00, -3.12897090e-01,  2.62831011e-01],
           [-9.75304830e-01,  1.82573933e+00, -1.06988428e+00],
           [-9.75304830e-01, -4.10107836e-01,  6.00265106e-02],
           [-9.75304830e-01,  1.04805336e+00, -8.96051849e-01],
           [-9.75304830e-01, -1.09058306e+00, -1.12782842e+00],
           [ 1.02532046e+00, -1.86826903e+00,  2.08236764e-03],
           [-9.75304830e-01,  7.59458956e-02,  2.62831011e-01],
           [ 1.02532046e+00, -1.18779381e+00,  3.20775154e-01],
           [ 1.02532046e+00, -1.28500455e+00,  2.91803083e-01],
           [-9.75304830e-01, -9.93372315e-01,  4.36663440e-01],
           [ 1.02532046e+00,  1.63131784e+00, -8.96051849e-01],
           [-9.75304830e-01,  1.14526411e+00,  5.23579655e-01],
           [ 1.02532046e+00,  1.04805336e+00,  5.23579655e-01],
           [ 1.02532046e+00,  1.33968560e+00,  2.31984809e+00],
           [-9.75304830e-01, -3.12897090e-01, -1.42777990e-01],
           [ 1.02532046e+00,  3.67578135e-01, -4.61470776e-01],
           [ 1.02532046e+00, -4.10107836e-01, -7.80163563e-01],
           [ 1.02532046e+00, -1.18475597e-01, -5.19414919e-01],
           [-9.75304830e-01,  9.50842613e-01, -1.15680049e+00],
           [ 1.02532046e+00, -8.96161568e-01, -7.80163563e-01],
           [ 1.02532046e+00, -2.15686344e-01, -5.19414919e-01],
           [-9.75304830e-01, -1.09058306e+00, -4.61470776e-01],
           [-9.75304830e-01, -1.18779381e+00,  1.39274180e+00]])

```python
x_test
```

    array([[ 1.        , -0.49618606,  0.56021375],
           [-1.        ,  0.2389044 , -0.59133674],
           [ 1.        , -0.03675452,  0.18673792],
           [-1.        , -0.49618606,  0.31122986],
           [-1.        , -0.03675452, -0.59133674],
           [ 1.        , -0.77184498, -1.52502632],
           [-1.        , -0.40429975, -1.68064126],
           [ 1.        ,  0.05513178,  2.33422397],
           [-1.        , -1.59882175, -0.03112299],
           [ 1.        ,  1.06588117, -0.80919764],
           [ 1.        , -0.49618606, -0.62245972],
           [-1.        , -0.67995868, -0.43572181],
           [ 1.        ,  0.14701809, -0.43572181],
           [ 1.        ,  0.33079071,  0.24898389],
           [ 1.        , -1.41504914,  0.52909077],
           [-1.        , -0.31241345,  1.49390334],
           [ 1.        ,  0.14701809,  0.24898389],
           [ 1.        , -1.50693545,  0.49796778],
           [-1.        ,  1.80097163,  1.89850216],
           [ 1.        , -0.03675452, -1.46278035],
           [-1.        , -0.03675452, -0.6847057 ],
           [-1.        ,  1.06588117,  2.33422397],
           [-1.        ,  0.51456332, -0.56021375],
           [ 1.        ,  1.06588117,  1.1204275 ],
           [-1.        , -1.13939022, -1.27604243],
           [-1.        ,  1.24965379,  2.24085501],
           [-1.        , -0.67995868,  0.56021375],
           [ 1.        , -0.58807237,  0.34235285],
           [-1.        ,  0.14701809, -0.2178609 ],
           [-1.        , -0.31241345,  0.52909077],
           [ 1.        , -1.32316283,  0.59133674],
           [ 1.        ,  0.14701809,  0.31122986],
           [ 1.        ,  1.98474425, -0.28010688],
           [ 1.        ,  0.14701809, -0.49796778],
           [ 1.        , -1.04750391, -0.34235285],
           [ 1.        , -1.59882175, -0.52909077],
           [ 1.        , -1.23127652,  0.37347583],
           [-1.        , -0.12864083, -0.80919764],
           [-1.        , -0.40429975, -1.08930452],
           [-1.        ,  1.24965379, -1.02705855],
           [ 1.        , -0.77184498,  0.59133674],
           [ 1.        ,  0.51456332, -0.52909077],
           [-1.        , -0.77184498,  0.46684479],
           [ 1.        , -0.03675452, -1.52502632],
           [-1.        ,  0.69833594,  1.33828841],
           [-1.        , -0.77184498, -0.34235285],
           [-1.        ,  0.14701809,  0.34235285],
           [-1.        ,  1.52531271,  0.65358271],
           [ 1.        , -0.86373129, -1.21379646],
           [ 1.        ,  1.24965379,  0.52909077],
           [-1.        ,  1.98474425,  1.64951827],
           [-1.        , -0.12864083, -1.36941139],
           [-1.        , -0.03675452, -0.37347583],
           [ 1.        , -0.12864083,  1.43165737],
           [-1.        ,  2.16851686,  0.59133674],
           [ 1.        ,  0.88210855, -1.15155049],
           [-1.        , -0.58807237,  0.43572181],
           [-1.        , -0.86373129,  0.34235285],
           [ 1.        ,  1.24965379, -1.27604243],
           [ 1.        , -1.13939022, -1.52502632],
           [ 1.        , -0.31241345, -1.5872723 ],
           [ 1.        ,  2.26040317, -0.84032063],
           [ 1.        , -1.50693545,  0.2178609 ],
           [ 1.        ,  0.05513178,  0.93368959],
           [-1.        , -1.50693545, -1.33828841],
           [ 1.        ,  2.26040317,  0.43572181],
           [-1.        , -1.04750391,  0.62245972],
           [ 1.        , -0.77184498, -0.34235285],
           [ 1.        ,  0.42267702, -0.6847057 ],
           [-1.        ,  0.60644963,  0.03112299],
           [-1.        , -0.31241345,  2.52096188],
           [-1.        , -0.03675452,  0.24898389],
           [-1.        , -1.23127652, -0.18673792],
           [ 1.        ,  0.88210855, -1.46278035],
           [ 1.        , -0.77184498,  0.62245972],
           [ 1.        , -1.59882175,  0.40459882],
           [-1.        ,  0.60644963,  0.31122986],
           [-1.        ,  0.42267702, -0.28010688],
           [-1.        ,  1.61719902, -1.08930452],
           [-1.        ,  1.06588117,  1.18267348]])

```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2) # We mention in the parameters for it to take
# 5 nearest neigbors(default value) and p = 2(default value = Eucledian distance)
classifier.fit(x_train, y_train) # Feed the training data to the classifier.
```

    KNeighborsClassifier()

```python
y_pred = classifier.predict(x_test) # Predicting for x_test data
```

```python
y_pred
```

    array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
           1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
           0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1])

```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
```

```python
cm
```

    array([[53,  5],
           [ 1, 21]])

```python
ac
```

    0.925

### Now we try the model's accuracy for different values of k

```python
from sklearn import metrics
scores = {}
scores_list = []
for k in range(1, 30):
  classifier1 = KNeighborsClassifier(n_neighbors = k)
  classifier1.fit(x_train, y_train)
  y_pred1 = classifier1.predict(x_test)
  temp = metrics.accuracy_score(y_test, y_pred1)
  scores[k] = temp
  scores_list.append(temp)
```

```python
import matplotlib.pyplot as plt
k_range = range(1, 30)
plt.plot(k_range, scores_list)
plt.xlabel('Value of k')
plt.ylabel('Testing accuracy')
```

    Text(0, 0.5, 'Testing accuracy')

    
![png](output_58_1.png)
    

### Inference from the plot:

1.   The best accuracy we get is 0.95 = 95%
2.   Best k value would be around k = 7 or 8 (betwen 5 and 10) as we get the best accuracy with the least number of computations.
3.   We do however get the best peak value from k = 10 upto k = 20 as well but that has more computations than the value which we already obtained at k = 7 or 8.

# Conclusion:
## From the above experiment, I learn the following:
1.	Implement KNN technique for the classification.
2.  Make inbuilt functions and use functions/methods from built in libraries to obtain the values.
