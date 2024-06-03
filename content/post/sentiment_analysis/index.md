---
title: Sentiment Analysis
date: '2024-06-03'
---
```python
# import libraries used:
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import spacy
from gensim.models import Word2Vec
from numpy import dot, nan_to_num
from numpy.linalg import norm
```

### Task 1: Select a dataset and identify the problem statement

```python
df = pd.read_csv('/content/Apple-Twitter-Sentiment-DFE.csv', encoding='ISO-8859-1')
```

### Task 2: Perform EDA, text preprocessing, feature engineering

```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3886 entries, 0 to 3885
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   _unit_id              3886 non-null   int64  
     1   _golden               3886 non-null   bool   
     2   _unit_state           3886 non-null   object 
     3   _trusted_judgments    3886 non-null   int64  
     4   _last_judgment_at     3783 non-null   object 
     5   sentiment             3886 non-null   object 
     6   sentiment:confidence  3886 non-null   float64
     7   date                  3886 non-null   object 
     8   id                    3886 non-null   float64
     9   query                 3886 non-null   object 
     10  sentiment_gold        103 non-null    object 
     11  text                  3886 non-null   object 
    dtypes: bool(1), float64(2), int64(2), object(7)
    memory usage: 337.9+ KB

```python
df.describe()
```

  <div id="df-32850436-6608-4845-93de-1644b1359bf2" class="colab-df-container">
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
      <th>_unit_id</th>
      <th>_trusted_judgments</th>
      <th>sentiment:confidence</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.886000e+03</td>
      <td>3886.000000</td>
      <td>3886.000000</td>
      <td>3.886000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.234975e+08</td>
      <td>3.687082</td>
      <td>0.829526</td>
      <td>5.410039e+17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.171906e+03</td>
      <td>2.004595</td>
      <td>0.175864</td>
      <td>7.942752e+14</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.234955e+08</td>
      <td>3.000000</td>
      <td>0.332700</td>
      <td>5.400000e+17</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.234965e+08</td>
      <td>3.000000</td>
      <td>0.674475</td>
      <td>5.400000e+17</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.234975e+08</td>
      <td>3.000000</td>
      <td>0.811250</td>
      <td>5.410000e+17</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.234984e+08</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>5.420000e+17</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.235173e+08</td>
      <td>27.000000</td>
      <td>1.000000</td>
      <td>5.420000e+17</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-32850436-6608-4845-93de-1644b1359bf2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
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

    .colab-df-buttons div {
      margin-bottom: 4px;
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
        document.querySelector('#df-32850436-6608-4845-93de-1644b1359bf2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-32850436-6608-4845-93de-1644b1359bf2');
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

<div id="df-51933430-4165-47b6-a222-e499273f1fc5">
  <button class="colab-df-quickchart" onclick="quickchart('df-51933430-4165-47b6-a222-e499273f1fc5')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-51933430-4165-47b6-a222-e499273f1fc5 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>

```python
df.head()
```

  <div id="df-3668bf32-2cb2-4df9-9707-6cc01f8683ee" class="colab-df-container">
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
      <th>_unit_id</th>
      <th>_golden</th>
      <th>_unit_state</th>
      <th>_trusted_judgments</th>
      <th>_last_judgment_at</th>
      <th>sentiment</th>
      <th>sentiment:confidence</th>
      <th>date</th>
      <th>id</th>
      <th>query</th>
      <th>sentiment_gold</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>623495513</td>
      <td>True</td>
      <td>golden</td>
      <td>10</td>
      <td>NaN</td>
      <td>3</td>
      <td>0.6264</td>
      <td>Mon Dec 01 19:30:03 +0000 2014</td>
      <td>5.400000e+17</td>
      <td>#AAPL OR @Apple</td>
      <td>3\nnot_relevant</td>
      <td>#AAPL:The 10 best Steve Jobs emails ever...htt...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>623495514</td>
      <td>True</td>
      <td>golden</td>
      <td>12</td>
      <td>NaN</td>
      <td>3</td>
      <td>0.8129</td>
      <td>Mon Dec 01 19:43:51 +0000 2014</td>
      <td>5.400000e+17</td>
      <td>#AAPL OR @Apple</td>
      <td>3\n1</td>
      <td>RT @JPDesloges: Why AAPL Stock Had a Mini-Flas...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>623495515</td>
      <td>True</td>
      <td>golden</td>
      <td>10</td>
      <td>NaN</td>
      <td>3</td>
      <td>1.0000</td>
      <td>Mon Dec 01 19:50:28 +0000 2014</td>
      <td>5.400000e+17</td>
      <td>#AAPL OR @Apple</td>
      <td>3</td>
      <td>My cat only chews @apple cords. Such an #Apple...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>623495516</td>
      <td>True</td>
      <td>golden</td>
      <td>17</td>
      <td>NaN</td>
      <td>3</td>
      <td>0.5848</td>
      <td>Mon Dec 01 20:26:34 +0000 2014</td>
      <td>5.400000e+17</td>
      <td>#AAPL OR @Apple</td>
      <td>3\n1</td>
      <td>I agree with @jimcramer that the #IndividualIn...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>623495517</td>
      <td>False</td>
      <td>finalized</td>
      <td>3</td>
      <td>12/12/14 12:14</td>
      <td>3</td>
      <td>0.6474</td>
      <td>Mon Dec 01 20:29:33 +0000 2014</td>
      <td>5.400000e+17</td>
      <td>#AAPL OR @Apple</td>
      <td>NaN</td>
      <td>Nobody expects the Spanish Inquisition #AAPL</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3668bf32-2cb2-4df9-9707-6cc01f8683ee')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
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

    .colab-df-buttons div {
      margin-bottom: 4px;
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
        document.querySelector('#df-3668bf32-2cb2-4df9-9707-6cc01f8683ee button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3668bf32-2cb2-4df9-9707-6cc01f8683ee');
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

<div id="df-da0c7af8-58f3-48e1-867e-68321e2e1580">
  <button class="colab-df-quickchart" onclick="quickchart('df-da0c7af8-58f3-48e1-867e-68321e2e1580')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-da0c7af8-58f3-48e1-867e-68321e2e1580 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>

```python
df['sentiment'].value_counts()
```

    3               2162
    1               1219
    5                423
    not_relevant      82
    Name: sentiment, dtype: int64

```python
def preprocess_text(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word.isalnum()])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
```

```python
df['text_cleaned'] = df['text'].apply(preprocess_text)
```

```python
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df['text_cleaned'])
y = df['sentiment']
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
```

<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(max_iter=1000)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=1000)</pre></div></div></div></div></div>

```python
y_pred = classifier.predict(X_test)
```

```python
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
```

    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))

```python
confusion
```

    array([[140,  97,   3,   0],
           [ 26, 397,   1,   0],
           [ 12,  67,  20,   0],
           [  1,  14,   0,   0]])

```python
accuracy
```

    0.7159383033419023

```python
report
```

    '              precision    recall  f1-score   support\n\n           1       0.78      0.58      0.67       240\n           3       0.69      0.94      0.79       424\n           5       0.83      0.20      0.33        99\nnot_relevant       0.00      0.00      0.00        15\n\n    accuracy                           0.72       778\n   macro avg       0.58      0.43      0.45       778\nweighted avg       0.72      0.72      0.68       778\n'

### Task 3: Implement sentiment analysis on the given dataset in Natural Language Processing

```python
nlp = spacy.load("en_core_web_sm")
```

```python
def preprocess(text):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_punct and not token.is_space]

tokenized_tweets = [preprocess(tweet) for tweet in df['text']]
```

```python
model = Word2Vec(tokenized_tweets, vector_size=100, window=5, min_count=1, sg=0)
document_embeddings = []

for tokenized_tweet in tokenized_tweets:

    valid_tokens = [token for token in tokenized_tweet if token in model.wv]

    if valid_tokens:

        avg_embedding = sum(model.wv[token] for token in valid_tokens) / len(valid_tokens)
        document_embeddings.append(avg_embedding)
    else:

        document_embeddings.append(None)
```

### Task 4: Analyze and comprehend the results obtained

```python
tweet1_embedding = document_embeddings[0]
tweet2_embedding = document_embeddings[1]
```

```python
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))
```

```python
if tweet1_embedding is not None and tweet2_embedding is not None:
    similarity = cosine_similarity(tweet1_embedding, tweet2_embedding)
else:
    similarity = None
print(f"Similarity between tweet 1 and tweet 2: {similarity}")
```

    Similarity between tweet 1 and tweet 2: 0.9979607462882996

### Conclusion:
Based on the experiment conducted, we can conclude that the sentiment analysis performed on the given dataset in Natural Language Processing was successful. The dataset was selected and the problem statement was identified. EDA, text preprocessing, and feature engineering were performed to prepare the dataset for sentiment analysis. The results obtained were analyzed and comprehended.
