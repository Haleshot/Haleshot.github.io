---
title: Feature Engineering
date: '2024-06-03'
---
```python
# import libraries used
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import nltk
from nltk import sent_tokenize, word_tokenize, RegexpTokenizer, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
```

# Tasks 

### a. To implement label encoding and one hot encoding on textual data

#### Performing preprocessing operations such as tokenization, punctuation removal and stop word removal before operating on the data

```python
sentence = "This is a demo text used for testing the various built in methods of nltk library"

# Case uniformity
sentence = sentence.lower()

# Tokenization and Stopword removal
stopword = stopwords.words('english')
word_tokens = nltk.word_tokenize(sentence)
removing_stopwords = [word for word in word_tokens if word not in stopword]
print(removing_stopwords)
```

    ['demo', 'text', 'used', 'testing', 'various', 'built', 'methods', 'nltk', 'library']

#### Label Encoding:

```python
# Creating an initial dataframe
dog_types = ("affenpinscher", 
"Afghan hound", 
"Airedale terrier", 
"Akita", 
"Alaskan Malamute", 
"American Staffordshire terrier", 
"American water spaniel", 
"Australian cattle dog", 
"Australian shepherd", 
"Australian terrier", 
"basenji", 
"basset hound", 
"beagle", 
"bearded collie", 
"Bedlington terrier", 
"Bernese mountain dog", 
"bichon frise", 
"black and tan coonhound", 
"bloodhound", 
"border collie", 
"border terrier", 
"borzoi", 
"Boston terrier", 
"bouvier des Flandres", 
"boxer", 
"briard", 
"Brittany", 
"Brussels griffon", 
"bull terrier", 
"bulldog", 
"bullmastiff", 
"cairn terrier", 
"Canaan dog", 
"Chesapeake Bay retriever", 
"Chihuahua", 
"Chinese crested", 
"Chinese shar-pei", 
"chow chow", 
"Clumber spaniel", 
"cocker spaniel", 
"collie", 
"curly-coated retriever", 
"dachshund", 
"Dalmatian", 
"Doberman pinscher", 
"English cocker spaniel", 
"English setter", 
"English springer spaniel", 
"English toy spaniel", 
"Eskimo dog", 
"Finnish spitz", 
"flat-coated retriever", 
"fox terrier", 
"foxhound", 
"French bulldog", 
"German shepherd", 
"German shorthaired pointer", 
"German wirehaired pointer", 
"golden retriever", 
"Gordon setter", 
"Great Dane", 
"greyhound", 
"Irish setter", 
"Irish water spaniel", 
"Irish wolfhound", 
"Jack Russell terrier", 
"Japanese spaniel", 
"keeshond", 
"Kerry blue terrier", 
"komondor", 
"kuvasz", 
"Labrador retriever", 
"Lakeland terrier", 
"Lhasa apso", 
"Maltese", 
"Manchester terrier", 
"mastiff", 
"Mexican hairless", 
"Newfoundland", 
"Norwegian elkhound", 
"Norwich terrier", 
"otterhound", 
"papillon", 
"Pekingese", 
"pointer", 
"Pomeranian", 
"poodle", 
"pug", 
"puli", 
"Rhodesian ridgeback", 
"Rottweiler", 
"Saint Bernard", 
"saluki", 
"Samoyed", 
"schipperke", 
"schnauzer", 
"Scottish deerhound", 
"Scottish terrier", 
"Sealyham terrier", 
"Shetland sheepdog", 
"shih tzu", 
"Siberian husky", 
"silky terrier", 
"Skye terrier", 
"Staffordshire bull terrier", 
"soft-coated wheaten terrier", 
"Sussex spaniel", 
"spitz", 
"Tibetan terrier", 
"vizsla", 
"Weimaraner", 
"Welsh terrier", 
"West Highland white terrier", 
"whippet", 
"Yorkshire terrier")

dogs_df = pd.DataFrame(dog_types, columns = ['Dog_Types'])

# Creating instance of labelencoder
labelencoder = LabelEncoder()
dogs_df['Dog_Types_Categories'] = labelencoder.fit_transform(dogs_df['Dog_Types'])
```

```python
dogs_df
```

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
      <th>Dog_Types</th>
      <th>Dog_Types_Categories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>affenpinscher</td>
      <td>68</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghan hound</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Airedale terrier</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Akita</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alaskan Malamute</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Weimaraner</td>
      <td>64</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Welsh terrier</td>
      <td>65</td>
    </tr>
    <tr>
      <th>112</th>
      <td>West Highland white terrier</td>
      <td>66</td>
    </tr>
    <tr>
      <th>113</th>
      <td>whippet</td>
      <td>114</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Yorkshire terrier</td>
      <td>67</td>
    </tr>
  </tbody>
</table>
<p>115 rows × 2 columns</p>
</div>

```python
dogs_df['Dog_Types_Categories']
```

    0       68
    1        0
    2        1
    3        2
    4        3
          ... 
    110     64
    111     65
    112     66
    113    114
    114     67
    Name: Dog_Types_Categories, Length: 115, dtype: int32

#### One Hot Encoding:

#### Using sci-kit learn library approach:

```python
# Creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore') # ‘ignore’ : When an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros. In the inverse transform, an unknown category will be denoted as None.

# passing Dog_Types_Categories column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(dogs_df[['Dog_Types_Categories']]).toarray())

# Merge with main df bridge_df on key values
dogs_df = dogs_df.join(enc_df)
```

```python
dogs_df
```

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
      <th>Dog_Types</th>
      <th>Dog_Types_Categories</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>...</th>
      <th>105</th>
      <th>106</th>
      <th>107</th>
      <th>108</th>
      <th>109</th>
      <th>110</th>
      <th>111</th>
      <th>112</th>
      <th>113</th>
      <th>114</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>affenpinscher</td>
      <td>68</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghan hound</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Airedale terrier</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Akita</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alaskan Malamute</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Weimaraner</td>
      <td>64</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Welsh terrier</td>
      <td>65</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>112</th>
      <td>West Highland white terrier</td>
      <td>66</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>whippet</td>
      <td>114</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Yorkshire terrier</td>
      <td>67</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>115 rows × 117 columns</p>
</div>

#### Get Dummies method:

```python
# Creating an initial dataframe
dog_types = ("affenpinscher", 
"Afghan hound", 
"Airedale terrier", 
"Akita", 
"Alaskan Malamute", 
"American Staffordshire terrier", 
"American water spaniel", 
"Australian cattle dog", 
"Australian shepherd", 
"Australian terrier", 
"basenji", 
"basset hound", 
"beagle", 
"bearded collie", 
"Bedlington terrier", 
"Bernese mountain dog", 
"bichon frise", 
"black and tan coonhound", 
"bloodhound", 
"border collie", 
"border terrier", 
"borzoi", 
"Boston terrier", 
"bouvier des Flandres", 
"boxer", 
"briard", 
"Brittany", 
"Brussels griffon", 
"bull terrier", 
"bulldog", 
"bullmastiff", 
"cairn terrier", 
"Canaan dog", 
"Chesapeake Bay retriever", 
"Chihuahua", 
"Chinese crested", 
"Chinese shar-pei", 
"chow chow", 
"Clumber spaniel", 
"cocker spaniel", 
"collie", 
"curly-coated retriever", 
"dachshund", 
"Dalmatian", 
"Doberman pinscher", 
"English cocker spaniel", 
"English setter", 
"English springer spaniel", 
"English toy spaniel", 
"Eskimo dog", 
"Finnish spitz", 
"flat-coated retriever", 
"fox terrier", 
"foxhound", 
"French bulldog", 
"German shepherd", 
"German shorthaired pointer", 
"German wirehaired pointer", 
"golden retriever", 
"Gordon setter", 
"Great Dane", 
"greyhound", 
"Irish setter", 
"Irish water spaniel", 
"Irish wolfhound", 
"Jack Russell terrier", 
"Japanese spaniel", 
"keeshond", 
"Kerry blue terrier", 
"komondor", 
"kuvasz", 
"Labrador retriever", 
"Lakeland terrier", 
"Lhasa apso", 
"Maltese", 
"Manchester terrier", 
"mastiff", 
"Mexican hairless", 
"Newfoundland", 
"Norwegian elkhound", 
"Norwich terrier", 
"otterhound", 
"papillon", 
"Pekingese", 
"pointer", 
"Pomeranian", 
"poodle", 
"pug", 
"puli", 
"Rhodesian ridgeback", 
"Rottweiler", 
"Saint Bernard", 
"saluki", 
"Samoyed", 
"schipperke", 
"schnauzer", 
"Scottish deerhound", 
"Scottish terrier", 
"Sealyham terrier", 
"Shetland sheepdog", 
"shih tzu", 
"Siberian husky", 
"silky terrier", 
"Skye terrier", 
"Staffordshire bull terrier", 
"soft-coated wheaten terrier", 
"Sussex spaniel", 
"spitz", 
"Tibetan terrier", 
"vizsla", 
"Weimaraner", 
"Welsh terrier", 
"West Highland white terrier", 
"whippet", 
"Yorkshire terrier")

dogs_df = pd.DataFrame(dog_types, columns = ['Dog_Types'])

dum_df = pd.get_dummies(dogs_df, columns=["Dog_Types"], prefix=["Type_is"] )

# Merge with main df bridge_df on key values
dogs_df = dogs_df.join(dum_df)
```

```python
dogs_df
```

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
      <th>Dog_Types</th>
      <th>Type_is_Afghan hound</th>
      <th>Type_is_Airedale terrier</th>
      <th>Type_is_Akita</th>
      <th>Type_is_Alaskan Malamute</th>
      <th>Type_is_American Staffordshire terrier</th>
      <th>Type_is_American water spaniel</th>
      <th>Type_is_Australian cattle dog</th>
      <th>Type_is_Australian shepherd</th>
      <th>Type_is_Australian terrier</th>
      <th>...</th>
      <th>Type_is_puli</th>
      <th>Type_is_saluki</th>
      <th>Type_is_schipperke</th>
      <th>Type_is_schnauzer</th>
      <th>Type_is_shih tzu</th>
      <th>Type_is_silky terrier</th>
      <th>Type_is_soft-coated wheaten terrier</th>
      <th>Type_is_spitz</th>
      <th>Type_is_vizsla</th>
      <th>Type_is_whippet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>affenpinscher</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghan hound</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Airedale terrier</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Akita</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alaskan Malamute</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Weimaraner</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Welsh terrier</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>112</th>
      <td>West Highland white terrier</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>113</th>
      <td>whippet</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Yorkshire terrier</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>115 rows × 116 columns</p>
</div>

### b.	To implement Bag of Words (BoW) feature engineering technique on textual data

#### Using user defined function after preprocessing:

```python
doc1 = 'Game of Thrones is an amazing tv series!'
doc2 = 'Game of Thrones is the best tv series!'
doc3 = 'Game of Thrones is so great'

l_doc1 = re.sub(r"[^a-zA-Z0-9]", " ", doc1.lower()).split()
l_doc2 = re.sub(r"[^a-zA-Z0-9]", " ", doc2.lower()).split()
l_doc3 = re.sub(r"[^a-zA-Z0-9]", " ", doc3.lower()).split()

wordset12 = np.union1d(l_doc1,l_doc2)
wordset =  np.union1d(wordset12,l_doc3)
print(wordset)
```

    ['amazing' 'an' 'best' 'game' 'great' 'is' 'of' 'series' 'so' 'the'
     'thrones' 'tv']

```python
def calculateBOW(wordset,l_doc):
  tf_diz = dict.fromkeys(wordset,0)
  for word in l_doc:
      tf_diz[word]=l_doc.count(word)
  return tf_diz

bow1 = calculateBOW(wordset,l_doc1)
bow2 = calculateBOW(wordset,l_doc2)
bow3 = calculateBOW(wordset,l_doc3)
df_bow = pd.DataFrame([bow1,bow2,bow3])
df_bow.head()
```

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
      <th>amazing</th>
      <th>an</th>
      <th>best</th>
      <th>game</th>
      <th>great</th>
      <th>is</th>
      <th>of</th>
      <th>series</th>
      <th>so</th>
      <th>the</th>
      <th>thrones</th>
      <th>tv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

#### Using sci-kit learn library:

```python
doc1 = 'Game of Thrones is an amazing tv series!'
doc2 = 'Game of Thrones is the best tv series!'
doc3 = 'Game of Thrones is so great'

CountVec = CountVectorizer(ngram_range=(1,1), # to use bigrams ngram_range=(2,2)
                           stop_words='english')

# Transform
Count_data = CountVec.fit_transform([doc1, doc2, doc3])
 
# Initializing the dataframe
cv_dataframe = pd.DataFrame(Count_data.toarray(), columns=CountVec.get_feature_names_out())
cv_dataframe.head()
```

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
      <th>amazing</th>
      <th>best</th>
      <th>game</th>
      <th>great</th>
      <th>series</th>
      <th>thrones</th>
      <th>tv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

### c.	To implement TF-IDF feature engineering technique

```python
documents = ["Inflation has increased unemployment", 
             "The company has increased its sales", 
              "Fear increased his pulse"]
# Preprocessing
def return_preprocessed_document(document):
    document = document.lower()
    
    # Word Tokenization
    words = word_tokenize(document)
    
    # Stop Words removal
    words = [word for word in words if word not in stopwords.words("english")]

    # Forming the complete sentence using String join
    document = " ".join(words)
    return document
documents = [return_preprocessed_document(document) for document in documents]
```

```python
documents
```

    ['inflation increased unemployment',
     'company increased sales',
     'fear increased pulse']

```python
# Creation of a TF-IDF model using Tfidf vectorizer function.

vectorizer = TfidfVectorizer()
tfidf_model = vectorizer.fit_transform(documents)
print(tfidf_model)
```

      (0, 6)	0.652490884512534
      (0, 2)	0.3853716274664007
      (0, 3)	0.652490884512534
      (1, 5)	0.652490884512534
      (1, 0)	0.652490884512534
      (1, 2)	0.3853716274664007
      (2, 4)	0.652490884512534
      (2, 1)	0.652490884512534
      (2, 2)	0.3853716274664007

```python
pd.DataFrame(tfidf_model.toarray(), columns = vectorizer.get_feature_names_out())
```

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
      <th>company</th>
      <th>fear</th>
      <th>increased</th>
      <th>inflation</th>
      <th>pulse</th>
      <th>sales</th>
      <th>unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.385372</td>
      <td>0.652491</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.652491</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.652491</td>
      <td>0.000000</td>
      <td>0.385372</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.652491</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.652491</td>
      <td>0.385372</td>
      <td>0.000000</td>
      <td>0.652491</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>

#### From the above output, we can infer the following:
##### We created our TF-IDF model where the sample sentences are converted into matrix format with higher weights assigned to semantically important words in a document such as inflation and unemployment in:
#### sentence 1, company and sales in sentence 2, and fear and pulse in sentence. 3. While frequent word across all documents, increased, assigned with lower weights, i.e., 0.385372

#### Additional task of classifying documents:

```python
file = open("textfile.txt", 'r', encoding="mbcs")
d = {}
for i in file.read().split():
    print(i)
file.close()
```

    Medical:
    Hospital
    Emergency
    Room
    (ER)
    Intensive
    Care
    Unit
    (ICU)
    Operating
    Room
    (OR)
    Exam
    Diagnosis
    Prescription
    Urine
    sample
    Blood
    sample
    Hypertension
    Cast
    Vein
    Syringe
    Painkiller/pain
    reliever
    Numb
    Dosage
    Biopsy
    (of
    abnormal
    cells)
    Finanace:
    1.
    Amortization:
    Amortization
    is
    a
    method
    of
    spreading
    an
    intangible
    asset's
    cost
    over
    the
    course
    of
    its
    useful
    life.
    Intangible
    assets
    are
    non-physical
    assets
    that
    are
    essential
    to
    a
    company,
    such
    as
    a
    trademark,
    patent,
    copyright,
    or
    franchise
    agreement.
    2.
    Assets:
    Assets
    are
    items
    you
    own
    that
    can
    provide
    future
    benefit
    to
    your
    business,
    such
    as
    cash,
    inventory,
    real
    estate,
    office
    equipment,
    or
    accounts
    receivable,
    which
    are
    payments
    due
    to
    a
    company
    by
    its
    customers.
    There
    are
    different
    types
    of
    assets,
    including:
    Current
    Assets:
    Which
    can
    be
    converted
    to
    cash
    within
    a
    year
    Fixed
    Assets:
    Which
    canâ€™t
    immediately
    be
    turned
    into
    cash,
    but
    are
    tangible
    items
    that
    a
    company
    owns
    and
    uses
    to
    generate
    long-term
    income
    3.
    Asset
    Allocation:
    Asset
    allocation
    refers
    to
    how
    you
    choose
    to
    spread
    your
    money
    across
    different
    investment
    types,
    also
    known
    as
    asset
    classes.
    These
    include:
    Bonds:
    Bonds
    represent
    a
    form
    of
    borrowing.
    When
    you
    buy
    a
    bond,
    typically
    from
    the
    government
    or
    a
    corporation,
    youâ€™re
    essentially
    lending
    them
    money.
    You
    receive
    periodic
    interest
    payments
    and
    get
    back
    the
    loaned
    amount
    at
    the
    time
    of
    the
    bondâ€™s
    maturityâ€”or
    the
    defined
    term
    at
    which
    the
    bond
    can
    be
    redeemed.
    Stocks:
    A
    stock
    is
    a
    share
    of
    ownership
    in
    a
    public
    or
    private
    company.
    When
    you
    buy
    stock
    in
    a
    company,
    you
    become
    a
    shareholder
    and
    can
    receive
    dividendsâ€”the
    companyâ€™s
    profitsâ€”if
    and
    when
    they
    are
    distributed.
    Cash
    and
    Cash
    Equivalents:
    This
    refers
    to
    any
    asset
    in
    the
    form
    of
    cash,
    or
    which
    can
    be
    converted
    to
    cash
    easily
    in
    the
    event
    it's
    necessary.
    4.
    Balance
    Sheet:
    A
    balance
    sheet
    is
    an
    important
    financial
    statement
    that
    communicates
    an
    organizationâ€™s
    worth,
    or
    â€œbook
    value.â€
    The
    balance
    sheet
    includes
    a
    tally
    of
    the
    organizationâ€™s
    assets,
    liabilities,
    and
    shareholdersâ€™
    equity
    for
    a
    given
    reporting
    period.
    The
    Balance
    Sheet
    Equation:
    Balance
    sheets
    are
    arranged
    according
    to
    the
    following
    equation:
    Assets
    =
    Liabilities
    +
    Ownersâ€™
    Equity
    5.
    Capital
    Gain:
    A
    capital
    gain
    is
    an
    increase
    in
    the
    value
    of
    an
    asset
    or
    investment
    above
    the
    price
    you
    initially
    paid
    for
    it.
    If
    you
    sell
    the
    asset
    for
    less
    than
    the
    original
    purchase
    price,
    that
    would
    be
    considered
    a
    capital
    loss.
    Related:
    6
    Ways
    Understanding
    Finance
    Can
    Help
    You
    Excel
    Professionally
    6.
    Capital
    Market:
    This
    is
    a
    market
    where
    buyers
    and
    sellers
    engage
    in
    the
    trade
    of
    financial
    assets,
    including
    stocks
    and
    bonds.
    Capital
    markets
    feature
    several
    participants,
    including:
    Companies:
    Firms
    that
    sell
    stocks
    and
    bonds
    to
    investors
    Institutional
    investors:
    Investors
    who
    purchase
    stocks
    and
    bonds
    on
    behalf
    of
    a
    large
    capital
    base
    Mutual
    funds:
    A
    mutual
    fund
    is
    an
    institutional
    investor
    that
    manages
    the
    investments
    of
    thousands
    of
    individuals
    Hedge
    funds:
    A
    hedge
    fund
    is
    another
    type
    of
    institutional
    investor,
    which
    controls
    risk
    through
    hedgingâ€”a
    process
    of
    buying
    one
    stock
    and
    then
    shorting
    a
    similar
    stock
    to
    make
    money
    from
    the
    difference
    in
    their
    relative
    performance
    7.
    Cash
    Flow:
    Cash
    flow
    refers
    to
    the
    net
    balance
    of
    cash
    moving
    in
    and
    out
    of
    a
    business
    at
    a
    specific
    point
    in
    time.
    Cash
    flow
    is
    commonly
    broken
    into
    three
    categories,
    including:
    Operating
    Cash
    Flow:
    The
    net
    cash
    generated
    from
    normal
    business
    operations
    Investing
    Cash
    Flow:
    The
    net
    cash
    generated
    from
    investing
    activities,
    such
    as
    securities
    investments
    and
    the
    purchase
    or
    sale
    of
    assets
    Financing
    Cash
    Flow:
    The
    net
    cash
    generated
    financing
    a
    business,
    including
    debt
    payments,
    shareholdersâ€™
    equity,
    and
    dividend
    payments
    8.
    Cash
    Flow
    Statement:
    A
    cash
    flow
    statement
    is
    a
    financial
    statement
    prepared
    to
    provide
    a
    detailed
    analysis
    of
    what
    happened
    to
    a
    companyâ€™s
    cash
    during
    a
    given
    period
    of
    time.
    This
    document
    shows
    how
    the
    business
    generated
    and
    spent
    its
    cash
    by
    including
    an
    overview
    of
    cash
    flows
    from
    operating,
    investing,
    and
    financing
    activities
    during
    the
    reporting
    period.
    9.
    Compound
    Interest:
    This
    refers
    to
    â€œinterest
    on
    interest.â€
    Rather,
    when
    youâ€™re
    investing
    or
    saving,
    compound
    interest
    is
    earned
    on
    the
    amount
    you
    deposited,
    plus
    any
    interest
    youâ€™ve
    accumulated
    over
    time.
    While
    it
    can
    grow
    your
    savings,
    it
    can
    also
    increase
    your
    debt;
    compound
    interest
    is
    charged
    on
    the
    initial
    amount
    you
    were
    loaned,
    as
    well
    as
    the
    expenses
    added
    to
    your
    outstanding
    balance
    over
    time.
    10.
    Depreciation:
    Depreciation
    represents
    the
    decrease
    in
    an
    assetâ€™s
    value.
    Itâ€™s
    a
    term
    commonly
    used
    in
    accounting
    and
    shows
    how
    much
    of
    an
    assetâ€™s
    value
    a
    business
    has
    used
    over
    a
    period
    of
    time.
    Equity

```python
# We take two documents containing text about medical and finance related words.
# We need to read the text file and classify which field the document belongs to.
# This can be done by counting the words relating to finance and medical and seeing which count is larger (simplest method).

file = open("textfile.txt", 'r', encoding="mbcs")
d = {}
for i in file.read().split():
    if i in d:
        d[i] += 1
    else:
        d[i] = 1
print(d)
file.close()
```

    {'Medical:': 1, 'Hospital': 1, 'Emergency': 1, 'Room': 2, '(ER)': 1, 'Intensive': 1, 'Care': 1, 'Unit': 1, '(ICU)': 1, 'Operating': 2, '(OR)': 1, 'Exam': 1, 'Diagnosis': 1, 'Prescription': 1, 'Urine': 1, 'sample': 2, 'Blood': 1, 'Hypertension': 1, 'Cast': 1, 'Vein': 1, 'Syringe': 1, 'Painkiller/pain': 1, 'reliever': 1, 'Numb': 1, 'Dosage': 1, 'Biopsy': 1, '(of': 1, 'abnormal': 1, 'cells)': 1, 'Finanace:': 1, '1.': 1, 'Amortization:': 1, 'Amortization': 1, 'is': 11, 'a': 29, 'method': 1, 'of': 23, 'spreading': 1, 'an': 9, 'intangible': 1, "asset's": 1, 'cost': 1, 'over': 4, 'the': 26, 'course': 1, 'its': 3, 'useful': 1, 'life.': 1, 'Intangible': 1, 'assets': 3, 'are': 8, 'non-physical': 1, 'that': 7, 'essential': 1, 'to': 17, 'company,': 2, 'such': 3, 'as': 6, 'trademark,': 1, 'patent,': 1, 'copyright,': 1, 'or': 9, 'franchise': 1, 'agreement.': 1, '2.': 1, 'Assets:': 3, 'Assets': 2, 'items': 2, 'you': 9, 'own': 1, 'can': 7, 'provide': 2, 'future': 1, 'benefit': 1, 'your': 5, 'business,': 2, 'cash,': 3, 'inventory,': 1, 'real': 1, 'estate,': 1, 'office': 1, 'equipment,': 1, 'accounts': 1, 'receivable,': 1, 'which': 4, 'payments': 3, 'due': 1, 'company': 2, 'by': 2, 'customers.': 1, 'There': 1, 'different': 2, 'types': 1, 'assets,': 3, 'including:': 3, 'Current': 1, 'Which': 2, 'be': 5, 'converted': 2, 'cash': 10, 'within': 1, 'year': 1, 'Fixed': 1, 'canâ€™t': 1, 'immediately': 1, 'turned': 1, 'into': 2, 'but': 1, 'tangible': 1, 'owns': 1, 'and': 17, 'uses': 1, 'generate': 1, 'long-term': 1, 'income': 1, '3.': 1, 'Asset': 2, 'Allocation:': 1, 'allocation': 1, 'refers': 4, 'how': 3, 'choose': 1, 'spread': 1, 'money': 2, 'across': 1, 'investment': 2, 'types,': 1, 'also': 2, 'known': 1, 'asset': 4, 'classes.': 1, 'These': 1, 'include:': 1, 'Bonds:': 1, 'Bonds': 1, 'represent': 1, 'form': 2, 'borrowing.': 1, 'When': 2, 'buy': 2, 'bond,': 1, 'typically': 1, 'from': 5, 'government': 1, 'corporation,': 1, 'youâ€™re': 2, 'essentially': 1, 'lending': 1, 'them': 1, 'money.': 1, 'You': 2, 'receive': 2, 'periodic': 1, 'interest': 4, 'get': 1, 'back': 1, 'loaned': 1, 'amount': 3, 'at': 3, 'time': 1, 'bondâ€™s': 1, 'maturityâ€”or': 1, 'defined': 1, 'term': 2, 'bond': 1, 'redeemed.': 1, 'Stocks:': 1, 'A': 6, 'stock': 4, 'share': 1, 'ownership': 1, 'in': 11, 'public': 1, 'private': 1, 'company.': 1, 'become': 1, 'shareholder': 1, 'dividendsâ€”the': 1, 'companyâ€™s': 2, 'profitsâ€”if': 1, 'when': 2, 'they': 1, 'distributed.': 1, 'Cash': 9, 'Equivalents:': 1, 'This': 4, 'any': 2, 'easily': 1, 'event': 1, "it's": 1, 'necessary.': 1, '4.': 1, 'Balance': 3, 'Sheet:': 1, 'balance': 4, 'sheet': 2, 'important': 1, 'financial': 3, 'statement': 3, 'communicates': 1, 'organizationâ€™s': 2, 'worth,': 1, 'â€œbook': 1, 'value.â€\x9d': 1, 'The': 5, 'includes': 1, 'tally': 1, 'liabilities,': 1, 'shareholdersâ€™': 2, 'equity': 1, 'for': 3, 'given': 2, 'reporting': 2, 'period.': 2, 'Sheet': 1, 'Equation:': 1, 'sheets': 1, 'arranged': 1, 'according': 1, 'following': 1, 'equation:': 1, '=': 1, 'Liabilities': 1, '+': 1, 'Ownersâ€™': 1, 'Equity': 2, '5.': 1, 'Capital': 3, 'Gain:': 1, 'capital': 3, 'gain': 1, 'increase': 2, 'value': 2, 'above': 1, 'price': 1, 'initially': 1, 'paid': 1, 'it.': 1, 'If': 1, 'sell': 2, 'less': 1, 'than': 1, 'original': 1, 'purchase': 3, 'price,': 1, 'would': 1, 'considered': 1, 'loss.': 1, 'Related:': 1, '6': 1, 'Ways': 1, 'Understanding': 1, 'Finance': 1, 'Can': 1, 'Help': 1, 'Excel': 1, 'Professionally': 1, '6.': 1, 'Market:': 1, 'market': 1, 'where': 1, 'buyers': 1, 'sellers': 1, 'engage': 1, 'trade': 1, 'including': 3, 'stocks': 3, 'bonds.': 1, 'markets': 1, 'feature': 1, 'several': 1, 'participants,': 1, 'Companies:': 1, 'Firms': 1, 'bonds': 2, 'investors': 1, 'Institutional': 1, 'investors:': 1, 'Investors': 1, 'who': 1, 'on': 4, 'behalf': 1, 'large': 1, 'base': 1, 'Mutual': 1, 'funds:': 2, 'mutual': 1, 'fund': 2, 'institutional': 2, 'investor': 1, 'manages': 1, 'investments': 2, 'thousands': 1, 'individuals': 1, 'Hedge': 1, 'hedge': 1, 'another': 1, 'type': 1, 'investor,': 1, 'controls': 1, 'risk': 1, 'through': 1, 'hedgingâ€”a': 1, 'process': 1, 'buying': 1, 'one': 1, 'then': 1, 'shorting': 1, 'similar': 1, 'make': 1, 'difference': 1, 'their': 1, 'relative': 1, 'performance': 1, '7.': 1, 'Flow:': 4, 'flow': 3, 'net': 4, 'moving': 1, 'out': 1, 'business': 4, 'specific': 1, 'point': 1, 'time.': 5, 'commonly': 2, 'broken': 1, 'three': 1, 'categories,': 1, 'generated': 4, 'normal': 1, 'operations': 1, 'Investing': 1, 'investing': 2, 'activities,': 1, 'securities': 1, 'sale': 1, 'Financing': 1, 'financing': 2, 'debt': 1, 'payments,': 1, 'equity,': 1, 'dividend': 1, '8.': 1, 'Flow': 1, 'Statement:': 1, 'prepared': 1, 'detailed': 1, 'analysis': 1, 'what': 1, 'happened': 1, 'during': 2, 'period': 2, 'document': 1, 'shows': 2, 'spent': 1, 'overview': 1, 'flows': 1, 'operating,': 1, 'investing,': 1, 'activities': 1, '9.': 1, 'Compound': 1, 'Interest:': 1, 'â€œinterest': 1, 'interest.â€\x9d': 1, 'Rather,': 1, 'saving,': 1, 'compound': 2, 'earned': 1, 'deposited,': 1, 'plus': 1, 'youâ€™ve': 1, 'accumulated': 1, 'While': 1, 'it': 2, 'grow': 1, 'savings,': 1, 'debt;': 1, 'charged': 1, 'initial': 1, 'were': 1, 'loaned,': 1, 'well': 1, 'expenses': 1, 'added': 1, 'outstanding': 1, '10.': 1, 'Depreciation:': 1, 'Depreciation': 1, 'represents': 1, 'decrease': 1, 'assetâ€™s': 2, 'value.': 1, 'Itâ€™s': 1, 'used': 2, 'accounting': 1, 'much': 1, 'has': 1}

```python
medical_words = ["Medical", "Prescription", "hospital", "health", "exam", "Blood"]
finance_words = ["Invest", "market", "payment", "Withdraw", "Cash", "Depriciation", "Equity"]

med_d = {}
finan_d = {}

file = open("textfile.txt", 'r', encoding="mbcs")

for i in file.read().split():
    if i.lower() in med_d and i in medical_words:
            med_d[i] += 1
    else:
        med_d[i] = 1
print(med_d)
file.close()

print()
print()

file = open("textfile.txt", 'r', encoding="mbcs")
for i in file.read().split():
    if i in finan_d and i in finance_words:
        finan_d[i] += 1
    else:
        finan_d[i] = 1
print(finan_d)
```

    {'Medical:': 1, 'Hospital': 1, 'Emergency': 1, 'Room': 1, '(ER)': 1, 'Intensive': 1, 'Care': 1, 'Unit': 1, '(ICU)': 1, 'Operating': 1, '(OR)': 1, 'Exam': 1, 'Diagnosis': 1, 'Prescription': 1, 'Urine': 1, 'sample': 1, 'Blood': 1, 'Hypertension': 1, 'Cast': 1, 'Vein': 1, 'Syringe': 1, 'Painkiller/pain': 1, 'reliever': 1, 'Numb': 1, 'Dosage': 1, 'Biopsy': 1, '(of': 1, 'abnormal': 1, 'cells)': 1, 'Finanace:': 1, '1.': 1, 'Amortization:': 1, 'Amortization': 1, 'is': 1, 'a': 1, 'method': 1, 'of': 1, 'spreading': 1, 'an': 1, 'intangible': 1, "asset's": 1, 'cost': 1, 'over': 1, 'the': 1, 'course': 1, 'its': 1, 'useful': 1, 'life.': 1, 'Intangible': 1, 'assets': 1, 'are': 1, 'non-physical': 1, 'that': 1, 'essential': 1, 'to': 1, 'company,': 1, 'such': 1, 'as': 1, 'trademark,': 1, 'patent,': 1, 'copyright,': 1, 'or': 1, 'franchise': 1, 'agreement.': 1, '2.': 1, 'Assets:': 1, 'Assets': 1, 'items': 1, 'you': 1, 'own': 1, 'can': 1, 'provide': 1, 'future': 1, 'benefit': 1, 'your': 1, 'business,': 1, 'cash,': 1, 'inventory,': 1, 'real': 1, 'estate,': 1, 'office': 1, 'equipment,': 1, 'accounts': 1, 'receivable,': 1, 'which': 1, 'payments': 1, 'due': 1, 'company': 1, 'by': 1, 'customers.': 1, 'There': 1, 'different': 1, 'types': 1, 'assets,': 1, 'including:': 1, 'Current': 1, 'Which': 1, 'be': 1, 'converted': 1, 'cash': 1, 'within': 1, 'year': 1, 'Fixed': 1, 'canâ€™t': 1, 'immediately': 1, 'turned': 1, 'into': 1, 'but': 1, 'tangible': 1, 'owns': 1, 'and': 1, 'uses': 1, 'generate': 1, 'long-term': 1, 'income': 1, '3.': 1, 'Asset': 1, 'Allocation:': 1, 'allocation': 1, 'refers': 1, 'how': 1, 'choose': 1, 'spread': 1, 'money': 1, 'across': 1, 'investment': 1, 'types,': 1, 'also': 1, 'known': 1, 'asset': 1, 'classes.': 1, 'These': 1, 'include:': 1, 'Bonds:': 1, 'Bonds': 1, 'represent': 1, 'form': 1, 'borrowing.': 1, 'When': 1, 'buy': 1, 'bond,': 1, 'typically': 1, 'from': 1, 'government': 1, 'corporation,': 1, 'youâ€™re': 1, 'essentially': 1, 'lending': 1, 'them': 1, 'money.': 1, 'You': 1, 'receive': 1, 'periodic': 1, 'interest': 1, 'get': 1, 'back': 1, 'loaned': 1, 'amount': 1, 'at': 1, 'time': 1, 'bondâ€™s': 1, 'maturityâ€”or': 1, 'defined': 1, 'term': 1, 'bond': 1, 'redeemed.': 1, 'Stocks:': 1, 'A': 1, 'stock': 1, 'share': 1, 'ownership': 1, 'in': 1, 'public': 1, 'private': 1, 'company.': 1, 'become': 1, 'shareholder': 1, 'dividendsâ€”the': 1, 'companyâ€™s': 1, 'profitsâ€”if': 1, 'when': 1, 'they': 1, 'distributed.': 1, 'Cash': 1, 'Equivalents:': 1, 'This': 1, 'any': 1, 'easily': 1, 'event': 1, "it's": 1, 'necessary.': 1, '4.': 1, 'Balance': 1, 'Sheet:': 1, 'balance': 1, 'sheet': 1, 'important': 1, 'financial': 1, 'statement': 1, 'communicates': 1, 'organizationâ€™s': 1, 'worth,': 1, 'â€œbook': 1, 'value.â€\x9d': 1, 'The': 1, 'includes': 1, 'tally': 1, 'liabilities,': 1, 'shareholdersâ€™': 1, 'equity': 1, 'for': 1, 'given': 1, 'reporting': 1, 'period.': 1, 'Sheet': 1, 'Equation:': 1, 'sheets': 1, 'arranged': 1, 'according': 1, 'following': 1, 'equation:': 1, '=': 1, 'Liabilities': 1, '+': 1, 'Ownersâ€™': 1, 'Equity': 1, '5.': 1, 'Capital': 1, 'Gain:': 1, 'capital': 1, 'gain': 1, 'increase': 1, 'value': 1, 'above': 1, 'price': 1, 'initially': 1, 'paid': 1, 'it.': 1, 'If': 1, 'sell': 1, 'less': 1, 'than': 1, 'original': 1, 'purchase': 1, 'price,': 1, 'would': 1, 'considered': 1, 'loss.': 1, 'Related:': 1, '6': 1, 'Ways': 1, 'Understanding': 1, 'Finance': 1, 'Can': 1, 'Help': 1, 'Excel': 1, 'Professionally': 1, '6.': 1, 'Market:': 1, 'market': 1, 'where': 1, 'buyers': 1, 'sellers': 1, 'engage': 1, 'trade': 1, 'including': 1, 'stocks': 1, 'bonds.': 1, 'markets': 1, 'feature': 1, 'several': 1, 'participants,': 1, 'Companies:': 1, 'Firms': 1, 'bonds': 1, 'investors': 1, 'Institutional': 1, 'investors:': 1, 'Investors': 1, 'who': 1, 'on': 1, 'behalf': 1, 'large': 1, 'base': 1, 'Mutual': 1, 'funds:': 1, 'mutual': 1, 'fund': 1, 'institutional': 1, 'investor': 1, 'manages': 1, 'investments': 1, 'thousands': 1, 'individuals': 1, 'Hedge': 1, 'hedge': 1, 'another': 1, 'type': 1, 'investor,': 1, 'controls': 1, 'risk': 1, 'through': 1, 'hedgingâ€”a': 1, 'process': 1, 'buying': 1, 'one': 1, 'then': 1, 'shorting': 1, 'similar': 1, 'make': 1, 'difference': 1, 'their': 1, 'relative': 1, 'performance': 1, '7.': 1, 'Flow:': 1, 'flow': 1, 'net': 1, 'moving': 1, 'out': 1, 'business': 1, 'specific': 1, 'point': 1, 'time.': 1, 'commonly': 1, 'broken': 1, 'three': 1, 'categories,': 1, 'generated': 1, 'normal': 1, 'operations': 1, 'Investing': 1, 'investing': 1, 'activities,': 1, 'securities': 1, 'sale': 1, 'Financing': 1, 'financing': 1, 'debt': 1, 'payments,': 1, 'equity,': 1, 'dividend': 1, '8.': 1, 'Flow': 1, 'Statement:': 1, 'prepared': 1, 'detailed': 1, 'analysis': 1, 'what': 1, 'happened': 1, 'during': 1, 'period': 1, 'document': 1, 'shows': 1, 'spent': 1, 'overview': 1, 'flows': 1, 'operating,': 1, 'investing,': 1, 'activities': 1, '9.': 1, 'Compound': 1, 'Interest:': 1, 'â€œinterest': 1, 'interest.â€\x9d': 1, 'Rather,': 1, 'saving,': 1, 'compound': 1, 'earned': 1, 'deposited,': 1, 'plus': 1, 'youâ€™ve': 1, 'accumulated': 1, 'While': 1, 'it': 1, 'grow': 1, 'savings,': 1, 'debt;': 1, 'charged': 1, 'initial': 1, 'were': 1, 'loaned,': 1, 'well': 1, 'expenses': 1, 'added': 1, 'outstanding': 1, '10.': 1, 'Depreciation:': 1, 'Depreciation': 1, 'represents': 1, 'decrease': 1, 'assetâ€™s': 1, 'value.': 1, 'Itâ€™s': 1, 'used': 1, 'accounting': 1, 'much': 1, 'has': 1}
    
    
    {'Medical:': 1, 'Hospital': 1, 'Emergency': 1, 'Room': 1, '(ER)': 1, 'Intensive': 1, 'Care': 1, 'Unit': 1, '(ICU)': 1, 'Operating': 1, '(OR)': 1, 'Exam': 1, 'Diagnosis': 1, 'Prescription': 1, 'Urine': 1, 'sample': 1, 'Blood': 1, 'Hypertension': 1, 'Cast': 1, 'Vein': 1, 'Syringe': 1, 'Painkiller/pain': 1, 'reliever': 1, 'Numb': 1, 'Dosage': 1, 'Biopsy': 1, '(of': 1, 'abnormal': 1, 'cells)': 1, 'Finanace:': 1, '1.': 1, 'Amortization:': 1, 'Amortization': 1, 'is': 1, 'a': 1, 'method': 1, 'of': 1, 'spreading': 1, 'an': 1, 'intangible': 1, "asset's": 1, 'cost': 1, 'over': 1, 'the': 1, 'course': 1, 'its': 1, 'useful': 1, 'life.': 1, 'Intangible': 1, 'assets': 1, 'are': 1, 'non-physical': 1, 'that': 1, 'essential': 1, 'to': 1, 'company,': 1, 'such': 1, 'as': 1, 'trademark,': 1, 'patent,': 1, 'copyright,': 1, 'or': 1, 'franchise': 1, 'agreement.': 1, '2.': 1, 'Assets:': 1, 'Assets': 1, 'items': 1, 'you': 1, 'own': 1, 'can': 1, 'provide': 1, 'future': 1, 'benefit': 1, 'your': 1, 'business,': 1, 'cash,': 1, 'inventory,': 1, 'real': 1, 'estate,': 1, 'office': 1, 'equipment,': 1, 'accounts': 1, 'receivable,': 1, 'which': 1, 'payments': 1, 'due': 1, 'company': 1, 'by': 1, 'customers.': 1, 'There': 1, 'different': 1, 'types': 1, 'assets,': 1, 'including:': 1, 'Current': 1, 'Which': 1, 'be': 1, 'converted': 1, 'cash': 1, 'within': 1, 'year': 1, 'Fixed': 1, 'canâ€™t': 1, 'immediately': 1, 'turned': 1, 'into': 1, 'but': 1, 'tangible': 1, 'owns': 1, 'and': 1, 'uses': 1, 'generate': 1, 'long-term': 1, 'income': 1, '3.': 1, 'Asset': 1, 'Allocation:': 1, 'allocation': 1, 'refers': 1, 'how': 1, 'choose': 1, 'spread': 1, 'money': 1, 'across': 1, 'investment': 1, 'types,': 1, 'also': 1, 'known': 1, 'asset': 1, 'classes.': 1, 'These': 1, 'include:': 1, 'Bonds:': 1, 'Bonds': 1, 'represent': 1, 'form': 1, 'borrowing.': 1, 'When': 1, 'buy': 1, 'bond,': 1, 'typically': 1, 'from': 1, 'government': 1, 'corporation,': 1, 'youâ€™re': 1, 'essentially': 1, 'lending': 1, 'them': 1, 'money.': 1, 'You': 1, 'receive': 1, 'periodic': 1, 'interest': 1, 'get': 1, 'back': 1, 'loaned': 1, 'amount': 1, 'at': 1, 'time': 1, 'bondâ€™s': 1, 'maturityâ€”or': 1, 'defined': 1, 'term': 1, 'bond': 1, 'redeemed.': 1, 'Stocks:': 1, 'A': 1, 'stock': 1, 'share': 1, 'ownership': 1, 'in': 1, 'public': 1, 'private': 1, 'company.': 1, 'become': 1, 'shareholder': 1, 'dividendsâ€”the': 1, 'companyâ€™s': 1, 'profitsâ€”if': 1, 'when': 1, 'they': 1, 'distributed.': 1, 'Cash': 9, 'Equivalents:': 1, 'This': 1, 'any': 1, 'easily': 1, 'event': 1, "it's": 1, 'necessary.': 1, '4.': 1, 'Balance': 1, 'Sheet:': 1, 'balance': 1, 'sheet': 1, 'important': 1, 'financial': 1, 'statement': 1, 'communicates': 1, 'organizationâ€™s': 1, 'worth,': 1, 'â€œbook': 1, 'value.â€\x9d': 1, 'The': 1, 'includes': 1, 'tally': 1, 'liabilities,': 1, 'shareholdersâ€™': 1, 'equity': 1, 'for': 1, 'given': 1, 'reporting': 1, 'period.': 1, 'Sheet': 1, 'Equation:': 1, 'sheets': 1, 'arranged': 1, 'according': 1, 'following': 1, 'equation:': 1, '=': 1, 'Liabilities': 1, '+': 1, 'Ownersâ€™': 1, 'Equity': 2, '5.': 1, 'Capital': 1, 'Gain:': 1, 'capital': 1, 'gain': 1, 'increase': 1, 'value': 1, 'above': 1, 'price': 1, 'initially': 1, 'paid': 1, 'it.': 1, 'If': 1, 'sell': 1, 'less': 1, 'than': 1, 'original': 1, 'purchase': 1, 'price,': 1, 'would': 1, 'considered': 1, 'loss.': 1, 'Related:': 1, '6': 1, 'Ways': 1, 'Understanding': 1, 'Finance': 1, 'Can': 1, 'Help': 1, 'Excel': 1, 'Professionally': 1, '6.': 1, 'Market:': 1, 'market': 1, 'where': 1, 'buyers': 1, 'sellers': 1, 'engage': 1, 'trade': 1, 'including': 1, 'stocks': 1, 'bonds.': 1, 'markets': 1, 'feature': 1, 'several': 1, 'participants,': 1, 'Companies:': 1, 'Firms': 1, 'bonds': 1, 'investors': 1, 'Institutional': 1, 'investors:': 1, 'Investors': 1, 'who': 1, 'on': 1, 'behalf': 1, 'large': 1, 'base': 1, 'Mutual': 1, 'funds:': 1, 'mutual': 1, 'fund': 1, 'institutional': 1, 'investor': 1, 'manages': 1, 'investments': 1, 'thousands': 1, 'individuals': 1, 'Hedge': 1, 'hedge': 1, 'another': 1, 'type': 1, 'investor,': 1, 'controls': 1, 'risk': 1, 'through': 1, 'hedgingâ€”a': 1, 'process': 1, 'buying': 1, 'one': 1, 'then': 1, 'shorting': 1, 'similar': 1, 'make': 1, 'difference': 1, 'their': 1, 'relative': 1, 'performance': 1, '7.': 1, 'Flow:': 1, 'flow': 1, 'net': 1, 'moving': 1, 'out': 1, 'business': 1, 'specific': 1, 'point': 1, 'time.': 1, 'commonly': 1, 'broken': 1, 'three': 1, 'categories,': 1, 'generated': 1, 'normal': 1, 'operations': 1, 'Investing': 1, 'investing': 1, 'activities,': 1, 'securities': 1, 'sale': 1, 'Financing': 1, 'financing': 1, 'debt': 1, 'payments,': 1, 'equity,': 1, 'dividend': 1, '8.': 1, 'Flow': 1, 'Statement:': 1, 'prepared': 1, 'detailed': 1, 'analysis': 1, 'what': 1, 'happened': 1, 'during': 1, 'period': 1, 'document': 1, 'shows': 1, 'spent': 1, 'overview': 1, 'flows': 1, 'operating,': 1, 'investing,': 1, 'activities': 1, '9.': 1, 'Compound': 1, 'Interest:': 1, 'â€œinterest': 1, 'interest.â€\x9d': 1, 'Rather,': 1, 'saving,': 1, 'compound': 1, 'earned': 1, 'deposited,': 1, 'plus': 1, 'youâ€™ve': 1, 'accumulated': 1, 'While': 1, 'it': 1, 'grow': 1, 'savings,': 1, 'debt;': 1, 'charged': 1, 'initial': 1, 'were': 1, 'loaned,': 1, 'well': 1, 'expenses': 1, 'added': 1, 'outstanding': 1, '10.': 1, 'Depreciation:': 1, 'Depreciation': 1, 'represents': 1, 'decrease': 1, 'assetâ€™s': 1, 'value.': 1, 'Itâ€™s': 1, 'used': 1, 'accounting': 1, 'much': 1, 'has': 1}

```python
if (len(', '.join(str(x) for x in finan_d.values() if x == 2)) > len(', '.join(str(x) for x in med_d.values()  if x == 2))):
    print("The document is related to finance")
else:
    print("The document is related to medical")
```

    The document is related to finance

#### Hence from content stored in the textfile, we can see that the document is related to finance.
