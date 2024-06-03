---
title: Corpus Analyzing
date: '2024-06-03'
---
Ref:

https://www.nltk.org/book/ch02.html

```python
import nltk
```

```python
#To know various files included in nltk corpus from Gutenberg
files= nltk.corpus.gutenberg.fileids()
print(files)
print(type(files))
```

    ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']
    <class 'list'>

```python
#To know words in Emma by Jane Austen
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)
```

    192427

```python
#To apply concordance
emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
emma.concordance("surprize")
```

    Displaying 25 of 37 matches:
    er father , was sometimes taken by surprize at his being still able to pity ` 
    hem do the other any good ." " You surprize me ! Emma must do Harriet good : a
    Knightley actually looked red with surprize and displeasure , as he stood up ,
    r . Elton , and found to his great surprize , that Mr . Elton was actually on 
    d aid ." Emma saw Mrs . Weston ' s surprize , and felt that it must be great ,
    father was quite taken up with the surprize of so sudden a journey , and his f
    y , in all the favouring warmth of surprize and conjecture . She was , moreove
    he appeared , to have her share of surprize , introduction , and pleasure . Th
    ir plans ; and it was an agreeable surprize to her , therefore , to perceive t
    talking aunt had taken me quite by surprize , it must have been the death of m
    f all the dialogue which ensued of surprize , and inquiry , and congratulation
     the present . They might chuse to surprize her ." Mrs . Cole had many to agre
    the mode of it , the mystery , the surprize , is more like a young woman ' s s
     to her song took her agreeably by surprize -- a second , slightly but correct
    " " Oh ! no -- there is nothing to surprize one at all .-- A pretty fortune ; 
    t to be considered . Emma ' s only surprize was that Jane Fairfax should accep
    of your admiration may take you by surprize some day or other ." Mr . Knightle
    ation for her will ever take me by surprize .-- I never had a thought of her i
     expected by the best judges , for surprize -- but there was great joy . Mr . 
     sound of at first , without great surprize . " So unreasonably early !" she w
    d Frank Churchill , with a look of surprize and displeasure .-- " That is easy
    ; and Emma could imagine with what surprize and mortification she must be retu
    tled that Jane should go . Quite a surprize to me ! I had not the least idea !
     . It is impossible to express our surprize . He came to speak to his father o
    g engaged !" Emma even jumped with surprize ;-- and , horror - struck , exclai

```python
#To use import function to call Gutenberg corpus
from nltk.corpus import gutenberg
files=gutenberg.fileids()
print(files)
```

    ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']

```python
emma = gutenberg.words('austen-emma.txt')
len(emma)
```

    192427

```python
#Write a function to know length of each text in the corpus

def text_analyse(text_name):
    num_chars = len(gutenberg.raw(text_name))
    num_words = len(gutenberg.words(text_name))
    num_sents = len(gutenberg.sents(text_name))
    num_vocab = len(set(w.lower() for w in gutenberg.words(text_name)))
    return round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), text_name
```

```python
#Apply for loop to access all the corpus in gutenberg

from nltk.corpus import gutenberg

files=gutenberg.fileids()

for text in files:
    print(text_analyse(text))
```

    (5, 25, 26, 'austen-emma.txt')
    (5, 26, 17, 'austen-persuasion.txt')
    (5, 28, 22, 'austen-sense.txt')
    (4, 34, 79, 'bible-kjv.txt')
    (5, 19, 5, 'blake-poems.txt')
    (4, 19, 14, 'bryant-stories.txt')
    (4, 18, 12, 'burgess-busterbrown.txt')
    (4, 20, 13, 'carroll-alice.txt')
    (5, 20, 12, 'chesterton-ball.txt')
    (5, 23, 11, 'chesterton-brown.txt')
    (5, 18, 11, 'chesterton-thursday.txt')
    (4, 21, 25, 'edgeworth-parents.txt')
    (5, 26, 15, 'melville-moby_dick.txt')
    (5, 52, 11, 'milton-paradise.txt')
    (4, 12, 9, 'shakespeare-caesar.txt')
    (4, 12, 8, 'shakespeare-hamlet.txt')
    (4, 12, 7, 'shakespeare-macbeth.txt')
    (5, 36, 12, 'whitman-leaves.txt')

### Explore brown corpus

```python
from nltk.corpus import brown
```

```python
files= brown.fileids()
print(files)
```

    ['ca01', 'ca02', 'ca03', 'ca04', 'ca05', 'ca06', 'ca07', 'ca08', 'ca09', 'ca10', 'ca11', 'ca12', 'ca13', 'ca14', 'ca15', 'ca16', 'ca17', 'ca18', 'ca19', 'ca20', 'ca21', 'ca22', 'ca23', 'ca24', 'ca25', 'ca26', 'ca27', 'ca28', 'ca29', 'ca30', 'ca31', 'ca32', 'ca33', 'ca34', 'ca35', 'ca36', 'ca37', 'ca38', 'ca39', 'ca40', 'ca41', 'ca42', 'ca43', 'ca44', 'cb01', 'cb02', 'cb03', 'cb04', 'cb05', 'cb06', 'cb07', 'cb08', 'cb09', 'cb10', 'cb11', 'cb12', 'cb13', 'cb14', 'cb15', 'cb16', 'cb17', 'cb18', 'cb19', 'cb20', 'cb21', 'cb22', 'cb23', 'cb24', 'cb25', 'cb26', 'cb27', 'cc01', 'cc02', 'cc03', 'cc04', 'cc05', 'cc06', 'cc07', 'cc08', 'cc09', 'cc10', 'cc11', 'cc12', 'cc13', 'cc14', 'cc15', 'cc16', 'cc17', 'cd01', 'cd02', 'cd03', 'cd04', 'cd05', 'cd06', 'cd07', 'cd08', 'cd09', 'cd10', 'cd11', 'cd12', 'cd13', 'cd14', 'cd15', 'cd16', 'cd17', 'ce01', 'ce02', 'ce03', 'ce04', 'ce05', 'ce06', 'ce07', 'ce08', 'ce09', 'ce10', 'ce11', 'ce12', 'ce13', 'ce14', 'ce15', 'ce16', 'ce17', 'ce18', 'ce19', 'ce20', 'ce21', 'ce22', 'ce23', 'ce24', 'ce25', 'ce26', 'ce27', 'ce28', 'ce29', 'ce30', 'ce31', 'ce32', 'ce33', 'ce34', 'ce35', 'ce36', 'cf01', 'cf02', 'cf03', 'cf04', 'cf05', 'cf06', 'cf07', 'cf08', 'cf09', 'cf10', 'cf11', 'cf12', 'cf13', 'cf14', 'cf15', 'cf16', 'cf17', 'cf18', 'cf19', 'cf20', 'cf21', 'cf22', 'cf23', 'cf24', 'cf25', 'cf26', 'cf27', 'cf28', 'cf29', 'cf30', 'cf31', 'cf32', 'cf33', 'cf34', 'cf35', 'cf36', 'cf37', 'cf38', 'cf39', 'cf40', 'cf41', 'cf42', 'cf43', 'cf44', 'cf45', 'cf46', 'cf47', 'cf48', 'cg01', 'cg02', 'cg03', 'cg04', 'cg05', 'cg06', 'cg07', 'cg08', 'cg09', 'cg10', 'cg11', 'cg12', 'cg13', 'cg14', 'cg15', 'cg16', 'cg17', 'cg18', 'cg19', 'cg20', 'cg21', 'cg22', 'cg23', 'cg24', 'cg25', 'cg26', 'cg27', 'cg28', 'cg29', 'cg30', 'cg31', 'cg32', 'cg33', 'cg34', 'cg35', 'cg36', 'cg37', 'cg38', 'cg39', 'cg40', 'cg41', 'cg42', 'cg43', 'cg44', 'cg45', 'cg46', 'cg47', 'cg48', 'cg49', 'cg50', 'cg51', 'cg52', 'cg53', 'cg54', 'cg55', 'cg56', 'cg57', 'cg58', 'cg59', 'cg60', 'cg61', 'cg62', 'cg63', 'cg64', 'cg65', 'cg66', 'cg67', 'cg68', 'cg69', 'cg70', 'cg71', 'cg72', 'cg73', 'cg74', 'cg75', 'ch01', 'ch02', 'ch03', 'ch04', 'ch05', 'ch06', 'ch07', 'ch08', 'ch09', 'ch10', 'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16', 'ch17', 'ch18', 'ch19', 'ch20', 'ch21', 'ch22', 'ch23', 'ch24', 'ch25', 'ch26', 'ch27', 'ch28', 'ch29', 'ch30', 'cj01', 'cj02', 'cj03', 'cj04', 'cj05', 'cj06', 'cj07', 'cj08', 'cj09', 'cj10', 'cj11', 'cj12', 'cj13', 'cj14', 'cj15', 'cj16', 'cj17', 'cj18', 'cj19', 'cj20', 'cj21', 'cj22', 'cj23', 'cj24', 'cj25', 'cj26', 'cj27', 'cj28', 'cj29', 'cj30', 'cj31', 'cj32', 'cj33', 'cj34', 'cj35', 'cj36', 'cj37', 'cj38', 'cj39', 'cj40', 'cj41', 'cj42', 'cj43', 'cj44', 'cj45', 'cj46', 'cj47', 'cj48', 'cj49', 'cj50', 'cj51', 'cj52', 'cj53', 'cj54', 'cj55', 'cj56', 'cj57', 'cj58', 'cj59', 'cj60', 'cj61', 'cj62', 'cj63', 'cj64', 'cj65', 'cj66', 'cj67', 'cj68', 'cj69', 'cj70', 'cj71', 'cj72', 'cj73', 'cj74', 'cj75', 'cj76', 'cj77', 'cj78', 'cj79', 'cj80', 'ck01', 'ck02', 'ck03', 'ck04', 'ck05', 'ck06', 'ck07', 'ck08', 'ck09', 'ck10', 'ck11', 'ck12', 'ck13', 'ck14', 'ck15', 'ck16', 'ck17', 'ck18', 'ck19', 'ck20', 'ck21', 'ck22', 'ck23', 'ck24', 'ck25', 'ck26', 'ck27', 'ck28', 'ck29', 'cl01', 'cl02', 'cl03', 'cl04', 'cl05', 'cl06', 'cl07', 'cl08', 'cl09', 'cl10', 'cl11', 'cl12', 'cl13', 'cl14', 'cl15', 'cl16', 'cl17', 'cl18', 'cl19', 'cl20', 'cl21', 'cl22', 'cl23', 'cl24', 'cm01', 'cm02', 'cm03', 'cm04', 'cm05', 'cm06', 'cn01', 'cn02', 'cn03', 'cn04', 'cn05', 'cn06', 'cn07', 'cn08', 'cn09', 'cn10', 'cn11', 'cn12', 'cn13', 'cn14', 'cn15', 'cn16', 'cn17', 'cn18', 'cn19', 'cn20', 'cn21', 'cn22', 'cn23', 'cn24', 'cn25', 'cn26', 'cn27', 'cn28', 'cn29', 'cp01', 'cp02', 'cp03', 'cp04', 'cp05', 'cp06', 'cp07', 'cp08', 'cp09', 'cp10', 'cp11', 'cp12', 'cp13', 'cp14', 'cp15', 'cp16', 'cp17', 'cp18', 'cp19', 'cp20', 'cp21', 'cp22', 'cp23', 'cp24', 'cp25', 'cp26', 'cp27', 'cp28', 'cp29', 'cr01', 'cr02', 'cr03', 'cr04', 'cr05', 'cr06', 'cr07', 'cr08', 'cr09']

```python
brown.categories()
```

    ['adventure',
     'belles_lettres',
     'editorial',
     'fiction',
     'government',
     'hobbies',
     'humor',
     'learned',
     'lore',
     'mystery',
     'news',
     'religion',
     'reviews',
     'romance',
     'science_fiction']

```python
from urllib import request
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
print(type(raw))

print(len(raw))

print(raw[1000:1200])
```

    <class 'str'>
    1176812
    CE
    
    A few words about Dostoevsky himself may help the English reader to
    understand his work.
    
    Dostoevsky was the son of a doctor. His parents were very hard-working
    and deeply religious people, 

