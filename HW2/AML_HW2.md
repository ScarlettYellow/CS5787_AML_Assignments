# ML Programming 

NLP with Disaster Tweets

- problem: social media monitoring - disasters; predicts which Tweets are about real disasters and which one’s aren’t
- type: classification
- dataset
  - a dataset of 10,000 tweets that were hand classified.



data cleaning 

- 删除 tag，超链接，HTML字符，代码，表情符号，语气符号，@ #等内容。
- 删除“ Keyboard”和“ location”列，并仅使用推文文本信息，因为我们只做基于文本的分类

 

work allocation 

- a,b,c - pre
  - work
    - a - describe data
    - b - split data
    - c - preprocess data
  - allocation 
    - 1 - done by one person
    - 2 - done by both, write by one
- d-i @sun evening 
  - work
    - d - bag
    - e - NB Implementation
    - f, g, h - lr, linear svm, non-linear svm
    - i - n-gram
  - allocation 
    - 1 - d,e,i  @me
    - 2 - d,f,g,h @zihan
- i-j @mon evening 
  - work
    - i - n-gram, repeat e-h
    - j - add additional cols, repeat f-h
  - allocation 
    - 1 - i
    - 2 - j 
- k-l @Tue evening 
  - work
    - k - use full trainset, rebuild model using the best approach 
    - l - how u choose model if considering interpretability
  - allocation 
    - 1 - done by one person
    - 2 - one k, one l





BOW

- M: use model to choose M

*F1的核心思想在于，在尽可能地提高精确度(Precision)和召回率(Recall)的同时，也希望两者之间的差异尽可能的小*。









































































































