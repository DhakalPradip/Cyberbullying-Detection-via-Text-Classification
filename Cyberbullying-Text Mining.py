#!/usr/bin/env python
# coding: utf-8

# Importing all the required packages

# In[1]:


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import met
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from wordcloud import WordCloud
from sklearn.model_selection import GridSearchCV


# In[2]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# # Loading Data

# In[3]:


def openTextasList(filename):
  with open(filename, encoding="utf8") as file_in:
    lines = []
    for line in file_in:
      # remove whitespace characters like `\n` at the end of each line
      line=line.strip()
      lines.append(line)
  return(lines)


# In[4]:


ethnicity=openTextasList("..\DataMiningIFinalProject/8000ethnicity.txt")
religion=openTextasList("..\DataMiningIFinalProject/8000religion.txt")
notcb=openTextasList("..\DataMiningIFinalProject/8000notcb.txt")
alldoc=notcb + ethnicity + religion
print("There are %d ethnicity tweets:\n %s " % (len(ethnicity),ethnicity[0:5]))
print("There are %d religion tweets:\n %s " % (len(religion),religion[0:5]))
print("There are %d notcb tweets:\n %s " % (len(notcb),notcb[0:5]))


# In[5]:


alldoc[0:2]


# In[6]:


# define training labels
class_label = np.array(["notcb"for _ in range(8000)] + 
                       ["ethencity"for _ in range(8000)] + 
                       ["religion"for _ in range(8000)])
class_label.shape


# In[7]:


# Construct a dataframe
lst = alldoc

# list of int
lst2 = class_label

# zipping both lists with columns specified
df = pd.DataFrame(list(zip(lst, lst2)),
            columns =['Tweets', 'Labels'])
df


# In[8]:


df['Labels'].value_counts()


# # Data Cleaning

# In[9]:


np.sum(df.isnull())


# In[10]:


# storing both Tweets and Labels in lists
tweets, labels = list(df['Tweets']), list(df['Labels'])


# In[11]:


# labels
labelencoder = LabelEncoder()
df['LabelsEncoded'] = labelencoder.fit_transform(df['Labels'])

#Changing notcb to 0
df.LabelsEncoded = df.LabelsEncoded.replace([0,1,2], [1,0,2])
df[['Labels', 'LabelsEncoded']].value_counts()


# In[12]:


# converting tweets to lower case
df['Tweets'] = df['Tweets'].str.lower()
df.head()


# In[13]:


# removing stopwords
def RemoveStopWords(input_text):
    StopWordsList = stopwords.words('english')
    # Words that might indicate some sentiments are assigned to 
    # WhiteList and are not removed
    WhiteList = ["n't", "not", "no"]
    words = input_text.split() 
    CleanWords = [word for word in words if (word not in StopWordsList or word in WhiteList) and len(word) > 1] 
    return " ".join(CleanWords)

df.Tweets = df["Tweets"].apply(RemoveStopWords)
df.Tweets.head()


# In[14]:


# removing URLs
def RemoveURLs(text):
    return re.sub(r"((www.[^s]+)|(http\S+))","",text)

df['Tweets'] = df['Tweets'].apply(lambda x : RemoveURLs(x))
df.Tweets.head()


# In[15]:


# removing mentions
def MentionsRemover(input_text):
    return re.sub(r'@\w+', '', input_text)

df.Tweets = df["Tweets"].apply(MentionsRemover)
df.Tweets.head()


# In[16]:


# removing numeric data
def RemoveNumeric(text):
    return re.sub('[0-9]+', '', text)
    
df['Tweets'] = df['Tweets'].apply(lambda x: RemoveNumeric(x))
df.Tweets.head()


# In[17]:


# removing punctuations
Punctuations = string.punctuation
print(Punctuations)

def RemovePunctuations(text):
    translator = str.maketrans('','', Punctuations)
    return text.translate(translator)

df['Tweets'] = df['Tweets'].apply(lambda x : RemovePunctuations(x))
df.Tweets.head()


# In[18]:


# removing emojis
def RemoveEmoji(text):
    EmojiPattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return EmojiPattern.sub(r'',text)

df['Tweets'] = df['Tweets'].apply(lambda x : RemoveEmoji(x))
df.Tweets.head()


# In[19]:


# Tokenization of tweets
df.Tweets = df.Tweets.tolist()

TokenizeText = [word_tokenize(i) for i in df.Tweets]
# for i in TokenizeText:
#     print(i)
df.Tweets = TokenizeText
print(df.Tweets.head())


# In[20]:


# Lemmatization
lemmatizer = WordNetLemmatizer()

def Lemmatization(text):
    text = [lemmatizer.lemmatize(word) for word in text]
    return text


df['Tweets'] = df['Tweets'].apply(lambda x: Lemmatization(x))
print(df['Tweets'].head())


# In[21]:


df


# In[22]:


# Joining all words with spaces
df['Tweets'] = df['Tweets'].apply(lambda x : " ".join(x))
df


# # Word Clouds

# In[25]:


# NOT cyberbullying tweets
NotCbDf = df.loc[df['LabelsEncoded'] == 0]
# Converting all tweets into a single list and then to single string
NotCbDfTweets = NotCbDf.Tweets.tolist()
NotCbDfTweets = " ".join(NotCbDfTweets)


# In[26]:


#pip install wordcloud


# In[27]:


# Word Cloud for NOT cyberbullying tweets
wordcloud = WordCloud(max_words=50).generate(NotCbDfTweets)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[28]:


# Etnicity tweets
EthnicityDf = df.loc[df['LabelsEncoded'] == 1]
# Converting all tweets into a single list and then to single string
EthnicityDfTweets = EthnicityDf.Tweets.tolist()
EthnicityDfTweets = " ".join(EthnicityDfTweets)


# In[29]:


# Word Cloud for etnicity tweets
wordcloud = WordCloud(max_words=50).generate(EthnicityDfTweets)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[30]:


# Religion tweets
ReligionDf = df.loc[df['LabelsEncoded'] == 2]
# Converting all tweets into a single list and then to single string
ReligionDfTweets = ReligionDf.Tweets.tolist()
ReligionDfTweets = " ".join(ReligionDfTweets)


# In[31]:


# Word Cloud for religion tweets
wordcloud = WordCloud(max_words=50).generate(ReligionDfTweets)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # Splitting the data into train-test

# In[32]:


# Splitting the data
X, y = df['Tweets'], df['LabelsEncoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 111)

print(X_train.shape)
print(X_test.shape)


# # Bag of Words and TF-IDF

# Bag of Words

# In[33]:


# Bag of Words
BoW = CountVectorizer(ngram_range= (1,1))
# Train data
BoW_X_train = BoW.fit_transform(X_train)
print(BoW_X_train.toarray())
print(BoW_X_train.toarray().shape)
# Test data
BoW_X_test = BoW.transform(X_test)
print(BoW_X_test.toarray())
print(BoW_X_test.toarray().shape)


# In[34]:


#Check
BoW_X_train.toarray()[100][21900:21950]


# TF-IDF

# In[35]:


# TF-IDF
TF_IDF = TfidfVectorizer(ngram_range=(1,1), max_features= 200000)
#Train Data
TF_IDF_X_train = TF_IDF.fit_transform(X_train)
print(TF_IDF_X_train.toarray())
print(TF_IDF_X_train.toarray().shape)
# Test Data
TF_IDF_X_test = TF_IDF.transform(X_test)
print(TF_IDF_X_test.toarray())
print(TF_IDF_X_test.toarray().shape)


# In[36]:


#Check
TF_IDF_X_train.toarray()[100][21900:21950]


# # Modeling

# Logistic Regression

# In[39]:


# Function for logistic regression to compare Bag of Words and TF-IDF
def LogisticRegressionFunction(X_train, X_test, y_train, y_test, description):
    LogitClassifier = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=111, n_jobs=-1)
    LogitClassifier.fit(X_train, y_train)
    y_prediction = LogitClassifier.predict(X_test)
    ConfMat = confusion_matrix(y_test, y_prediction)
    display = ConfusionMatrixDisplay(confusion_matrix= ConfMat)
    display.plot()
    plt.show()
    Accuracy = metrics.accuracy_score(y_test, y_prediction)
    F1 = metrics.f1_score(y_test, y_prediction, average='weighted')
    print('Accuracy for ', description,' is: {:.2f}'.format(Accuracy))
    print('F1 score for ', description,' is: {:.2f}'.format(F1))


# In[40]:


# Logistic regression with Bag of Words
LogisticRegressionFunction(BoW_X_train, BoW_X_test, y_train, y_test, 'Logistic Regression with Bag of Words')


# In[41]:


# Logistic regression with TF-IDF
LogisticRegressionFunction(TF_IDF_X_train, TF_IDF_X_test, y_train, y_test, 'Logistic regression with TF-IDF')


# Multinomial Naive Bayes 

# In[42]:


# Function for multinomial naive bayes to compare Bag of Words and TF-IDF
def MultinomialNaiveBayes(X_train, X_test, y_train, y_test, description):
    MultiNaiveBayesClassifier = MultinomialNB()
    MultiNaiveBayesClassifier.fit(X_train, y_train)
    y_prediction = MultiNaiveBayesClassifier.predict(X_test)
    ConfMat = confusion_matrix(y_test, y_prediction)
    display = ConfusionMatrixDisplay(confusion_matrix= ConfMat)
    display.plot()
    plt.show()
    Accuracy = metrics.accuracy_score(y_test, y_prediction)
    F1 = metrics.f1_score(y_test, y_prediction, average='weighted')
    print('Accuracy for ', description,' is: {:.2f}'.format(Accuracy))
    print('F1 score for ', description,' is: {:.2f}'.format(F1))


# In[43]:


# Multinomial Naive Bayes with Bag of Words
MultinomialNaiveBayes(BoW_X_train, BoW_X_test,y_train, y_test, 'Multinomial Naive Bayes with Bag of Words')


# In[44]:


# Multinomial Naive Bayes with TF-IDF
MultinomialNaiveBayes(TF_IDF_X_train, TF_IDF_X_test, y_train, y_test, 'Multinomial Naive Bayes with TF-IDF')


# K-Nearest Neighbor

# First, lets find the best value of k.

# In[45]:


# define grid parameters
grid_params = { 'n_neighbors' : list(range(1,26))}
# grid search
GS = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=5, n_jobs = -1)


# In[46]:


# fit the model for Bag of Words
GridResult = GS.fit(BoW_X_train, y_train)
# hyperparameters with the best score
GridResult.best_params_


# In[47]:


# fit the model for Bag of Words
GridResult = GS.fit(TF_IDF_X_train, y_train)
# hyperparameters with the best score
GridResult.best_params_


# In[48]:


# Function for knn to compare Bag of Words and TF-IDF
def KnnFunction(X_train, X_test, y_train, y_test, n, description):
    KnnClassifier = KNeighborsClassifier(n_neighbors= n)
    KnnClassifier.fit(X_train, y_train)
    y_prediction = KnnClassifier.predict(X_test)
    ConfMat = confusion_matrix(y_test, y_prediction)
    display = ConfusionMatrixDisplay(confusion_matrix= ConfMat)
    display.plot()
    plt.show()
    Accuracy = metrics.accuracy_score(y_test, y_prediction)
    F1 = metrics.f1_score(y_test, y_prediction, average='weighted')
    print('Accuracy for ', description,' is: {:.2f}'.format(Accuracy))
    print('F1 score for ', description,' is: {:.2f}'.format(F1))


# In[49]:


# KNN with Bag of Words
KnnFunction(BoW_X_train, BoW_X_test,y_train, y_test, 1, 'KNN with Bag of Words')


# In[50]:


# KNN with TF-IDF
KnnFunction(TF_IDF_X_train, TF_IDF_X_test, y_train, y_test, 1, 'KNN with TF-IDF')


# Extreme Gradient Boosting

# In[51]:


# Function for xgboost to compare Bag of Words and TF-IDF
def XgBoostFunction(X_train, X_test, y_train, y_test, learning_rate, max_depth, description):
    XgBoostClassifier = xgb.XGBClassifier(objective = 'multi:softmax',
                                          learning_rate = learning_rate, max_depth = max_depth, seed = 111)
    XgBoostClassifier.fit(X_train, y_train)
    y_prediction = XgBoostClassifier.predict(X_test)
    ConfMat = confusion_matrix(y_test, y_prediction)
    display = ConfusionMatrixDisplay(confusion_matrix= ConfMat)
    display.plot()
    plt.show()
    Accuracy = metrics.accuracy_score(y_test, y_prediction)
    F1 = metrics.f1_score(y_test, y_prediction, average='weighted')
    print('Accuracy for ', description,' is: {:.2f}'.format(Accuracy))
    print('F1 score for ', description,' is: {:.2f}'.format(F1))


# The default value for learning_rate is 0.3 and max_depth is 6.

# In[52]:


# Xgboost with Bag of Words (using default parameters)
XgBoostFunction(BoW_X_train, BoW_X_test,y_train, y_test, 0.3, 6, 'Xgboost with Bag of Words')


# In[53]:


# Xgboost with TF-IDF (Using default parameters) 
XgBoostFunction(TF_IDF_X_train, TF_IDF_X_test, y_train, y_test, 0.3, 6, 'Xgboost with TF-IDF')


# Let's tune a couple of parameters: learning_rate and max_depth.

# In[54]:


# define parameters
params = { 'max_depth': [3, 4, 5, 6, 7],
           'learning_rate': [0.01, 0.1, 0.3, 0.5, 0.7] }
# grid search
GrdSrch = GridSearchCV(estimator= xgb.XGBClassifier(objective = 'multi:softmax', seed = 111), 
                       param_grid= params,
                       scoring='accuracy', 
                       verbose=1)


# In[55]:


# fit the model for Bag of Words
GrdSrchResult = GrdSrch.fit(BoW_X_train, y_train)
# hyperparameters with the best score
GrdSrchResult.best_params_


# In[56]:


# fit the model for TF-IDF
GrdSrchResult = GrdSrch.fit(TF_IDF_X_train, y_train)
# hyperparameters with the best score
GrdSrchResult.best_params_


# In[57]:


# Xgboost with Bag of Words (using tuned parameters)
XgBoostFunction(BoW_X_train, BoW_X_test,y_train, y_test, 0.5, 4, 'Xgboost with Bag of Words')


# For TF-IDF, we got default values as the best ones.

# In[ ]:




