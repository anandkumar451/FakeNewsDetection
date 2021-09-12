import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv("C:\\Users\\anand\\OneDrive\\Desktop\\news.csv")
labels = df.label

x_train, x_test, y_train, y_test = train_test_split(df['text'],labels, test_size=0.2, random_state=7)

tdidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df=0.7)

tdidf_train = tdidf_vectorizer.fit_transform(x_train)
tdidf_test = tdidf_vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tdidf_train,y_train)
y_pred = pac.predict(tdidf_test)
score = accuracy_score(y_test, y_pred)
print('Accuracy: %f', score*100,'%%')

cm=confusion_matrix(y_test,y_pred,labels = ['FAKE','REAL'],normalize='true')

df_cm = pd.DataFrame(cm,index=['FAKE','REAL'],columns=['FAKE','REAL'])
plt.figure(figsize=(10,10))
sn.heatmap(df_cm,annot=True)

plt.show()