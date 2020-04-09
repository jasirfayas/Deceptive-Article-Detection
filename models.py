import pandas as pd
import numpy as np

from sklearn.utils import shuffle
head_ds = pd.read_csv('dataset/clickbaitdata.csv')
head_ds = head_ds.drop(['index','id'], axis = 1)
                     
fake_raw = pd.read_csv('dataset/fake.csv')
fake_raw=fake_raw.drop(['uuid','ord_in_thread','author','published','title','language','crawled','site_url','country','domain_rank','spam_score','main_img_url','replies_count','participants_count','likes','comments','shares','type'], axis = 1)

fake_ds = fake_raw.dropna(axis = 0, how ='any')
fake_ds.columns = ['article','titles']

fake_ds_head = fake_ds.drop(['article'],axis=1)
fake_ds_head['clickbait'] = 1

# fake_ds_body = fake_ds.drop(['titles'],axis=1)
# fake_ds_body['label'] = 1

headlines_fake=pd.read_csv('dataset/clickbait_data', sep="\n", header=None, names=['titles'])
headlines_real=pd.read_csv('dataset/non_clickbait_data', sep="\n", header=None, names=['titles'])

headlines_fake['clickbait'] = 1
headlines_real['clickbait'] = 0

head_ds2 = pd.concat([headlines_fake,headlines_real])
head_ds2 = shuffle(head_ds2, random_state=27).reset_index(drop=True)

head_combined = pd.concat([head_ds,head_ds2,fake_ds_head],sort=False)
head_combined = shuffle(head_combined, random_state=27).reset_index(drop=True)

body_ds = pd.read_csv('dataset/fr_news.csv')
body_ds.drop(body_ds.columns[body_ds.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
body_ds = body_ds.drop(['title'], axis = 1)
body_ds.columns = ['article', 'label']

train_ds=pd.read_csv('dataset/train.csv')
test_ds=pd.read_csv('dataset/test.csv')

tt_all = pd.concat([train_ds,test_ds])
tt_all = shuffle(tt_all, random_state=27).reset_index(drop=True)
tt_all.columns = ['article', 'label']

# body_ds2 = pd.concat([tt_all,fake_ds_body],sort=True)
# body_ds2 = shuffle(body_ds2, random_state=27).reset_index(drop=True)

# body_combined = pd.concat([body_ds,body_ds2],sort=True)
body_combined = pd.concat([body_ds,tt_all],sort=True)
body_combined = shuffle(body_combined, random_state=27).reset_index(drop=True)

body_combined.label = body_combined.label.replace({True: "REAL",False: "FAKE"})
body_combined.label = body_combined.label.replace({"REAL": 0, "FAKE": 1})

# head_all.describe()

# checking for Nulls
body_combined.isnull().sum()
head_combined.isnull().sum()

#splitting to train-test dataset
from sklearn.model_selection import train_test_split
X = body_combined.article
y = body_combined.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

X2 = head_combined.titles
y2 = head_combined.clickbait
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.25)


#pipeline1 NaiveBayes
print("\n||||||||-------------NaiveBayes------------|||||||||")
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
pipe_body_nb = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('mnb', MultinomialNB())])
pipe_body_nb.fit(X, y)
predicted1 = pipe_body_nb.predict(X_test)
print (np.mean(predicted1 == y_test))
from sklearn.metrics import classification_report, confusion_matrix
print (classification_report(y_test, predicted1))
confusion_matrix(y_test, predicted1)

#pipeline5 NaiveBayes
pipe_head_nb = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('mnb', MultinomialNB())])
pipe_head_nb.fit(X2, y2)
predicted5 = pipe_head_nb.predict(X_test2)
print (np.mean(predicted5 == y_test2))
print (classification_report(y_test2, predicted5))
confusion_matrix(y_test2, predicted5)


#pipeline2 KNN
print("\n\n|||||||||----------------KNN---------------||||||||||")
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn import svm
pipe_body_knn = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('knn', KNeighborsClassifier())])
pipe_body_knn.fit(X, y)
predicted2 = pipe_body_knn.predict(X_test)
print (np.mean(predicted2 == y_test))
print (classification_report(y_test, predicted2))
confusion_matrix(y_test, predicted2)

pipe_head_knn= Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('knn', KNeighborsClassifier())])
pipe_head_knn.fit(X2, y2)
predicted6 = pipe_head_knn.predict(X_test2)
print (np.mean(predicted6 == y_test2))
print (classification_report(y_test2, predicted6))
confusion_matrix(y_test2, predicted6)

#pipeline3 RandomForest
print("\n\n||||||||||--------RandomForest---------||||||||||||")
from sklearn.ensemble import RandomForestClassifier
pipe_body_rf = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('rf', RandomForestClassifier())])
pipe_body_rf.fit(X, y)
predicted3 = pipe_body_rf.predict(X_test)
print (np.mean(predicted3 == y_test))
print (classification_report(y_test, predicted3))
confusion_matrix(y_test, predicted3)

pipe_head_rf = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('rf', RandomForestClassifier())])
pipe_head_rf.fit(X2, y2)
predicted7 = pipe_head_rf.predict(X_test2)
print (np.mean(predicted7 == y_test2))
print (classification_report(y_test2, predicted7))
confusion_matrix(y_test2, predicted7)

#pipeline Logistic Regression
print("\n\n||||||||--------LogisticRegression--------|||||||||")
from sklearn.linear_model import  LogisticRegression
pipe_body_lgr = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('lgr', LogisticRegression())])
pipe_body_lgr.fit(X, y)
predicted4 = pipe_body_lgr.predict(X_test)
print (np.mean(predicted4 == y_test))
print (classification_report(y_test, predicted4))
confusion_matrix(y_test, predicted4)

pipe_head_lgr = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('lgr', LogisticRegression())])
pipe_head_lgr.fit(X2, y2)
predicted8 = pipe_head_lgr.predict(X_test2)
print (np.mean(predicted8 == y_test2))
print (classification_report(y_test2, predicted8))
confusion_matrix(y_test2, predicted8)

#Serialising objects using joblib
import joblib
print("\n\n||||DUMPING PIPELINES AS EXTERNAL FILES............|||")

joblib.dump(pipe_body_nb, 'body_nb.pkl') 
joblib.dump(pipe_body_knn, 'body_knn.pkl') 
joblib.dump(pipe_body_rf, 'body_rf.pkl') 
joblib.dump(pipe_body_lgr, 'body_lgr.pkl')

joblib.dump(pipe_head_nb, 'head_nb.pkl') 
joblib.dump(pipe_head_knn, 'head_knn.pkl')
joblib.dump(pipe_head_rf, 'head_rf.pkl')  
joblib.dump(pipe_head_lgr, 'head_lgr.pkl')  

print("\n\nclassifier training is completed successfully............")