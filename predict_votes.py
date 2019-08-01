# -*- coding: utf-8 -*-
"""
Program: Predicting Users' Votes
Description: Extract data, manipulate and create a model.
            Split data intr train and test data.
            Train model and predict values for test data.
Last Modified: 02 May 2019
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('pictures-train.tsv', delimiter='\t')

#Drop row with negative value
data.loc[data.votes < 0, 'votes']=np.nan
data.loc[data.viewed < 0, 'viewed']=np.nan
data.loc[data.n_comments < 0, 'n_comments']=np.nan

#Transfer data to pandaDataTime
data['takenon'] = data['takenon'].replace({'-00':'-01'},regex=True)
data['takenon'] = pd.to_datetime(data['takenon'])
years = data['takenon']
data['takenon'] = data['takenon']-data['takenon'].min()
data['takenon'] = data['takenon'].astype('timedelta64[D]')
data['votedon'] = pd.to_datetime(data['votedon'])
data['votedon'] = data['votedon']-data['votedon'].min()
data['votedon'] = data['votedon'].astype('timedelta64[D]')

df = data
df['year'] = years.dt.year

#get etitle and region as dummy
edummy = pd.get_dummies(data['etitle'],columns=['etitle']).astype(bool)
data = pd.concat([data,edummy],axis=1).drop(['etitle'],axis=1)
rdummy = pd.get_dummies(data['region'],columns=['region']).astype(bool)
data = pd.concat([data,rdummy],axis=1).drop(['region'],axis=1)
data = data.dropna()

#Take log of voting 
data['votes'] = np.log(data['votes'])
data['votes'] = data['votes'].replace([np.inf, -np.inf], np.nan)
data = data.dropna()

#Split train and test 
feature = data.drop(['votes'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(feature, data['votes'], test_size=0.4)

#Random Forest Regressor model
rfr = RandomForestRegressor(n_estimators=100, max_features=5)
rfr.fit(X_train, y_train)

#Save model into pickle file
with open('model.p', 'wb') as f:
    pickle.dump(rfr, f)

print('Random Forest Regresssor')
print('Train score: ' + str(rfr.score(X_train, y_train)))
print('Test score: ' + str(rfr.score(X_test, y_test)))

y0 = rfr.predict(X_test)

#Produce Graphs
REPORT = '../report/'
plt.figure(1,figsize=(10,10))
count_region = df.groupby('region').count().iloc[:,0]
region_pie = plt.pie(count_region,labels=count_region.index,autopct='%.2f')
plt.legend(count_region.index,loc='best')
plt.title('Pie Chart of Distribution of Region')
plt.savefig(REPORT + 'Region distribution.png', dpi=200)
plt.close

plt.figure(2,figsize=(10,10))
count_etitle = df.groupby('etitle').count().iloc[:,0]
region_pie = plt.pie(count_etitle,labels=count_etitle.index,autopct='%.2f')
plt.legend(count_etitle.index,loc='best')
plt.title('Pie Chart of Distribution of Category')
plt.savefig(REPORT + 'Category distribution.png', dpi=200)
plt.close

plt.figure(3)
plt.title('Histogram Number of UpVotes')
plt.hist(df.loc[:, ['votes']].values, bins=20, log=True)
plt.savefig(REPORT + 'Number of UpVotes.png', dpi=200)
plt.close

plt.figure(4)
plt.title('Histogram Number of Views')
plt.hist(df.loc[:, ['viewed']].values, bins=20, log=True)
plt.savefig(REPORT + 'Number of Views.png', dpi=200)
plt.close

plt.figure(5)
plt.title('Histogram Number of Comments')
plt.hist(df.loc[:, ['n_comments']].values, bins=20, log=True)
plt.savefig(REPORT + 'Number of Comments.png', dpi=200)
plt.close

plt.figure(6)
plt.title('Average per Year')
plt.plot(df.loc[:,['viewed', 'year']].groupby('year').mean())
plt.plot(df.loc[:,['n_comments', 'year']].groupby('year').mean())
plt.plot(df.loc[:,['votes', 'year']].groupby('year').mean())
plt.plot(df.loc[:,['votes', 'year']].groupby('year').count())
plt.legend(['Views', 'Comments', 'UpVotes', 'Number of pictures'])
plt.yscale('log')
plt.savefig(REPORT + 'line graph per year.png', dpi=200)
plt.close

plt.figure(7)
plt.title('Histogram Predicted vs Real UpVotes')
plt.hist(y0, bins=20, log=True, alpha=0.5)
plt.hist(y_test, bins=20, log=True, alpha=0.5)
plt.legend(['Predicted', 'Real'])
plt.savefig(REPORT + 'UpVotes predicted-real.png', dpi=200)
plt.close



