from itertools import count
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from pandas._libs import missing 
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression 
import numpy as np 
import seaborn as sns

#Make a model for the entire of the dataset (done)
#Which state have the highest average case count (done)
#Which state have the highest deaths (done)
#Show that deaths are linearly increasing (done)
#How many recovered per confirmed cases
#Deaths per confirmed cases by state


#Concatenated from directories into one big file 

# name = "C:/Users/flyhi/OneDrive/Desktop/india/Covid19_India_datasets-main/data-Andaman and Nicobar Islands.csv"
# csv_file = Path(name)

# f_to_start = pd.read_csv(csv_file)
# #print(os.listdir())

# #directory = "C:\Users"+ "'\'" +  "flyhi" + "'\'" + "OneDrive" + "'\'" + 'Desktop' + "'\'" + 'india' + "'\'" + "Covid19_India_datasets-main"
# directory = "C:/Users/flyhi/OneDrive/Desktop/india/Covid19_India_datasets-main"
# thing = "C:/Users/flyhi/OneDrive/Desktop/india/Covid19_India_datasets-main/"
# directory = Path(directory)
# #thing = Path(thing)
# for filename in os.listdir(directory):
#     view = thing + filename
#     new = Path(view)
#     isDirectory = os.path.isdir(new)
#     if isDirectory == True:
#         os.chdir(new)
#         csv_file_name =os.listdir()[0]
#         df_to_concat = pd.read_csv(csv_file_name)
#         f_to_start = f_to_start.append(df_to_concat)
#         os.chdir(directory)




#Reading the data in 
location = "C:/Users/flyhi/OneDrive/Desktop/india/Covid19_India_datasets-main/combined.csv"
new = Path(location)
df = pd.read_csv(location)


#Data Preprocessing

#Dropping bad columns
df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
df['Time'] = pd.to_datetime(df['Time'])


#Exploratory data analysis


#The death rate is linear 

# plt.scatter(x=df['Time'], y=df['Deaths'])
# plt.show()

#Average case count by country

# fig, ax = plt.subplots() 

missing_set = df[df['New Cases'].notnull()]
# vz = missing_set.groupby('State')['New Cases'].agg(['sum', 'count'])
# vz['average'] = vz['sum'] / vz['count']
# print(type(vz))
# print(vz)



# #Setting Parameters
# states = list(vz.index)

# # Example data
# y_pos = np.arange(len(list(missing_set['State'].unique())))


# #Preformance is the values of the actual thing
# performance =  vz['average']

# #Idk error to account for (its that black bar on top of the)
# error = np.random.rand(len(y_pos)) 

# #Graphing
# ax.barh(y_pos, performance, align='center')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(list(missing_set['State'].unique()))
# plt.ylabel(list(missing_set['State'].unique()))
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Average Case Count')
# ax.set_title('Average Case Count by State')

# plt.show()


# #Deaths per confirmed case 

# fig, ax = plt.subplots() 
# last_row = pd.DataFrame()
# for i in missing_set['State'].unique():
#     new_df = missing_set[missing_set['State'] == i]
#     last_row=last_row.append(new_df.iloc[[0, -1]], ignore_index=True)
    



# #Setting Parameters
# states = list(missing_set['State'].unique())

# # setting the y-pos
# y_pos = np.arange(len(list(missing_set['State'].unique())))

# #Preformance is the values of the actual thing
# new_df = last_row.iloc[lambda x: x.index % 2 == 1]

# new_df['deaths_to_confirmed'] = new_df['Deaths'] / new_df['Confirmed']

# performance = new_df['deaths_to_confirmed']




# #Idk error to account for (its that black bar on top of the)
# error = np.random.rand(len(y_pos)) 


# #Graphing
# ax.barh(y_pos, performance, align='center')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(list(missing_set['State'].unique()))
# plt.ylabel(list(missing_set['State'].unique()))
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Deaths per confirmed by each state')
# ax.set_title('Deaths per confirmed')

# plt.show()





#Recoveries per confirmed case 
fig, ax = plt.subplots() 
last_row = pd.DataFrame()
for i in missing_set['State'].unique():
    new_df = missing_set[missing_set['State'] == i]
    last_row=last_row.append(new_df.iloc[[0, -1]], ignore_index=True)
    



#Setting Parameters
states = list(missing_set['State'].unique())

# setting the y-pos
y_pos = np.arange(len(list(missing_set['State'].unique())))

#Preformance is the values of the actual thing
new_df = last_row.iloc[lambda x: x.index % 2 == 1]

new_df['recoveries_to_confirmed'] = new_df['Recovered'] / new_df['Confirmed']

performance = new_df['recoveries_to_confirmed']




#Idk error to account for (its that black bar on top of the)
error = np.random.rand(len(y_pos)) 


#Graphing
ax.barh(y_pos, performance, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(list(missing_set['State'].unique()))
plt.ylabel(list(missing_set['State'].unique()))
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Recoveries per Confirmed Case by each state')
ax.set_title('Recoveries per Confirmed')

plt.show()




#Modeling and predicting

#Selecting and processing features
# plt.figure(figsize=(15,8))
# sns.heatmap(missing_set.corr(), annot=True, linewidths=1)
# plt.show()

# forecast_out = 10
# x = np.array(missing_set[['Confirmed', 'Recovered', 'Active', 'New Cases']])
# x = preprocessing.scale(x)
# x= x[:-forecast_out]
# x_predict = x[-forecast_out:]
# y = np.array(missing_set['Deaths'])
# y = y[:-forecast_out]


# #Trainig
# x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)


# #Our linear classifier
# clf = LinearRegression()
# clf.fit(x_train, y_train)

# #Accuracy
# accuracy = clf.score(x_test, y_test)
# print(accuracy)


# #Predicting 
# print(x_predict)
# forecast_set = clf.predict(x_predict)
# # print(x_predict)

# #Using data from:

# #https://www.nytimes.com/interactive/2021/05/25/world/asia/india-covid-death-estimates.html

# #The "likely scenario" indicated about 5.3x the number of reported deaths, our model predicted lower then the actual reported deaths so we have to multipy the number of deaths 
# #from our predictive model by (reported_deaths/predicted model deaths) + 5.3x

# # print("")
# # print("")

# print(forecast_set)






