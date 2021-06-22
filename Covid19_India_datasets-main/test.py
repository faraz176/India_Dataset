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

#Make a model for the entire of the dataset done
#Which countries have the highest average case count
#Which countries have the highest deaths
#Which countries have the highest death/per confirmed 
#Which countires are recovering the fastest 



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


# f_to_start.to_csv("combined.csv")

#Reading the data in 
location = "C:/Users/flyhi/OneDrive/Desktop/india/Covid19_India_datasets-main/combined.csv"
new = Path(location)
df = pd.read_csv(location)
#print(len(df['State'].unique()))

#####Initial Analysis############

#Dropping bad columns
df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
df['Time'] = pd.to_datetime(df['Time'])

# print(df.head())


#The death rate is linear 

plt.scatter(x=df['Time'], y=df['Deaths'])
plt.show()

#Highest average case count 

fig, ax = plt.subplots() 



missing_set = df[df['New Cases'].notnull()]
cases = list(missing_set['New Cases'])
print(missing_set['State'].unique())

list_of_cases = []
for i in missing_set['State'].unique():
    new_state = missing_set[missing_set['State'] == i]
    series_cases = list(new_state['New Cases'])
    count = 0 
    for i in range(len(series_cases)):
        try:
            
            count = count + series_cases[i+1] - series_cases[i]
            #rint(count)
        except IndexError:
            print('completed')

    #final_count = count / new_state.shape[0]
    final_count = count
    list_of_cases.append(final_count)


print(len(list_of_cases))

print(list_of_cases)
# print(list_of_cases[0])
# print(len(list_of_cases))
    
          


#         try:
#             #if series_cases[i+1] != None:
#             if i == 0:
#                 print(list(new_state['State'])[0])
#             count = series_cases[i+1] 
#             print(count)
#         except IndexError:
#             print('completed')


    
#     list_of_cases.append(count)

# print(list_of_cases)



# count = 0 
# for i in len(cases):
#     count = cases 
# vz = missing_set.groupby('State')['New Cases'].agg(['sum', 'count'])
# vz['average'] = vz['sum'] / vz['count']




# states = list(vz.index)
# print(vz)
# # Example data
#people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
#y_pos = np.arange(len(list(missing_set['State'].unique())))
#print(y_pos)
#Preformance is the values of the actual thing
#performance =  final_count
#Idk error to account for (its that black bar on top of the)
#error = np.random.rand(len(people)) 

# ax.barh(y_pos, performance, align='center')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(list(missing_set['State'].unique()))
# plt.ylabel(list(missing_set['State'].unique()))
# #plt.set_yticklabels(list(missing_set['State'].unique()))
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Performance')
# ax.set_title('How fast do you want to go today?')

# plt.show()








#Modeling and predicting

#Selecting and processing features
plt.figure(figsize=(15,8))
sns.heatmap(missing_set.corr(), annot=True, linewidths=1)
plt.show()

forecast_out = 10
x = np.array(missing_set[['Confirmed', 'Recovered', 'Active', 'New Cases']])
x = preprocessing.scale(x)
x= x[:-forecast_out]
x_predict = x[-forecast_out:]
y = np.array(missing_set['Deaths'])
y = y[:-forecast_out]


#Trainig
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)


#Our linear classifier
clf = LinearRegression()
clf.fit(x_train, y_train)

#Accuracy
accuracy = clf.score(x_test, y_test)
#print(accuracy)


#Predicting 
forecast_set = clf.predict(x_predict)

#Using data from:

#https://www.nytimes.com/interactive/2021/05/25/world/asia/india-covid-death-estimates.html

#The "likely scenario" indicated about 5.3x the number of reported deaths, our model predicted lower then the actual reported deaths so we have to multipy the number of deaths 
#from our predictive model by (reported_deaths/predicted model deaths) + 5.3x

# print("")
# print("")

print(forecast_set)






