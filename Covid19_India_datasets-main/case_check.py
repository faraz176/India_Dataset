import pandas as pd 
import numpy as np 
from pathlib import Path 



#Importing the file 
location = "C:/Users/flyhi/OneDrive/Desktop/india/Covid19_India_datasets-main/combined.csv"
new = Path(location)
df = pd.read_csv(location)



#Dropping col's
df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
df['Time'] = pd.to_datetime(df['Time'])


#Getting cases 
missing_set = df[df['New Cases'].notnull()]
cases = list(missing_set['New Cases'])
print(missing_set['State'].unique())

list_of_cases = []
for i in list(missing_set['State'].unique()):
    new_state = missing_set[missing_set['State'] == i]
    z = new_state[(new_state['New Cases'] == 1276.0)]
    print(z)
    
#     series_cases = list(new_state['New Cases'])
#     for z in range(len(series_cases)):
#         list_of_cases.append(series_cases[z])


# for i in list_of_cases:
#     if i == 1276.0:
#         print('yay')