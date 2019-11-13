#Task 1 -  Data Loading
import numpy as numpy
import pandas as pd 
import matplotlib.pyplot as plt 
data = pd.read_csv(path)
data['Rating'].hist()
plt.xlabel('old App Rating')
plt.ylabel('frequency')
plt.title('histogram of Rating column')
data= data[data['Rating'] <= 5]
data['Rating'].hist()
plt.xlabel('New App Rating')
plt.ylabel('frequency')
plt.title('histogram of Rating column')
plt.show()


#Task 2 - Null Value Treatment
# code starts here
import pandas as pd
total_null = data.isnull().sum()
print('total null values in each column:',total_null)
print('='*200)
percent_null = (total_null/data.isnull().count())
print('percentage of null values:',percent_null)
print('='*200)
missing_data = pd.concat([total_null,percent_null],keys=['Total','Percent'],axis=1)
print('Missing data',missing_data)
print('='*200)
data.dropna(inplace=True)
total_null_1= data.isnull().sum()
print('New total of null values:',total_null_1)
print('='*200)
percent_null_1= (total_null_1/data.isnull().count())
print('New percent ofnull values:',percent_null_1)
print('='*200)
missing_data_1=pd.concat([total_null_1,total_null_1],keys=['Total','Percent'],axis=1)
print('missing_data_1:',missing_data_1)

# code ends here

#Task 3 - Category vs Rating

#Code starts here
import seaborn as sns
g=sns.catplot(x='Category',y='Rating',kind='box',data=data,height=10)
g.set_xticklabels(rotation=90)
plt.title('Rating vs Category [BoxPlot]')



#Code ends here

#Task 4 - Installs vs Ratings
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
#Code starts here
A= data['Installs'].value_counts()
print(A)
print('='*200)
data['Installs']=data['Installs'].str.replace('+','')
data['Installs']=data['Installs'].str.replace(',','')
print(data['Installs'].head(5))
print('='*200)
data['Installs']=data['Installs'].astype(int)
le= LabelEncoder()
data['Installs']=le.fit_transform(data['Installs'])
sns.regplot(x='Installs',y='Rating',data=data,color='Teal')
plt.title('Rating vs Installs [RegPlot]')



#Code ends here


#Task 5 -  Price vs Ratings
#Code starts here
data['Price'].value_counts()
data['Price']=data['Price'].str.replace('$','')
data['Price']=data['Price'].astype(float)
sns.regplot(x='Price',y='Rating',data=data,color='Teal')
plt.xticks(rotation=90)
plt.title('Rating vs Price [RegPlot]')
#Code ends here

#Task 6 - Genre vs Rating

#Code starts here

#Finding the length of unique genres
print( len(data['Genres'].unique()) , "genres")

#Splitting the column to include only the first genre of each app
data['Genres'] = data['Genres'].str.split(';').str[0]

#Grouping Genres and Rating
gr_mean=data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean()

print(gr_mean.describe())

#Sorting the grouped dataframe by Rating
gr_mean=gr_mean.sort_values('Rating')

print(gr_mean.head(1))

print(gr_mean.tail(1))

#Code ends here

#Code starts here
print(data['Last Updated'].head())
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()
data['Last Updated Days']=max_date - data['Last Updated']
data['Last Updated Days'] = data['Last Updated Days'].dt.days
sns.regplot(x="Last Updated Days",y="Rating",data=data,color='Teal')
plt.xticks(rotation=90)
plt.title('Rating vs Last Updated [RegPlot]')

