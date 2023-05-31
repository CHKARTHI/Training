######## Panndas

import pandas as pd
#data frame

#case 1
df=pd.read_csv("E:\\nyc_weather.csv")

#case 2
df=pd.read_csv("E:/nyc_weather.csv")
df
df.shape
df.head()
list(df)
df.columns
df

#case 3

df=pd.read_csv("nyc_weather.csv")
df
df.    
#accessing the columns

df[df['Temperature'],df['EST']
df['Temperature'].describe()
df['Temperature'].var()

df["Events"].value_counts()
df.isnull()
#true means blanks

df.sum()
df.isnull().sum()
df.isna().value_counts()
df[['Temperature','EST']]
df.columns[[0,3,5]]
df[df.columns[[0,3,5]]]
df.drop(df.columns[[0,3,5]], axis=1)  #dropping multiple columns


df
df.drop(df.columns[[0,3,5]], axis=1,inplace=True)
df
df=pd.read_csv("nyc_weather.csv")
df.drop([0,1,2])
df
df["Events"].value_counts()
df["Events"] == "Rain"
#filtering the data

df[df["Events"] == "Rain"]
df[(df["Events"] == "Rain") & (df["Temperature"] >45)]
df[["EST","Temperature"]][(df["Events"] == "Rain") & (df["Temperature"] >45)]
list(df)
d1={"Sea Level PressureIn":"SLP","DewPoint":"DP"}
df.rename(columns=d1, inplace=True)
list(df)

df.dropna(axis=0) #all na rows will be dropped
df.dropna(axis=1) #all na columns will be cleared

df[df["Events"]=="Rain"]
df.isnull()
df

data={"state":"a","dist":"b"}
type(data)
#df = pd.DataFrame(data)


df=pd.read_csv("nyc_weather.csv")

# Create dataframe
data = {'states':['Karnataka','Karnataka','Telangana','Telangana','Maharastra','Maharastra'],
       'Date':['01-05-2021','02-05-2021','01-05-2021','02-05-2021','01-05-2021','02-05-2021'],
       'Cases':[200,220,340,333,400,420],
       'Recovered':[40,50,150,130,270,300]}
type(data)
df = pd.DataFrame(data)
df

#Groupby
dlg = df.groupby("states")
dlg.sum()
dlg.mean()
dlg.describe().transpose()

India = {'states':['Karnataka','Karnataka','Telangana','Telangana','Maharastra','Maharastra'],
       'Date':['01-05-2021','02-05-2021','01-05-2021','02-05-2021','01-05-2021','02-05-2021'],
       'Cases':[200,220,340,333,400,420],
       'Recovered':[40,50,150,130,270,300]}

America={'states':['New','Tex','Tex','Na','Hos','Cr'],
       'Date':['01-05-2021','02-05-2021','01-05-2021','02-05-2021','01-05-2021','02-05-2021'],
       'Cases':[200,220,340,333,400,420],
       'Recovered':[40,50,150,130,270,300]}

india_weather = pd.DataFrame({
    "city": ["mumbai","delhi","banglore"],
    "temperature": [32,45,30],
    "humidity": [80, 60, 78]
})
india_weather

us_weather = pd.DataFrame({
    "city": ["new york","chicago","orlando"],
    "temperature": [21,14,35],
    "humidity": [68, 65, 75]
})
us_weather

df=pd.concat([india_weather,us_weather])
df
df=pd.concat([india_weather,us_weather],ignore_index=True)
df
dlg=df.groupby("city")
df

temperature_df = pd.DataFrame({
    "city": ["mumbai","delhi","banglore"],
    "temperature": [32,45,30],
}, index=[0,1,2])
temperature_df


windspeed_df = pd.DataFrame({
    "events": ["hotday","rainy","snowfall"],
    "windspeed": [7,25,10],
}, index=[0,1,2])
windspeed_df

df=pd.concat([temperature_df,windspeed_df],axis=1)
df


###################################
df1 = pd.DataFrame({
    "city": ["new york","chicago","orlando"],
    "temperature": [21,14,35],
})
df1

df2 = pd.DataFrame({
    "city": ["chicago","new york","orlando"],
    "humidity": [65,68,75],
})
df2
df=pd.concat([df1,df2])
df
df3=pd.merge(df1,df2, on="city")
df3
#df3.to_csv(location and name)

df1 = pd.DataFrame({
    "city": ["new york","chicago","orlando", "baltimore"],
    "temperature": [21,14,35, 38],
})
df1

df2 = pd.DataFrame({
    "city": ["chicago","new york","san diego"],
    "humidity": [65,68,71],
})
df2

#intersection
df3=pd.merge(df1,df2,on="city",how="inner")
df3
#union
df3=pd.merge(df1,df2,on="city",how="outer")

df3




















































