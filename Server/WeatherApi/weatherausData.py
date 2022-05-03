#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import request
import pandas as pd
from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt
import pickle


# In[4]:
def funcOne():

    df = pd.read_csv('./WeatherApi/weatherAUS.csv')
    df.head()


    # In[5]:


    df.Location.unique() #List of locations


    # In[6]:


    df.columns #Labels of each columns


    # In[7]:


    df.dtypes


    # In[8]:


    melb = df[df["Location"]=="Melbourne"] #Selecting the location
    melb["Date"] = pd.to_datetime(melb['Date']) #converting the date column into a datetime type
    #melb.head()


    # In[9]:


    #melb.dtypes #The date column is in datetime type


    # In[10]:


    plt.plot(melb["Date"], melb["Temp3pm"]) #looking over data
    #plt.show()


    # In[11]:


    melb["Year"] = melb["Date"].apply(lambda x: x.year) #looping through the years and extracting the year
    melb = melb[melb["Year"] <= 2015] #cutting out the missing data
    plt.plot(melb["Date"], melb["Temp3pm"])
    #plt.show()


    # In[12]:


    #3melb.tail()


    # In[13]:


    data = melb[["Date", "Temp3pm"]] #Neural can only be done with only 2 columns, so grabbing data that important
    data.dropna(inplace = True) #this will drop all of the NA
    data.columns = ["ds", "y"] 
    #data.head()


    # Training the Model

    # In[14]:


    m = NeuralProphet() #importing
    m.fit(data, freq="D", epochs=5) #1st parameter is grabbing the new data, 2nd the data is in daily frequency, that is why its in "D", 3rd the amount of times being trained  


    # Forecasting

    # In[15]:


    future = m.make_future_dataframe(data, periods=1200) #this will output 1200 dates into the future
    forecast = m.predict(future)
    #forecast.head()


    # In[16]:


    forecast.tail()


    # In[17]:


    plot = m.plot(forecast)


    # In[18]:


    plot2 = m.plot_components(forecast)


    # In[19]:


    with open ("forecast_model.pkl", "wb") as f:
        pickle.dump(m,f)


    # In[20]:


    m


    # In[21]:


    with open ("forecast_model.pkl", "rb") as f:
        m = pickle.load(f)


    # In[22]:


    m

    # In[23]:


    future = m.make_future_dataframe(data, periods=5) #this will output 1200 dates into the future
    forecast = m.predict(future)
    forecast.head()

    return forecast.to_json(date_format='iso')

    # In[24]:


    #plot = m.plot(forecast)

