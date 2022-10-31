#!/usr/bin/env python
# coding: utf-8

# # *Covid-19 Data Analysis in India*

# In[1]:


# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime


# In[2]:


# Read .csv files:

covid_df = pd.read_csv("Desktop/Covid_19_Project/covid_19_india.csv")
vaccine_df = pd.read_csv("Desktop/Covid_19_Project/covid_vaccine_statewise.csv")


# In[3]:


#Display table to view fields

covid_df.head()


# In[4]:


# Fetch info of the table

covid_df.info()


# In[5]:


# Decription of the table

covid_df.describe()


# In[6]:


# Drop unwanted columns from the table

covid_df.drop(["Sno","Time","ConfirmedIndianNational","ConfirmedForeignNational"], inplace = True, axis = 1)

covid_df.head()


# In[7]:


# Change Date format into proper format

covid_df['Date']=pd.to_datetime(covid_df['Date'], format = '%Y-%m-%d')

covid_df.head()


# In[8]:


# To find Active Cases

covid_df['Active_Cases'] = covid_df['Confirmed'] - (covid_df['Cured'] + covid_df['Deaths'])
covid_df.tail()


# ## ***State-wise Pivot Table***

# In[9]:


statewise = pd.pivot_table(covid_df, values = ['Confirmed', 'Deaths', 'Cured'], index = 'State/UnionTerritory', aggfunc = max)

statewise['Recovery Rate'] = statewise['Cured']*100 / statewise['Confirmed']

statewise['Mortality Rate'] = statewise['Deaths']*100 / statewise['Confirmed']

statewise = statewise.sort_values(by = 'Confirmed', ascending = False)

statewise.style.background_gradient(cmap = 'Accent')


# In[10]:


# Top 10 active cases

top_10_active_cases = covid_df.groupby(by = 'State/UnionTerritory').max()[['Active_Cases', 'Date']].sort_values(by = ['Active_Cases'], ascending = False).reset_index()

fig = plt.figure(figsize=(20,10))
plt.title('Top 10 states with most active cases in India', size = 25)
ax = sns.barplot(data = top_10_active_cases.iloc[:10], y = 'Active_Cases', x = 'State/UnionTerritory', linewidth = 2, edgecolor = 'red')

plt.xlabel('States')
plt.ylabel('Total Active Cases')
plt.show()


# In[11]:


# Top states with highest deaths

top_10_deaths = covid_df.groupby(by = 'State/UnionTerritory').max()[['Deaths', 'Date']].sort_values(by = ['Deaths'], ascending = False).reset_index()

fig = plt.figure(figsize=(20,10))

plt.title('Top 10 states with most deaths in India', size = 25)

ax = sns.barplot(data = top_10_deaths.iloc[:12], y = 'Deaths', x = 'State/UnionTerritory', linewidth = 2, edgecolor = 'black')

plt.xlabel('States')
plt.ylabel('Total Deaths')
plt.show()


# In[12]:


# Trend of Cases

fig = plt.figure(figsize = (20,10))

ax = sns.lineplot(data = covid_df[covid_df['State/UnionTerritory'].isin(['Maharashtra','Karnataka','Kerala','Tamil Nadu','Uttar Pradesh'])], x = 'Date', y = 'Active_Cases', hue = 'State/UnionTerritory')

ax .set_title('Top 5 Affected States in India', size = 25)


# In[13]:


# Display Vaccine Table
vaccine_df.head()


# In[14]:


# Rename Columns

vaccine_df.rename(columns = {'Updates On' : 'Vaccine_Date'}, inplace = True)
vaccine_df.head()


# In[15]:


# Fetching info of the table

vaccine_df.info()


# In[16]:


# To find the sum of all missing values for each column

vaccine_df.isnull().sum()


# In[17]:


# Drop unwanted columns from the table

vaccination = vaccine_df.drop(columns = ['Sputnik V (Doses Administered)', 'AEFI', '18-44 Years (Doses Administered)','45-60 Years (Doses Administered)','60+ Years (Doses Administered)'],axis =1)
vaccination.head()


# In[18]:


# Comparison of male vs. female vaccination


male = vaccination['Male(Individuals Vaccinated)'].sum()
female = vaccination['Female(Individuals Vaccinated)'].sum()
px.pie(names=['Male','Female'], values = [male,female], title = 'Male and Female Vaccination')


# In[19]:


# Remove rows where state = India

vaccine = vaccine_df[vaccine_df.State!='India']


# In[20]:


vaccine.head()


# In[21]:


# Rename last column to a shorter name

vaccine.rename(columns = {'Total Individuals Vaccinated' : 'Total'}, inplace = True)
vaccine


# ## ***Top 5 Vaccinated States***

# In[22]:


# Most Vaccinated State

max_vac = vaccine.groupby('State')['Total'].sum().to_frame('Total')
max_vac = max_vac.sort_values('Total', ascending = False)[:5]
max_vac


# In[23]:


fig = plt.figure(figsize=(20,10))
plt.title('Top 5 Vaccinated States in India', size = 25)
ax = sns.barplot(data = max_vac.iloc[:10], y = max_vac.Total, x = max_vac.index, linewidth = 2, edgecolor = 'black')

plt.xlabel('States')
plt.ylabel('Vaccination')
plt.show()


# ## ***Least 5 Vaccinated States***

# In[24]:


# Least Vaccinated States

least_vac = vaccine.groupby('State')['Total'].sum().to_frame('Total')
least_vac = least_vac.sort_values('Total', ascending = True)[:5]
least_vac


# In[25]:


fig = plt.figure(figsize=(20,10))
plt.title('Least 5 Vaccinated States in India', size = 25)
ax = sns.barplot(data = least_vac.iloc[:10], y = least_vac.Total, x = least_vac.index, linewidth = 2, edgecolor = 'black')

plt.xlabel('States')
plt.ylabel('Vaccination')
plt.show()

