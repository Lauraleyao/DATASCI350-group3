#!/usr/bin/env python
# coding: utf-8

# In[1]:


import wbgapi as wb
import pandas as pd


# In[2]:


#Variables of interest
INDICATORS = {
    "SP.DYN.LE00.IN": "life_expectancy_birth",
    "SH.H2O.BASW.ZS": "basic_water_access_pct",
    "SH.STA.BASS.ZS": "basic_sanitation_access_pct",
    "SE.PRM.ENRR": "primary_school_enrollment",
    "NY.GDP.PCAP.KD": "gdp_per_capita_constant_usd"
}


# In[3]:


# Choose our regions
ssa = set(wb.region.members("SSF"))  # Sub-Saharan Africa
sa = set(wb.region.members("SAS"))   # South Asia
filtered_countries = list(ssa | sa)  # Union of sets


# In[4]:


#Create df of countries and years
df = wb.data.DataFrame(
    list(INDICATORS.keys()),       # indicator codes
    economy=filtered_countries,    
    time=range(2000, 2021),
    labels=False,                  # keep codes to rename to descriptive names later
    numericTimeKeys=True           # years are integers
)


# In[5]:


df.reset_index(inplace=True)


# In[6]:


# Melt the df so it's one row per country-year-indicator
df_long = df.melt(
    id_vars=['economy', 'series'],
    value_vars=list(range(2000, 2021)),
    var_name='Year',
    value_name='Value'
)


# In[7]:


# Pivot so each indicator is a separate column
df_final = df_long.pivot_table(
    index=['economy', 'Year'],
    columns='series',
    values='Value'
).reset_index()


# In[8]:


# Rename indicators to be their descriptive names
df_final.rename(columns=INDICATORS, inplace=True)
df_final.rename(columns={'economy': 'Country'}, inplace=True)

#Add region label
region_labels = {}

for c in ssa:
    region_labels[c] = "Sub-Saharan Africa"

for c in sa:
    region_labels[c] = "South Asia"

# Add the region to df_final
df_final["Region"] = df_final["Country"].map(region_labels)

df_final


# In[9]:


#Save as CSV
df_final.to_csv("qtm350_final.csv", index=False)


# In[10]:


#Save as db
import sqlite3

# Create a connection to a new database file
conn = sqlite3.connect("qtm350_final.db")

# Write the DataFrame to a table called "indicators"
df_final.to_sql("indicators", conn, if_exists="replace", index=False)

# Close the connection
conn.close()

