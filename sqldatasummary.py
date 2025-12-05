#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect("data/qtm350_final.db") 
cursor = conn.cursor()
print("Database connected.")

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())


# In[38]:


#Table info
pd.read_sql_query("PRAGMA table_info(indicators);", conn)


# In[39]:


#Average, min, and max for each variable from 2000 to 2020
avg_min_max_overall= """
SELECT
    AVG(life_expectancy_birth) AS avg_life_expectancy,
    MIN(life_expectancy_birth) AS min_life_expectancy,
    MAX(life_expectancy_birth) AS max_life_expectancy,
    AVG(basic_water_access_pct) AS avg_water_access,
    MIN(basic_water_access_pct) AS min_water_access,
    MAX(basic_water_access_pct) AS max_water_access,
    AVG(basic_sanitation_access_pct) AS avg_sanitation_access,
    MIN(basic_sanitation_access_pct) AS min_sanitation_access,
    MAX(basic_sanitation_access_pct) AS max_sanitation_access,
    AVG(primary_school_enrollment) AS avg_primary_enrollment,
    MIN(primary_school_enrollment) AS min_primary_enrollment,
    MAX(primary_school_enrollment) AS max_primary_enrollment,
    AVG(gdp_per_capita_constant_usd) AS avg_gdp_per_capita,
    MIN(gdp_per_capita_constant_usd) AS min_gdp_per_capita,
    MAX(gdp_per_capita_constant_usd) AS max_gdp_per_capita,
    COUNT(*) AS total_rows
FROM indicators;
"""

pd.read_sql_query(avg_min_max_overall, conn)


# In[40]:


summary_by_country = """
SELECT
    Country,
    AVG(life_expectancy_birth) AS avg_life_expectancy,
    MIN(life_expectancy_birth) AS min_life_expectancy,
    MAX(life_expectancy_birth) AS max_life_expectancy,
    
    AVG(basic_water_access_pct) AS avg_water_access,
    MIN(basic_water_access_pct) AS min_water_access,
    MAX(basic_water_access_pct) AS max_water_access,
    
    AVG(basic_sanitation_access_pct) AS avg_sanitation_access,
    MIN(basic_sanitation_access_pct) AS min_sanitation_access,
    MAX(basic_sanitation_access_pct) AS max_sanitation_access,
    
    AVG(primary_school_enrollment) AS avg_primary_enrollment,
    MIN(primary_school_enrollment) AS min_primary_enrollment,
    MAX(primary_school_enrollment) AS max_primary_enrollment,
    
    AVG(gdp_per_capita_constant_usd) AS avg_gdp_per_capita,
    MIN(gdp_per_capita_constant_usd) AS min_gdp_per_capita,
    MAX(gdp_per_capita_constant_usd) AS max_gdp_per_capita,
    
    COUNT(*) AS total_rows
FROM indicators
GROUP BY Country
ORDER BY Country;
"""
pd.read_sql_query(summary_by_country, conn)


# In[41]:


summary_by_region = """
SELECT
    Region,
    AVG(life_expectancy_birth) AS avg_life_expectancy,
    MIN(life_expectancy_birth) AS min_life_expectancy,
    MAX(life_expectancy_birth) AS max_life_expectancy,
    
    AVG(basic_water_access_pct) AS avg_water_access,
    MIN(basic_water_access_pct) AS min_water_access,
    MAX(basic_water_access_pct) AS max_water_access,
    
    AVG(basic_sanitation_access_pct) AS avg_sanitation_access,
    MIN(basic_sanitation_access_pct) AS min_sanitation_access,
    MAX(basic_sanitation_access_pct) AS max_sanitation_access,
    
    AVG(primary_school_enrollment) AS avg_primary_enrollment,
    MIN(primary_school_enrollment) AS min_primary_enrollment,
    MAX(primary_school_enrollment) AS max_primary_enrollment,
    
    AVG(gdp_per_capita_constant_usd) AS avg_gdp_per_capita,
    MIN(gdp_per_capita_constant_usd) AS min_gdp_per_capita,
    MAX(gdp_per_capita_constant_usd) AS max_gdp_per_capita,
    
    COUNT(*) AS total_rows
FROM indicators
GROUP BY Region
ORDER BY Region;
"""
pd.read_sql_query(summary_by_region, conn)


# In[42]:


conn.close()

