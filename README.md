## The Relationship Between a Country's Corruption Levels and Football Performance

- See FinalProject File for full report. 
## Overview
This project investigates whether and how political corruption and institutional quality correlate with national football success. It situates the analysis within the concept of sportwashing, i.e, when governments with high corruption invest in sports to improve their image or distract from internal issues. 

## Research Question
To what extent does political corruption–defined as the abuse of power for private gain and measured by Corruption Perceptions Index and Press Freedom Index–impact the long-term performance of national football teams in international tournaments as measured by FIFA World Rankings progression over years? Furthermore, how does institutional strength correlate with sustained competitive success when accounting for economic factors like GDP per capita?

## Data overview
We all got our datasets from the sites below and made csv files from them. We opened them in google sheets to make them into shareable webs for reading with pandas. After reading them into jupyter, we cleaned they data by removing any columns with unecessary information and reshaping the data to be in a more readable format. We also reindexed it so that the countries are on the left hand side and the years describe the columns.

## Press Freedom Index
Dataset Name: Press Freedom Index
Link to the dataset: https://data360.worldbank.org/en/dataset/RWB_PFI
Number of observations: 135
Number of variables: 23
Descripton: The Corruption Perceptions Index (CPI) dataset ranks countries based on perceived levels of public sector corruption, using a score from 0 to 100 where higher scores indicate cleaner governance. Each entry includes a country name, score, rank, and score change from the previous year, all of which serve as proxies for transparency and institutional integrity. A time series component tracks score changes from 2012 to 2023, enabling analysis of trends across different political and economic contexts. The data is derived from at least three expert and businessperson surveys out of thirteen trusted sources, such as the World Bank. Preprocessing steps include converting scores and ranks to numeric types, standardizing country names, handling missing values, and structuring the time series for longitudinal analysis.

## Corruption Perception Index
Dataset Name: Corruption Perception Index
Link to the dataset: https://www.transparency.org/en/cpi/2024/index/dnk
Number of observations: 236
Number of variables: 23
Descripton: The Corruption Perceptions Index (CPI) dataset ranks countries based on perceived levels of public sector corruption, using a score from 0 to 100 where higher scores indicate cleaner governance. Each entry includes a country name, score, rank, and score change from the previous year, all of which serve as proxies for transparency and institutional integrity. A time series component tracks score changes from 2012 to 2023, enabling analysis of trends across different political and economic contexts. The data is derived from at least three expert and businessperson surveys out of thirteen trusted sources, such as the World Bank. Preprocessing steps include converting scores and ranks to numeric types, standardizing country names, handling missing values, and structuring the time series for longitudinal analysis.

## FIFA Men's Ranking
Dataset Name: FIFA Men's Ranking
Link to the dataset: https://www.kaggle.com/datasets/cashncarry/fifaworldranking/code
Number of observations: 198
Number of variables: 23
Descripton: The FIFA World Rankings dataset includes both men's and women's national football teams, ranked based on their total points calculated from recent match results, tournament performance, and strength of opponents. Each entry contains key variables such as rank, team name, and total points, which serve as proxies for competitive strength and consistency. The men's rankings are updated as of April 3, 2025, and the women's as of March 6, 2025, offering a snapshot of current international standings. Cleaning the data requires dropping the statistics we are not analyzing and only including men's rankings, and reshaping the data to make it a cleaner table. This dataset can be valuable for performance modeling, regional comparisons, or assessing the impact of global tournaments on rankings. Cleaning the data requires dropping the statistics we are not analyzing and only including men's rankings.

## Gross Domestic Product (GDP) per capita
Dataset Name: Per capita GDP at current prices - US dollars
Link to the dataset: https://data.un.org/Data.aspx?d=SNAAMA&f=grID:101;currID:USD;pcFlag:1&c=2,3,5,6&s=_crEngNameOrderBy:asc,yr:desc&v=10
Number of observations: 208
Number of variables: 23
Descripton: The World Bank GDP per capita dataset includes both a global time series from 1960 to 2023 and the most recent GDP per capita values for individual countries. The key variables are year, country name, and GDP per capita, expressed in constant 2015 US dollars to account for inflation and enable accurate comparisons over time. These metrics serve as proxies for average individual economic output and overall development, providing insight into both global growth trends and national income disparities. Preprocessing would involve converting values to consistent numeric types, standardizing country names, handling missing data, and merging with population or regional indicators if needed. Together, these datasets offer powerful tools for analyzing long-term economic progress and cross-country differences in living standards.

Setup
# import packages
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from IPython.display import Image
import statsmodels as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
from scipy.stats import ttest_ind, chisquare, normaltest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
```

## Analysis of Averages of Data Across Years 2012-2023
To explore potential relationships between our variables we used a scatter matrix and heatmap based on the average values from 2012 to 2023 for each variable: corruption score, press freedom, FIFA ranking, and GDP across countries.
```
#Creating Scatter Matrix
fig = pd.plotting.scatter_matrix(df_avg[['Average PFI','Average CPI','Average FIFA', 'Average GDP']], figsize=(12,5))
```
<img width="1000" height="455" alt="image" src="https://github.com/user-attachments/assets/363b6fe7-f3f2-468f-b4ca-549cc4ff995d" />

PFI vs CPI:
In terms of the variables that describe corruption, PFI and CPI there is a strong negative correlation that shows that countries with less corruption tend to have higher press freedom. This is conclusive of our assumption that less press freedom and lower CPI score contribute to our definition of corruption.

GDP vs CPI:
There is a strong positive correlation between GDP and CPI. This supports the general description that less corrupt countries tend to have higher GDP’s therefore more stable economies.

FIFA vs PFI/CPI:
The scatter plot here displays a weaker linear relationship. There is a slight pattern suggesting that countries with more press freedom (lower PFI) and have less corruption (higher CPI) may perform slightly better in FIFA rankings.

FIFA vs GDP:
The plot shows that no clear linear association exists. However there is some clustering that points to a potential relationship where countries with high GDP tend to cluster around lower FIFA ranking (better performance). This relationship we will explore more in our cluster map to analyze the significance of these clusters that appear in our scatter matrix.


## Linear Regression Analysis
Analysis Comparing Average FIFA and CPI
The p-value from this receipt comparing average FIFA rankings and CPI scores is so small, 7.745325e-08, that it shows extreme significance of the relation between the two variables.
renamed_avg = df_avg.rename(columns={'Average FIFA' : 'fifa', 'Average CPI' : 'cpi', 'Average PFI' : 'pfi', 'Average GDP' : 'gdp'})

```
#MODEL 1: fifa ~ cpi
mod_1 = smf.ols('fifa ~ cpi', data=renamed_avg)
res_1 = mod_1.fit()
print(res_1.summary())
print(res_1.pvalues)
```

<img width="470" height="365" alt="Screenshot 2025-10-14 at 10 23 06 AM" src="https://github.com/user-attachments/assets/05051ea5-7000-4ab7-bd1c-f100cfb94bdb" />


Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Intercept    9.601188e-23
cpi          7.745325e-08
dtype: float64

## Analysis Comparing Average FIFA and CPI Controlling for PFI
When we made the same comparison controlling for PFI ranking, the p-vale became 0.099, making a statistically significant relationship between CPI and FIFA disappear. However, this can be justified since we know that PFI and CPI are multicollinear as seen in the scatter matrix and correlation heat map above.
```
#MODEL 2: fifa ~ cpi + pfi
mod_2 = smf.ols('fifa ~ cpi + pfi', data=renamed_avg)
res_2 = mod_2.fit()
print(res_2.summary())
```
<img width="464" height="375" alt="Screenshot 2025-10-14 at 10 23 16 AM" src="https://github.com/user-attachments/assets/b5a3cc3a-c2a2-49e3-8679-b502a08509f8" />

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

## Analysis Comparing Average FIFA and CPI Controlling for GDP
When controlling for GDP, we can see from GDP's p-value that it is not a predictor of a countries FIFA ranking. However, when looking at the CPI p-value controlled for GDP, its shows signifigance, supporting the sportwashing hypothesis. The economic state of a country does not affect a countries FIFA standing, however CPI does.
```
#MODEL 3: fifa ~ cpi + gdp
mod_3 = smf.ols('fifa ~ cpi + gdp', data=renamed_avg)
res_3 = mod_3.fit()
print(res_3.summary())
```
<img width="477" height="392" alt="Screenshot 2025-10-14 at 10 23 36 AM" src="https://github.com/user-attachments/assets/71c92cc4-c8ca-47cc-9bc9-62abbf5466f4" />


Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
To further investigate model 3 and analyze the affect of GDP on CPI, we check the averages of the top 5 and bottom 5 countries in GDP rankings.
```
sorted_countries = gdp_avg.sort_values('Average GDP', ascending=False)

#find average cpi of 25 countries with best gdp
selected_countries = sorted_countries.head(25).index.tolist()
sorted_gdp = df_avg[df_avg.index.isin(selected_countries)]
avg_top_cpi = sorted_gdp['Average CPI'].mean()

#find average cpi of 25 countries with worst gdp
selected_countries = sorted_countries.tail(25).index.tolist()
sorted_gdp = df_avg[df_avg.index.isin(selected_countries)]
avg_bottom_cpi = sorted_gdp['Average CPI'].mean()

print('Top 25 Countries in GDP Scores Average CPI: ' + str(avg_top_cpi))
print('Bottom 25 Countries in GDP Scores Average CPI: ' + str(avg_bottom_cpi))
```
Top 25 Countries in GDP Scores Average CPI: 68.30666666666667
Bottom 25 Countries in GDP Scores Average CPI: 30.04333333333333
The purpose of this was to further analyze and check that GDP does not influence the FIFA ranking of a country. We wanted to confirm that GDP was not a confounding variable in the relationship of corruption and FIFA ranking. The linear regression analysis verifies that GDP is not a confounding variable. However analyzing the top and bottom 25 countries in GDP and calculating their average CPI scores, we can identify that gdp has some sort of role in the relationship but requires deeper analysis.
