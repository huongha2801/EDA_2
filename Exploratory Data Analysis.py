#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# #### Agenda
# 1. Data Preparation <br>
# Structure of the data <br>
# Missing Values <br>
# 2. Feature Engineering<br>
# Age of Customers<br>
# Income <br>
# Months Since Enrollment<br>
# Total spending<br>
# Number of Children<br>
# Education<br>
# Marital Status<br>
# 3. Exploratory Data Analysis - Including statistical tests (T-tests, ANOVA)<br>
# Average Spendings: Marital Status Wise<br>
# Education Level<br>
# Child Status<br>
# Average Spendings: Child Status Wise<br>
# Age Distribution of Customers<br>
# 4. Multivariate Data Analysis <br/>
# Spending vs Age <br>
# Customer seniority vs Spending <br>
# Income vs Spending <br>
# 5. Marketing campaigns effectiveness

# ### Setting up

# In[1]:


# Loading libraries
import pandas as pd 
import numpy as np 
import warnings
import scipy.stats

# Visuals
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Loading the dataset
file = "D:\Myy Documents\[DS] All about Python\marketing_campaign.xlsx"
df = pd.read_excel(file)
df.head()


# In this analysis, we'll touch on basic variables only. These are: Birth year, Marital status, Income, Kids in the home, Amount spent on each of the 
# <br> The dataset was downloaded from Kaggle <br>
# Link to the data: https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign <br>
# I will quote the description for each variable here for convinient reference: <br>
# AcceptedCmp1 - 1 if customer accepted the offer in the 1st campaign, 0 otherwise<br>
# AcceptedCmp2 - 1 if customer accepted the offer in the 2nd campaign, 0 otherwise<br>
# AcceptedCmp3 - 1 if customer accepted the offer in the 3rd campaign, 0 otherwise<br>
# AcceptedCmp4 - 1 if customer accepted the offer in the 4th campaign, 0 otherwise<br>
# AcceptedCmp5 - 1 if customer accepted the offer in the 5th campaign, 0 otherwise<br>
# Response (target) - 1 if customer accepted the offer in the last campaign, 0 otherwise<br>
# Complain - 1 if customer complained in the last 2 years<br>
# DtCustomer - date of customer’s enrolment with the company<br>
# Education - customer’s level of education<br>
# Marital - customer’s marital status<br>
# Kidhome - number of small children in customer’s household<br>
# Teenhome - number of teenagers in customer’s household<br>
# Income - customer’s yearly household income<br>
# MntFishProducts - amount spent on fish products in the last 2 years<br>
# MntMeatProducts - amount spent on meat products in the last 2 years<br>
# MntFruits - amount spent on fruits products in the last 2 years<br>
# MntSweetProducts - amount spent on sweet products in the last 2 years<br>
# MntWines - amount spent on wine products in the last 2 years<br>
# MntGoldProds - amount spent on gold products in the last 2 years<br>
# NumDealsPurchases - number of purchases made with discount<br>
# NumCatalogPurchases - number of purchases made using catalogue<br>
# NumStorePurchases - number of purchases made directly in stores<br>
# NumWebPurchases - number of purchases made through company’s web site<br>
# NumWebVisitsMonth - number of visits to company’s web site in the last month<br>
# Recency - number of days since the last purchase<br>

# ### 1. Data Prepapration

# #### Structure of the data

# In[3]:


df.shape


# In[4]:


df.dtypes


# Many columns are not in the correct dtypes, we will correct them later when needs to.

# In[5]:


df.describe().round(2)


# #### Missing Values

# In[6]:


df.isnull().sum()


# There are incredibly few missing values. Quite a complete dataset. However, we will also have to examine if there are any
# inappropriate dtpoints, such as outliers and error values.

# Also, since there are few missing values for income, we can simply drop these rows.

# In[7]:


df = df.dropna()


# ### 2. Feature Engineering

# There is a lot of information given in the dataset related to the customers. In some cases we can group related columns to create more useful variables. This would help to better explore the data and draw meaningful insights from it.

# #### Age

# Since most of the activities by customers were between 2012 and 2014, we would assume that the data was collected in January 2015 for the sake of simplicity.

# In[8]:


import datetime as dt
df['Age'] = 2015 - df['Year_Birth']
df['Age'].describe()


# The maximum age found is 122, which would not be logical. Since this indicates that there are outliers in the Age column, we will drop these outliers then run the 'describe' command again to check for appropriateness.

# In[9]:


df = df[df['Age']<100]


# In[10]:


# Next, we will plot the age distribution
# Since we're going to use a lot of distribution diagrams, consider creating a function for it
def hist_with_vline(df, column):
    """This function gets data and column name.
    Plots a histogram with 100 bins, draws a Vline of the column mean and median"""
   
    plt.figure(figsize=(12,6))
    _ = sns.histplot(df[column], bins= 100)
    plt.title('Histogram of ' + column + ' distribution')
    miny, y_lim = plt.ylim()
    plt.text(s = f"Mean  {column} : {df[column].mean():.2f}", x =df[column].mean() * 1.1,  y = y_lim * 0.95, color = 'r')
    _ =plt.axvline(df[column].mean(), color = 'r')
    _ = plt.axvline(df[column].median(), color = 'g')
    plt.text(s = f"Median {column} : {df[column].median():.2f}", x= df[column].median() * 1.1, y= y_lim * 0.90, color = 'g')


# In[11]:


hist_with_vline(df, 'Age')


# The distribution is relatively normal, with the mean and median age overlaps quite closely. <br/>
# The mean age is 47, which suggests that the customer base includes most middle-aged shoppers.

# #### Income

# In[12]:


df.Income.describe()


# Quick examination (through histogram) reveals that there is 1 outlier (Income = 666666). We will drop this value.

# In[13]:


df = df[df['Income']<600000]


# In[14]:


hist_with_vline(df, 'Income')


# Both Age and Income are quite normally distributed (with mean and median values somewhat overlaping).

# #### Months Since Enrollment

# In[15]:


# We will check whether or not our customers stayed with us for long

df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['Date_Collected'] = '01-01-2015'
df['Date_Collected'] = pd.to_datetime(df.Date_Collected)
df['Days_Enrolled'] = (df['Date_Collected'] - df['Dt_Customer']).dt.days
df['Months_Enrolled'] = round(df['Days_Enrolled']/30)


# In[16]:


hist_with_vline(df, 'Months_Enrolled')


# #### Total items bought

# In[17]:


# First, we will examine how many units of each item were sold
columns = ['MntWines', 'MntFruits', 'MntMeatProducts','MntFishProducts', 'MntSweetProducts','MntGoldProds']
titles = ['Wines Sold', 'Fruits Sold', 'Meat Products Sold', 'Fish Products Sold', 'Sweets Sold', 'Gold Sold']
colors = ['blue', 'green', 'darkblue','red','orange','yellow']

fig, ax = plt.subplots(2,3, figsize=(16,10))
for i in range(len(columns)):
    sns.histplot(df[columns[i]], bins= 100, ax = ax[i//3, i%3],color=colors[i])
    ax[i//3, i%3].set_title('Distribution of ' + titles[i])
    ax[i//3, i%3].set_xlabel(titles[i])
    ax[i//3, i%3].text(s = f"Total spent on \n{columns[i]} is {df[columns[i]].sum()} ",
                       x = df[columns[i]].max()/3.5, y = 200)


# All products histograms are right skewed, indicating that most customers bought in small amounts, as expected from retail customers.<br>
# Generally, we can infer that Wines brought about highest sales value (675k) and Meat producs followed with 364k, while Fruit and Sweet products brought the lowest sales value (58k and 59k respectively). <br/>
# A probable conclusion is that, since the shoppers for this online store are mostly middle-aged, it would not be surprising to see Wines and Meat products topping the sales chart and Sweets as the least-bought.

# In[18]:


# Total spending - all commodities
df['total_spending'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']


# In[19]:


hist_with_vline(df, 'total_spending')


# The right skew indicates that most shoppers spent less than the average amount (which is around \$600). However, there were also many people who spent more than \\$1000

# #### Children

# Next up, we'll examine further into the customers' home.

# In[20]:


# The dataset made distinction between the number of small children and teenagers in a household. 
# However, for this analysis we would only need to consider the total number:
df['Children'] = df['Kidhome'] + df['Teenhome']


# In[21]:


df.groupby(['Children']).size()


# Most households have 1 or no children, and very few have 3 (as little as 2% of all households).

# #### Marital status

# In[22]:


df.groupby(['Marital_Status']).size()


# The data provides very granulated infomation about marital status. However, for simplicity we can divide this value into two groups.

# In[23]:


df.Marital_Status = df.Marital_Status.replace({'Together': 'Partner',
                                                           'Married': 'Partner',
                                                           'Divorced': 'Single',
                                                           'Widow': 'Single', 
                                                           'Alone': 'Single',
                                                           'Absurd': 'Single',
                                                           'YOLO': 'Single'})


# In[24]:


df.groupby(['Marital_Status']).size()


# Two thirds of the customers lived with partners.

# ### 3. Exploratory Data Analysis 

# #### Average spending: Marital Status Wise

# In[25]:


maritalspent = df.groupby('Marital_Status')['total_spending'].mean().sort_values(ascending=False)
maritalspent_df = pd.DataFrame(list(maritalspent.items()), columns=['Marital Status', 'Average spending'])
maritalspent_df


# In[26]:


plt.figure(figsize=(8,4))
sns.barplot(data = maritalspent_df, x="Average spending", y="Marital Status");

plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('Average Spending', fontsize=20, labelpad=5)
plt.ylabel('Marital Status', fontsize=20, labelpad=5);


# There seems to be little difference between the two amounts. We'll check to see if people living with partners did really spend less, i.e. if the difference holds statistically.

# In[27]:


# Create separate columns for testing purpose
singlespent = df[(df['Marital_Status'] == 'Single')].total_spending
partnerspent = df[(df['Marital_Status'] == 'Partner')].total_spending


# In[28]:


from scipy import stats
stats.ttest_ind(singlespent, partnerspent)


# Although there is a discrepancy between the two groups, that difference is not statistically significant. There is not enough evidence to conclude that people who lived on their own spent more than ones living with partners.

# #### Education Level

# In[29]:


df['Education'].value_counts()  


# In[30]:


# 2n Cycle also means Master (in Bologna Process), we can treat them similarly
df['Education'] = df['Education'].str.replace('2n Cycle', 'Master') 


# In[31]:


eduspent = df.groupby('Education')['total_spending'].mean().sort_values(ascending=False)
eduspent_df = pd.DataFrame(list(eduspent.items()), columns=['Education', 'Average spending'])
eduspent_df


# PhD-level customers seemed to have purchased the most. We will examine whether the discrepancy among education levels hold statistically.

# In[32]:


phdspent = df[(df['Education'] == 'PhD')].total_spending
gradspent = df[(df['Education'] == 'Graduation')].total_spending
msspent = df[(df['Education'] == 'Master')].total_spending
basicspent = df[(df['Education'] == 'Basic')].total_spending


# In[33]:


msspent.dtypes


# In[34]:


# Quick box plots
# Again, we'll be using this type of diagram quite often in this analysis, a function will be time-saving

def box_plot(data, column):
    fig, ax = plt.subplots(figsize=(12, 7))

    # Remove top and right border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Remove y-axis tick marks
    ax.yaxis.set_ticks_position('none')

    # Add major gridlines in the y-axis
    ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)

    # Set plot title
    ax.set_title('Spending by ' + column)

    # Set species names as labels for the boxplot
    labels = df[column].unique()
    ax.boxplot(data, labels=labels)
    plt.show()


# In[35]:


box_plot([phdspent, gradspent, msspent, basicspent], 'Education')


# When the population means of only two groups is to be compared, the t-test is used, but when means of more than two groups are to be compared, ANOVA is preferred.

# In[36]:


# ANOVA and Tukey hsd test
# The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean. 
# The test is applied to samples from two or more groups, possibly with differing sizes.

# Check if there was difference in mean values among education lv
from scipy.stats import f_oneway
f_oneway(phdspent, gradspent, msspent, basicspent)


# In[37]:


# There is enough evidence that the average purchase differs among education level so we will carry out turkey hsd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(endog = df['total_spending'], ## values to compare
                          groups = df['Education'], ## how values are grouped, here by lv of Education
                          alpha=0.05) ## lv of significance
print(tukey)


# We can conclude with confidence that the disparities in expenditure were significant between Basic and each of the other three and between Master and PhD customers.
# Meanwhile, there was no signigicant difference in the amount spent by Grad vs Master and PhD.

# #### Average Spending: Child Status Wise

# In[38]:


nospent = df[(df['Children'] == 0)].total_spending
onespent = df[(df['Children'] == 1)].total_spending
twospent = df[(df['Children'] == 2)].total_spending
threespent = df[(df['Children'] == 3)].total_spending


# In[39]:


box_plot([nospent, onespent, twospent, threespent], 'Children')


# Testing:

# In[40]:


f_oneway(nospent, onespent, twospent, threespent)


# In[41]:


tukey = pairwise_tukeyhsd(endog = df['total_spending'], 
                          groups = df['Children'], 
                          alpha=0.05)
print(tukey)


# Households with different number of children (1 exception, however) spent different amounts. Specifically, households with no children spent the highest on average.

# ### 4 - Multivariate Data Analysis

# #### Spending vs Age (Scatterplot)

# In[42]:


def scatter(x_axis,y_axis):
    plt.figure(figsize=(20,10))


    sns.scatterplot(x=df[x_axis], y=df[y_axis], s=100);

    plt.xticks( fontsize=16)
    plt.yticks( fontsize=16)
    plt.xlabel(x_axis, fontsize=20, labelpad=20)
    plt.ylabel(y_axis, fontsize=20, labelpad=20)
    
scatter('Age', 'total_spending')


# All is noise, it seems that there is no relationship between customers' age and their purchasing behavior.

# What about customers' seniority with the platform?

# #### Customer seniority and Spending

# In[43]:


scatter('Months_Enrolled','total_spending')


# Again, there is no particular pattern.

# #### Spending vs Income

# In[46]:


scatter('Income','total_spending')


# A clear exponential relationship is present. 

# ### 5. Marketing Capaign Effectiveness

# First, we will compare the number of people accepting offer in each campaign to see if any of the campaigns outperformed the others.

# Since the response was recorded as 1 or 0, we can quickly determine the total number by simple summing.

# In[47]:


accepted = df[['AcceptedCmp1', 'AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5']].sum()
print(accepted)


# Except that the second campaign attracted as few as 30 people, the remaining campaigns attracted similar number of shoppers.

# Suppose we hypthesize that people who accepted more offers would have spent more (or the other way around, people who spent a lot were more prone to accept offers). Let's check if this hypothesis is true.

# In[48]:


df['Total_accepted'] = df[['AcceptedCmp1', 'AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5']].sum(axis = 1, skipna = True)
print(df.Total_accepted)


# In[49]:


df.groupby(['Total_accepted']).size()


# In[50]:


acceptspent = df.groupby('Total_accepted')['total_spending'].mean().sort_values(ascending=False)
acceptspent_df = pd.DataFrame(list(acceptspent.items()), columns=['Total offers accepted', 'Average spending'])
acceptspent_df


# In[51]:


no_accept = df[(df['Total_accepted'] == 0)].total_spending
one_accept = df[(df['Total_accepted'] == 1)].total_spending
two_accept = df[(df['Total_accepted'] == 2)].total_spending
three_accept = df[(df['Total_accepted'] == 3)].total_spending
four_accept = df[(df['Total_accepted'] == 4)].total_spending


# In[52]:


box_plot([no_accept, one_accept, two_accept, three_accept, four_accept], 'Total_accepted')


# In[53]:


f_oneway(no_accept, one_accept, two_accept, three_accept, four_accept)


# From the results above, we can conclude that either the offers induced the customers to buy more, thus spent more; or the customers who spent a lot were more willing to take advantage of offers in marketing campaigns. <br>
# Since the direction of the causation is unclear, we cannot conclude decisively about the effectiveness of using offers in marketing campaign.
