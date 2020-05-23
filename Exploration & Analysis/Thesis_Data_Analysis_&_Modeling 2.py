#Import packages
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
#Set plotting font to be consistant with paper
sns.set(font = "Times New Roman", style = 'white')

#Read in survey data
df = pd.read_csv("/Users/joevorbeck/Documents/Data-Analytics-MA-Thesis/Files/data.csv")

#Inspect survey
#df.head()

### Data Pre-Processing ###

#Rename column for us region
df.rename(columns = {"If you live in the United States, which state or territory do you live in?" : "US_Region"},inplace = True)

#Extract column of comments for text analysis
survey_comments = df['Any additional notes or comments']

#Dropping the timestamps and the notes/comments field
df = df.drop(['Timestamp', 'Any additional notes or comments'], axis = 1)

#Check ages prior to binning
#df['Age'].value_counts()

#Need to remove ages that are non-sensical, i.e '-329','-99999'
df = df[(df.Age >= 18) & (df.Age <= 75)]

#Histogram of age - check distribution
sns.distplot(df['Age'], kde = True, rug = False, bins = 15, color = 'blue',hist_kws={'edgecolor':'black'},
    kde_kws={'linewidth': 1.5})
plt.title("Density Plot and Histogram of Age")
plt.ylabel("Density\n")
plt.xlabel("Age\n")

#Obtain average age
age_mean = df.Age.mean()
age_mean

#Column values
country_list = df['Country']
region_list = df['US_Region']

#Gender is a freeform field so need to account for typos and different spellings
gender_list = df['Gender']
#set(gender_list)

#Focusing on two tech-hub cities: NY & CA
region_list_binned = [x if x in ['NY','CA'] else "Other" for x in region_list]

#As there are other genders in the survey and we are concerned with male & female, the others will be binned
def val_changer(y):
    for x in y:
        if x in ['Male','Mail','M','Male ','Man','Malr','maile','male','msle']:
            x = 'Male'
        elif x in ['F','Female','Femake','Female ','femail','female','f']:
            x = 'Female'
        else:
            x = 'Other'
        return x

gender_list_binned = gender_list.apply(val_changer)

#List of company sizes for binning
smaller = ['1-5','6-25','26-100']
larger = ['100-500','500-1000','More than 1000']

#Bin company sizes
def company_bin(x):
    if x['How many employees does your company or organization have?'] in smaller:
        val = 'Smaller'
    else:
        val = 'Larger'
    return val

df['How many employees does your company or organization have?'] = df.apply(company_bin, axis = 1)

#Bin ease of leave for mental health conditions
def ease_of_leave(x):
    if x['How easy is it for you to take medical leave for a mental health condition?'] == "Very easy" or x['How easy is it for you to take medical leave for a mental health condition?'] == 'Somewhat easy':
        val = 'Easy'
    else:
        val = "Don't Know or Difficult"
    return val

df['How easy is it for you to take medical leave for a mental health condition?'] = df.apply(ease_of_leave, axis = 1)

#List of unique countries included in the survey
set(country_list)

#List of continents for binning
europe = ['Austria','Belgium','Bulgaria','Boznia and Herzegovina','Croatia','Czeh Republic','Denmark','Finland','France','Germany','Greece','Hungary','Ireland','Italy','Latvia','Moldova','Netherlands','Poland','Portugal','Poland','Romania','Slovenia','Spain','Sweden','United Kingdom','Russia','Norway','Switzerland']
asia = ['China','India','Japan','Philippines','Singapore','Thailand','Georgia']
north_america_non_us = ['Mexico','Canada','The Bahamas']
south_america = ['Brazil','Colombia','Costa Rica','Uruguay']
africa = ['South Africa','Zimbabwe','Nigera']
oceania = ['New Zealand','Australia']

#Bin Countries
def country_bin(x):
    if x['Country'] == 'United States':
        x = 'United States'
    elif x['Country'] in europe:
        x = 'Europe'
    elif x['Country'] in asia:
        x = 'Asia'
    elif x['Country'] in south_america:
        x = 'South America'
    elif x['Country'] in oceania:
        x = 'Oceania'
    elif x['Country'] in africa:
        x = 'Africa'
    else:
        x = 'North America (Excluding U.S)'
    return x

df['Country'] = df.apply(country_bin, axis = 1)

#Bin Ages - age approximates a normal distrib. so mean was used to binarize it
def age_bin(x):
        if x['Age'] <= age_mean:
            x = 1
        elif x['Age'] > age_mean:
            x = 0
        return x

df['Age'] = df.apply(age_bin, axis = 1)

#Create dependent variables
#DV 1 - Bin yes and some of them to 1, no 0
def supervisor_dv(x):
    if x['Would you be willing to discuss a mental health issue with your direct supervisor(s)?'] == 'Yes' or x['Would you be willing to discuss a mental health issue with your direct supervisor(s)?'] == 'Some of them':
        val = 1
    else:
        val = 0    
    return val 

#DV 2 - Bin yes and maybe to 1, no 0
def neg_consq(x):
    if x['Do you think that discussing a mental health issue with your employer would have negative consequences?'] == 'Yes' or x['Do you think that discussing a mental health issue with your employer would have negative consequences?'] == 'Maybe':
        val = 1
    else:
        val = 0
    return val

df['mh_issue_supervisor'] = df.apply(supervisor_dv, axis = 1)
df['mh_issue_neg_consq'] = df.apply(neg_consq, axis = 1)

#Drop old values and replace df with new ones for country and us region
df.drop(['US_Region','Gender'], axis = 1, inplace = True)
df['US_Region'] = region_list_binned
df['Gender'] = gender_list_binned

#Use 'other' gender as a filter for the DF - concerned with male & female for this study
df = df[df.Gender != 'Other']

#Create dummy variables 
df2 = pd.get_dummies(df)

### Begin Analysis & Modeling ###

#Create sets of independent variables
x_main = df2['Is your employer primarily a tech company/organization?_Yes']
x_main_demog = df2[['Is your employer primarily a tech company/organization?_Yes', 'Have you sought treatment for a mental health condition?_Yes','Age','Gender_Male','Country_Europe','Country_North America (Excluding U.S)', 'Country_Africa','Country_Oceania','Country_Asia','Country_South America','Do you have a family history of mental illness?_Yes']]
x_main_demog_work = df2[['Is your employer primarily a tech company/organization?_Yes', 'Have you sought treatment for a mental health condition?_Yes', 'Age','Gender_Male','Country_Europe','Country_North America (Excluding U.S)','Country_Africa','Country_Oceania','Country_Asia','Country_South America','Do you have a family history of mental illness?_Yes','Are you self-employed?_Yes','Do you work remotely (outside of an office) at least 50% of the time?_Yes']]
x_main_demog_work_employer = df2[['Is your employer primarily a tech company/organization?_Yes','Have you sought treatment for a mental health condition?_Yes', 'Age','Gender_Male','Country_Europe','Country_North America (Excluding U.S)','Country_Africa', 'Country_Oceania','Country_Asia','Country_South America','Do you have a family history of mental illness?_Yes','Are you self-employed?_Yes','Do you work remotely (outside of an office) at least 50% of the time?_Yes','How many employees does your company or organization have?_Smaller','Does your employer provide mental health benefits?_Yes','Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?_Yes','How easy is it for you to take medical leave for a mental health condition?_Easy']]

#Create sets of dependent variables
y_supervisor = df['mh_issue_supervisor']
y_consq = df['mh_issue_neg_consq']

#Group data by if employee works for a tech org or not and use employer level variables
tech_employer_level  = df2[['Is your employer primarily a tech company/organization?_Yes','Does your employer provide mental health benefits?_Yes','Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?_Yes','How easy is it for you to take medical leave for a mental health condition?_Easy']]
tech_employer_level.rename(columns = {"Is your employer primarily a tech company/organization?_Yes": "Primarily Tech Company/Organization_Yes", "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?_Yes": "Anonymity Protected_Yes","How easy is it for you to take medical leave for a mental health condition?_Easy" : "Ease of Leaving for Mental Health Condition_Easy"},inplace=True)

#Protection of anonymity when using mental health resources at work grouped by tech/nontech companies
ax1 = sns.catplot(x="Anonymity Protected_Yes",col="Primarily Tech Company/Organization_Yes",data=tech_employer_level, palette = "Blues_d", kind="count",height=5, aspect=1)
ax1.fig.subplots_adjust(top = .85)
plt.suptitle("Comparative Barplot of Anonymity Protection When Taking Advantage of Mental Health Treatment Resources by Company Designation")

#Ease of leaving for a mental health condition grouped by tech/nontech companies
ax2 = sns.catplot(x="Ease of Leaving for Mental Health Condition_Easy",col="Primarily Tech Company/Organization_Yes",data=tech_employer_level, palette = "Blues_d", kind="count",height=5, aspect=1)
ax2.fig.subplots_adjust(top = .85)
plt.suptitle("Comparative Barplot of Ease of Leaving for Mental Health Issues by Company Designation")

#Does employer provide mental health resource grouped by tech/nontech companies
ax3 = sns.catplot(x="Does your employer provide mental health benefits?_Yes",col="Primarily Tech Company/Organization_Yes",data=tech_employer_level, palette = "Blues_d", kind="count",height=5, aspect=1)
ax3.fig.subplots_adjust(top = .85)
plt.suptitle("Comparative Barplot of Provisions of Mental Health Resources by Company Designation")

#Compute variance for all IVs - help undestand how betas will change in stepwise logits
x_all_var = pd.DataFrame(x_main_demog_work_employer.var()).reset_index().sort_values(by = 0, ascending = False)
#Rename variable to condense plot
x_all_var.replace({"Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?_Yes" : "Anonymity Protected_Yes"}, inplace = True)
x_all_var.rename(columns = {0 : "Variance",  "index" : "Variable"}, inplace = True)

#Barplot of variance
sns.barplot(x = 'Variance', y = 'Variable', data = x_all_var, palette="Blues_d", orient = 'h')
plt.xticks(rotation = 90)
plt.title('Variance Across All Independent Variables')

#Compute correlation coefficients between all IVs - collinearity?
x_all_corr = x_main_demog_work_employer.corr()
#Rename field to condense plot
x_all_corr.rename(index = {"Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?_Yes": "Anonymity Protected_Yes"}, inplace = True)

#Heatmap of correlations
mask = np.zeros_like(x_all_corr)
mask[np.triu_indices_from(mask)] = False
sns.heatmap(x_all_corr, annot = False, linewidth = .01, xticklabels = False, vmin=-1, vmax = 1, cmap="Blues", mask= mask)
plt.title("Pearson's Correlation Across All Independent Variables")


### Starting with First Dependent Variable ###                                
### First Logit Model Using Main Independent Variable ###                          

logit_sup = sm.Logit(y_supervisor, x_main).fit()
logit_sup.summary()

### Second Logit Model Adding in Demographic Level Variables ###

logit_sup_2 = sm.Logit(y_supervisor, x_main_demog).fit()
logit_sup_2.summary()

### Third Logit Model Adding in Work Level Variables ###

logit_sup_3 = sm.Logit(y_supervisor, x_main_demog_work).fit()
logit_sup_3.summary()

### Fourth Logit Model Adding in Employer Level Variables ###

logit_sup_4 = sm.Logit(y_supervisor, x_main_demog_work_employer).fit()
logit_sup_4.summary()

### Second Dependent Variable ###

### First Logit Model Using Main Indepdent Variable ### 

logit_consq = sm.Logit(y_consq, x_main).fit()
logit_consq.summary()

### Second Logit Model Adding in Demographic Level Variables ###

logit_consq_2 = sm.Logit(y_consq, x_main_demog).fit()
logit_consq_2.summary()

### Third Logit Model Adding in Work Level Variables ###

logit_consq_3 = sm.Logit(y_consq, x_main_demog_work).fit()
logit_consq_3.summary()

### Fourth Logit Model Adding in Employer Level Variables ###

logit_consq_4 = sm.Logit(y_consq, x_main_demog_work_employer).fit()
logit_consq_4.summary()
