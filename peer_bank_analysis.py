#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 23:13:01 2021

@author: chrisarnold
"""


#%%  Create the App

import streamlit as st

st.title("""
         Peer Bank Analysis 
         
         ***Loss Forecasting Peer Banks***
         
         ***Allowance For Credit Loss Forecasting***
         
         ***CECL Coverage Rates***
         
        """)
        
        
#%% Install and import relevant packages

#pip install streamlit
#pip install yfinance as yf

#import yfinance as yf

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt


#%% Pull in the data from peer banks and macro data from fred qd #https://files.stlouisfed.org/files/htdocs/fred-md/quarterly/current.csv

# Using @st.cache tells streamlit to check three things: 1.bytecode that makes up the function, 2. code, variables and files that function depends on, 3. input parameters

streamlit_app_path = 'https://github.com/arnold-798/Bank-ACL-and-Loss-Forecasting/blob/main/SNL_Peer_Bank_042021.csv'

@st.cache
def load_peer_data(nrows):
    data_path = os.path.join(streamlit_app_path, 'SNL_Peer_Bank_042021.csv')
    data_path = data_path.replace('\\', '/'')
    data = pd.read_csv(data_path, sep = ',', nrows=nrows)
    data = pd.DataFrame(data)
    return data

@st.cache
def load_fredqd(nrows):
    dat_path = os.path.join(streamlit_app_path, 'fred_qd_042021.csv') 
    data_path = data_path.replace('\\', '/'')
    data = pd.read_csv(data_path, sep = ',', skiprows=[i for i in range(1,131)], nrows=nrows) 
    data = pd.DataFrame(data)
    return data

fred_qd_raw = load_fredqd(120)
peer_bank_raw = load_peer_data(2400)

st.header("Individual Bank Panel Data + FREDQD Macroeconomic Data")

st.subheader("Quarterly Bank Data")
st.write(peer_bank_raw)

st.subheader("Quarterly Macroeconomic Data")
st.write(fred_qd_raw)

#%% Explore the data


st.header("Exploration of Banks Panel Data")


# High level summary statistics

def df_sum(mydata):
  for i in mydata:
    # Take each column in the dataset and provide the following measurements
    datatype = dict(mydata.dtypes)
    nmb_obsv = dict(mydata.count())
    missing_na = dict(mydata.isnull().sum())
    
  
  # Put the measurement parameters together into a dataframe
  toh_sum_temp = pd.DataFrame([datatype, nmb_obsv, missing_na])
  toh_sum_temp_v2 = pd.DataFrame.transpose(toh_sum_temp)

  toh_sum = toh_sum_temp_v2.rename(columns = {0: 'Data_Type',
                                              1: '#Obsv',
                                              2: '#NA'})

  return (toh_sum)

st.subheader("High Level Summary Statistics")
st.write(pd.DataFrame(df_sum(peer_bank_raw)))

#print(df_sum(peer_bank_raw))

# Summary Statistics

def sum_stats(mydata):

   num_mydata = pd.DataFrame(mydata).select_dtypes(include = np.number)
   
   # Compute Summary Statistics
   agg_sum = dict(num_mydata.sum())
   num_obsv = dict(num_mydata.count())
   mean = dict(num_mydata.mean())
   median = dict(num_mydata.median())
   max = dict(num_mydata.max())
   min = dict(num_mydata.min())
   range = dict(num_mydata.max()-num_mydata.min())
   st_dev = dict(num_mydata.std())
   skewness = dict(num_mydata.skew())
 
   agg_sumstats_v1 = pd.DataFrame([agg_sum, num_obsv, mean, median, max, min, range, st_dev, skewness])
   agg_sumstats_v2 = pd.DataFrame.transpose(agg_sumstats_v1)
   agg_sumstats_v3 = agg_sumstats_v2.rename(columns = {0: 'agg_sum',
                                                       1: 'num_obsv',
                                                       2: 'mean',
                                                       3: 'median',
                                                       4: 'max',
                                                       5: 'min',
                                                       6: 'range',
                                                       7: 'st_dev',
                                                       8: 'skewness'})
   
 
   toh_sum = df_sum(num_mydata)
   agg_sumstats = pd.concat([toh_sum, agg_sumstats_v3], axis = 1)
   return (agg_sumstats)

# Test out the function on the training data

st.subheader("Descriptive Statistics on the Panel Bank Data")
st.write(pd.DataFrame(sum_stats(peer_bank_raw)))

#%% Allowance for Credit Loss Coverage Rates 

# Create a new dataframe with the allowance for credit loss and coverage rates

acl_data = peer_bank_raw[['BANK_TYPE', 'BANK_NAME_ABBR', 'YEAR_QUARTER', 'REG_STATE', 'TOTAL_LOAN_LEASES_EXCL_HFS', 'TOTAL_RESV', 'PROV_FOR_LOAN_LEASE_LOSS', 
                          'LN_AND_LS_HFI_AMORT_COST_RE_CONST_LOAN', 'LN_AND_LS_HFI_AMORT_COST_COM_RE_LOAN', 'LN_AND_LS_HFI_AMORT_COST_RES_RE_LOAN', 
                          'LN_AND_LS_HFI_AMORT_COST_COM_LOAN', 'LN_AND_LS_HFI_AMORT_COST_CREDIT_CARD', 'LN_AND_LS_HFI_AMORT_COST_OTHER_CONSUMER_LOAN', 
                          'LN_AND_LS_HFI_AMORT_COST_TOTAL', 'LN_AND_LS_HFI_ALLOWANCE_RE_CONST_LOAN', 'LN_AND_LS_HFI_ALLOWANCE_COM_RE_LOAN', 
                          'LN_AND_LS_HFI_ALLOWANCE_RES_RE_LOAN', 'LN_AND_LS_HFI_ALLOWANCE_COM_LOAN', 'LN_AND_LS_HFI_ALLOWANCE_CREDIT_CARD', 
                          'LN_AND_LS_HFI_ALLOWANCE_OTHER_CONSUMER_LOAN', 'LN_AND_LS_HFI_ALLOWANCE_UNALLOCATED', 'LN_AND_LS_HFI_ALLOWANCE_TOTAL', 'latitude', 'longitude']]

acl_data = acl_data.assign(Construction_RE_Coverage_Rate = acl_data['LN_AND_LS_HFI_ALLOWANCE_RE_CONST_LOAN'] / acl_data['LN_AND_LS_HFI_AMORT_COST_RE_CONST_LOAN'])
acl_data = acl_data.assign(Commercial_RE_Coverage_Rate = acl_data['LN_AND_LS_HFI_ALLOWANCE_COM_RE_LOAN'] / acl_data['LN_AND_LS_HFI_AMORT_COST_COM_RE_LOAN'])
acl_data = acl_data.assign(Residential_RE_Coverage_Rate = acl_data['LN_AND_LS_HFI_ALLOWANCE_RES_RE_LOAN'] / acl_data['LN_AND_LS_HFI_AMORT_COST_RES_RE_LOAN'])
acl_data = acl_data.assign(Commercial_Coverage_Rate = acl_data['LN_AND_LS_HFI_ALLOWANCE_COM_LOAN'] / acl_data['LN_AND_LS_HFI_AMORT_COST_COM_LOAN'])
acl_data = acl_data.assign(Credit_Card_Coverage_Rate = acl_data['LN_AND_LS_HFI_ALLOWANCE_CREDIT_CARD'] / acl_data['LN_AND_LS_HFI_AMORT_COST_CREDIT_CARD'])
acl_data = acl_data.assign(Other_Consumer_Coverage_Rate = acl_data['LN_AND_LS_HFI_ALLOWANCE_OTHER_CONSUMER_LOAN'] / acl_data['LN_AND_LS_HFI_AMORT_COST_OTHER_CONSUMER_LOAN'])
acl_data = acl_data.assign(Total_ACL_Coverage_Rate = acl_data['LN_AND_LS_HFI_ALLOWANCE_TOTAL'] / acl_data['LN_AND_LS_HFI_AMORT_COST_TOTAL'])

acl_coverage_rates = acl_data[['BANK_TYPE', 'BANK_NAME_ABBR', 'YEAR_QUARTER', 'REG_STATE', 'Construction_RE_Coverage_Rate', 'Commercial_RE_Coverage_Rate', 
                               'Residential_RE_Coverage_Rate', 'Commercial_Coverage_Rate', 'Credit_Card_Coverage_Rate', 'Other_Consumer_Coverage_Rate', 
                               'Total_ACL_Coverage_Rate', 'latitude', 'longitude']]

acl_cov_rate_20Q4 = acl_coverage_rates[acl_coverage_rates['YEAR_QUARTER'] == '2020Q4']

acl_cov_rate_20Q4_sort = pd.DataFrame(acl_cov_rate_20Q4, index=acl_cov_rate_20Q4['BANK_NAME_ABBR'])
acl_cov_rate_20Q4_sort = acl_cov_rate_20Q4_sort.sort_index(ascending=True)

bank_type_acl_coverage = acl_cov_rate_20Q4.groupby('BANK_TYPE').agg('mean')[['Construction_RE_Coverage_Rate', 'Commercial_RE_Coverage_Rate', 
                               'Residential_RE_Coverage_Rate', 'Commercial_Coverage_Rate', 'Credit_Card_Coverage_Rate', 'Other_Consumer_Coverage_Rate',
                               'Total_ACL_Coverage_Rate']].plot(kind='bar', figsize=(25, 7), stacked=False)


st.subheader("ACL Coverage Rates Across Comparison Groups")
st.bar_chart(acl_cov_rate_20Q4.groupby('BANK_TYPE').agg('mean')[['Construction_RE_Coverage_Rate', 'Commercial_RE_Coverage_Rate', 
                               'Residential_RE_Coverage_Rate', 'Commercial_Coverage_Rate', 'Credit_Card_Coverage_Rate', 'Other_Consumer_Coverage_Rate',
                               'Total_ACL_Coverage_Rate']])

    # .plot(kind='bar', figsize=(25, 7), stacked=False
    
bank_indiv_acl_coverage = acl_cov_rate_20Q4.groupby('BANK_NAME_ABBR').agg('mean')[['Total_ACL_Coverage_Rate']].plot(kind='bar', figsize=(25, 7), stacked=False)

st.subheader("ACL Coverage Rate Across Banks")
st.bar_chart(acl_cov_rate_20Q4.groupby('BANK_NAME_ABBR').agg('mean')[['Total_ACL_Coverage_Rate']])


#%% In-Depth Visuals on the Data

coverage_rates_states = pd.DataFrame(acl_cov_rate_20Q4[['latitude', 'longitude', 'Total_ACL_Coverage_Rate']])

st.header("ACL Coverage Rates Across States")
st.subheader("This would probably be more interesting to look at loan level data and examine coverage rates across states / counties in greater depth.")
st.subheader("Most likely in the event of a natural disaster")
st.map(coverage_rates_states)

#%% Historical NCO Rates Across Banks

st.header("Historical NCO Rates Across Banks")
hist_nco = peer_bank_raw[['BANK_NAME_ABBR', 'YEAR_QUARTER', 'REG_NCO_TO_AVG_LOAN']]
hist_nco_v1 = hist_nco.groupby('YEAR_QUARTER').agg('mean')[['REG_NCO_TO_AVG_LOAN']]
st.line_chart(hist_nco_v1)
hist_nco_v2 = hist_nco.groupby('BANK_NAME_ABBR').agg('mean')[['REG_NCO_TO_AVG_LOAN']]
st.line_chart(hist_nco_v2)

#%% Subset the data to include 5 peer Banks: PNC, KEY, MTB, USB, FITB

st.title("""
         
         # PNC Loss Forecast Prediction
         
         *** Random Forest Classifier Model ***
         
         *** Simple Linear Model ***
         
         """)

# Plot the predicted values against the actual values for PNC

st.header("Data Exploration")




#KEY = pd.DataFrame(peer_bank_raw[peer_bank_raw['BANK_NAME_ABBR'] == 'KEY'])

PNC = pd.DataFrame(peer_bank_raw[peer_bank_raw['BANK_NAME_ABBR'] == 'PNC'])

#FITB = pd.DataFrame(peer_bank_raw[peer_bank_raw['BANK_NAME_ABBR'] == 'FIFTH THIRD'])

#MTB = pd.DataFrame(peer_bank_raw[peer_bank_raw['BANK_NAME_ABBR'] == 'M&T BANK'])

#USB = pd.DataFrame(peer_bank_raw[peer_bank_raw['BANK_NAME_ABBR'] == 'USB'])

#peer_group_raw_1 = pd.concat([KEY, PNC], axis=1)

#peer_group_raw_2 = pd.concat([FITB, MTB], axis=1)

#peer_group_raw_3 = pd.concat([peer_group_raw_1, USB], axis=1)

#peer_group_raw = pd.concat([peer_group_raw_3, peer_group_raw_2], axis=1)

#peer_group_raw = pd.DataFrame(peer_group_raw)

#print(sum_stats(PNC))

#print(sum_stats(fred_qd_raw))


#%% Subset the peer bank data to only the necessary variables


peer_bank_data = PNC[['BANK_NAME_ABBR', 'YEAR_QUARTER', 'REG_NCO_TO_AVG_LOAN', 'PROV_FOR_LOAN_LEASE_LOSS', 'TOTAL_LOAN_LEASES_EXCL_HFS', 'REG_RESV_TO_NPA']]

peer_bank_data = peer_bank_data.reset_index(drop=True)

macro_data = fred_qd_raw[['sasdate','GDPC1', 'DPIC96', 'LNS12032194', 'UNRATE', 'GDPCTPI', 'CPILFESL', 'FEDFUNDS']]


#%% Append the peer bank data with the macro data

pnc_macro_data = pd.concat([peer_bank_data, macro_data], axis=1)

print(sum_stats(pnc_macro_data))

pnc_macro_sum_stats = sum_stats(pnc_macro_data)

st.subheader("High Level Summary Statistics Raw Data")
st.write(pd.DataFrame(pnc_macro_sum_stats))


st.subheader("Historical NCOs for PNC")

hist_values = np.histogram(pnc_macro_data['REG_NCO_TO_AVG_LOAN'], bins = 5)

st.line_chart(pnc_macro_data['REG_NCO_TO_AVG_LOAN'])

#%% If there are any missing values, impute with the median

# Build function to impute missing values

def single_imputation(mydata, impute_type='mean' or 'median', train_or_test = 'train' or 'test'):

  if train_or_test == 'train':
    num_mydata = pd.DataFrame(mydata).select_dtypes(include = np.number)
    
    if impute_type == 'mean':
      imputed_data = num_mydata.apply(lambda x: x.fillna(x.mean()), axis = 0)
    if impute_type == 'median':
      imputed_data = num_mydata.apply(lambda x: x.fillna(x.median()), axis = 0)
    
    str_mydata = pd.DataFrame(mydata).select_dtypes(include = ['object'])
    
    str_mydata_v1 = str_mydata.apply(lambda x: x.fillna('Unknown'))

    imputed_data_final = pd.concat([imputed_data, str_mydata_v1], axis = 1)
    
    col_mean = dict(num_mydata.mean())
    col_median = dict(num_mydata.median())

  if train_or_test == 'test' and impute_type == 'mean':
    num_mydata1 = pd.DataFrame(mydata).select_dtypes(include = np.number)

    imputed_data = num_mydata1.apply(lambda x: x.fillna(col_mean), axis = 0)
  if train_or_test == 'test' and impute_type == 'median':
    imputed_data = num_mydata1.apply(lambda x: x.fillna(col_median), axis = 0)

    str_mydata_v1 = str_mydata.apply(lambda x: x.fillna('Unknown'))

    #str_mydata_v2 = pd.DataFrame(mydata).select_dtypes(include = ['object'])
    
    imputed_data_final = pd.concat([imputed_data, str_mydata_v1], axis = 1)

  return (imputed_data_final)

# Test out function to make sure new datasets have no imputed values

sum_stats(single_imputation(pnc_macro_sum_stats, 'median', 'train'))

single_imputation(pnc_macro_sum_stats).head()


#%% Examine the NCO Rate variable to determine an appropriate number of buckets
sns.set_style('whitegrid')

#raw_df = pd.read_excel('2018_Sales_Total.xlsx')
df = pnc_macro_data.groupby(['BANK_NAME_ABBR', 'YEAR_QUARTER'])['REG_NCO_TO_AVG_LOAN'].sum().reset_index()

nco_bar_plot = df['REG_NCO_TO_AVG_LOAN'].plot(kind='hist')

#%% Create a "Bucket" variable for NCO rates

pnc_macro_data['NCO_RATE_SEG'] = pd.cut(pnc_macro_data['REG_NCO_TO_AVG_LOAN'], bins = 5, labels = ['Min', 'Low', 'Mid', 'High', "Max"])

pnc_macro_data['NCO_RATE_SEG'].head()


#%% Barplot

pnc_macro_data_sort = pd.DataFrame(pnc_macro_data, index=pnc_macro_data['YEAR_QUARTER'])
pnc_macro_data_sort = pnc_macro_data_sort.sort_index(ascending=True)

pnc_macro_data.groupby('YEAR_QUARTER').agg('mean')['REG_NCO_TO_AVG_LOAN'].plot(kind='bar', figsize=(25, 7), stacked=True, title='NCO Rates Across Quarters')

#%% Create training and test data

# take the first 70% of observations (120 obverstaions in the dataset)

pnc_macro_train = pnc_macro_data.iloc[:84]

pnc_macro_train = single_imputation(pnc_macro_train)

pnc_macro_test = pnc_macro_data.iloc[85:115]

pnc_macro_test = single_imputation(pnc_macro_test)

pnc_macro_validation = pnc_macro_data.iloc[116:120]

pnc_macro_validation = single_imputation(pnc_macro_validation)

st.subheader("Training Data Summary Statistics - Post Imputation")
st.write(pd.DataFrame(sum_stats(pnc_macro_train)))

st.subheader("Plot of the Macroeconomic Variables Impacting NCOs")
cecl_macro_subset = pnc_macro_data[['UNRATE', 'LNS12032194','GDPC1', 'CPILFESL', 'FEDFUNDS']]
cecl_macro_pct = cecl_macro_subset[['UNRATE', 'FEDFUNDS']]
st.line_chart(cecl_macro_pct)

#%% Select the predictors and create an object for the dependent variable

pnc_macro_train_x = pnc_macro_train[['PROV_FOR_LOAN_LEASE_LOSS', 'TOTAL_LOAN_LEASES_EXCL_HFS', 'GDPC1', 'DPIC96', 'UNRATE', 'CPILFESL', 'FEDFUNDS']]

pnc_macro_train_yc = pnc_macro_train['REG_NCO_TO_AVG_LOAN']

pnc_macro_train_yd = pnc_macro_data['NCO_RATE_SEG']

pnc_macro_train_yd = pnc_macro_train_yd.iloc[:84]

pnc_macro_test_x = pnc_macro_test[['PROV_FOR_LOAN_LEASE_LOSS', 'TOTAL_LOAN_LEASES_EXCL_HFS', 'GDPC1', 'DPIC96', 'UNRATE', 'CPILFESL', 'FEDFUNDS']]

pnc_macro_test_yc = pnc_macro_test['REG_NCO_TO_AVG_LOAN']

pnc_macro_test_yd = pnc_macro_data['NCO_RATE_SEG']

pnc_macro_test_yd = pnc_macro_test_yd.iloc[85:115]

pnc_macro_validation_x = pnc_macro_validation[['PROV_FOR_LOAN_LEASE_LOSS', 'TOTAL_LOAN_LEASES_EXCL_HFS', 'GDPC1', 'DPIC96', 'UNRATE', 'CPILFESL', 'FEDFUNDS']]

pnc_macro_validation_yc = pnc_macro_validation['REG_NCO_TO_AVG_LOAN']

pnc_macro_validation_yd = pnc_macro_data['NCO_RATE_SEG']

pnc_macro_validation_yd = pnc_macro_validation_yd.iloc[116:120]

#%% Build a Random Forest Calsifier

from sklearn.ensemble import RandomForestClassifier as RFC

Rand_Forest = RFC(n_estimators = 500, criterion = 'entropy', max_depth = 5, bootstrap=True)

rand_forest_1 = Rand_Forest.fit(pnc_macro_train_x, pnc_macro_train_yd)

rand_forest_2 = Rand_Forest.fit(pnc_macro_test_x, pnc_macro_test_yd)

rand_forest_3 = Rand_Forest.fit(pnc_macro_validation_x, pnc_macro_validation_yd)

#%% Create predictions for the training data with the random forest classifier

pnc_train_yd_pred = rand_forest_1.predict_proba(pnc_macro_train_x)

pnc_test_yd_pred = rand_forest_2.predict_proba(pnc_macro_test_x)

pnc_validation_yd_pred = rand_forest_3.predict_proba(pnc_macro_validation_x)


#%% Plot the training random forest

from sklearn import tree
from dtreeviz.trees import dtreeviz # will be used for tree visualization
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})

plt.figure(figsize=(20,20))
_ = tree.plot_tree(rand_forest_1.estimators_[0], feature_names=pnc_macro_train_x.columns, filled=True)

pnc_train_treeplot = tree.plot_tree(rand_forest_1.estimators_[0], feature_names = pnc_macro_train_x.columns, filled=True)

st.header("Random Forest Classifier - Tree Plot")
st.pyplot(pnc_train_treeplot)

#%% Build a Simple Linear Regression model

from sklearn.linear_model import LinearRegression as OLS

OLS_Reg = OLS(n_estimators = 500, criterion = 'entropy', max_depth = 5, bootstrap=True)

rand_forest_reg1 = OLS_Reg.fit(pnc_macro_train_x, pnc_macro_train_yc)

rand_forest_reg2 = OLS_Reg.fit(pnc_macro_test_x, pnc_macro_test_yc)

rand_forest_reg3 = OLS_Reg.fit(pnc_macro_validation_x, pnc_macro_validation_yc)


#%% Create predictions for the random forest regressor 

pnc_train_rfr_yd_pred = rand_forest_reg1.predict_proba(pnc_macro_train_x, pnc_macro_train_yc)

pnc_test_rfr_yd_pred = rand_forest_reg2.predict_proba(pnc_macro_test_x, pnc_macro_test_yc)

pnc_validation_rfr_yd_pred = rand_forest_reg3.predict_proba(pnc_macro_validation_x, pnc_macro_validation_yc)


#%% Create the app with loaded data

