# -*- coding: utf-8 -*-
"""
Created on Fri May 31 01:38:38 2019

@author: chenz
"""

import numpy as np
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

class CardTransactionData(object):
    ''' Create a CardTransactionData class to get and
    process the card transaction data '''
    def __init__(self, data_link):
        self.data_link = data_link
    def get_data(self):
        '''Download the card transaction data from the link provided'''
        resp = urlopen(self.data_link)
        zipfile = ZipFile(BytesIO(resp.read()))
        file_name = zipfile.namelist()[0]
        extracted_file = zipfile.open(file_name)
        return extracted_file
    def read_data(self, extracted_file, frac=0.02):
        '''Read in the json file as Pandas DataFrame.
           Use frac to take a small sample from the whole dataset, to save
           the processing time while testing the code.'''
        df_full = pd.read_json(extracted_file, lines=True)
        df = df_full.sample(frac=frac)
        return df 
    def datetime_transformation(self,df):
        '''Transform the columns into datatime format'''
        df[['accountOpenDate', 'currentExpDate', 'dateOfLastAddressChange', 'transactionDateTime']] =\
        df[['accountOpenDate', 'currentExpDate', 'dateOfLastAddressChange', 'transactionDateTime']]\
        .apply(pd.to_datetime)
        return df
    def columns_dropoff(self,df):
        ''' Column 'customerId' is identical to the column 'accountNumber', drop one of them.
        Some columns are empty and need to be dropped, such as ['echoBuffer','merchantCity', 'merchantState',
        'merchantZip', 'posOnPremises', 'recurringAuthInd']
        '''
        drop_columns= ['customerId','echoBuffer','merchantCity', 'merchantState','merchantZip', \
                       'posOnPremises', 'recurringAuthInd']
        df = df.drop(drop_columns, axis=1)
        return df
    def duplicate_identify(self,df, df_duplicate_group):
        ''' Return the duplicated and reversed transactions'''
        idx_dup = df_duplicate_group.index
        df_drop = pd.DataFrame()
        
        for idx in idx_dup:
            accountNum, last4Digit, transAmount = idx
            df_filter = df[(df['accountNumber'] == accountNum) & (df['cardLast4Digits'] == last4Digit) & (df['transactionAmount'] == transAmount)]
            num = df_filter.shape[0]
            transTime = df_filter['transactionDateTime'].values
            transType = df_filter['transactionType'].values
            for i in range(num-1):
                time1 = transTime[i]
                time2 = transTime[i+1]
                delta_time = (time2 - time1) / np.timedelta64(1,'s')
                type1 = transType[i]
                type2 = transType[i+1]
                if delta_time <= 300:
                    if type1==type2=='PURCHASE' or type1==type2=='':
                        df_filter = df_filter[df_filter['transactionDateTime'] != time1]
                    
            df_drop = pd.concat([df_drop, df_filter], axis=0)
        return df_drop
    def remove_duplicate(self,df,df_drop):
        ''' Drop the duplicated transactions.'''
        df_merge = df.merge(df_drop, on=list(df.columns), how='left', indicator=True)
        df = df_merge[df_merge['_merge'] =='left_only'][list(df.columns)]
        return df
    def over_limit(self, df):
        df['over_limit'] = (df['currentBalance']>df['creditLimit']).astype(float)
        return df
        
    def CVV_match(self, df):
        ''' Check if the column 'cardCVV' matches the column 'enteredCVV' '''
        df['CVV_match'] = df['cardCVV'] == df['enteredCVV']
        return df
    def country_match(self,df):
        ''' Check if the columns 'acqCountry' and 'merchantCountryCode' match.'''
        df['country_match'] = df['acqCountry'] == df['merchantCountryCode']
        return df 
    def add_days_features(self,df):
        ''' Add the feature 'days_open' by subtracting the column 'transactionDateTime' and 
            'accountOpenDate', add feature 'days_address_change' by by subtracting the column
            'transactionDateTime' and 'dateOfLastAddressChange', and feature 'days_expire' by
            by subtracting the column 'currentExpDate' and 'transactionDateTime'.
        '''
        df['days_open'] = df['transactionDateTime'] - df['accountOpenDate']
        df['days_open'] = df['days_open']/np.timedelta64(1, 'D')
        df['days_address_change'] = df['transactionDateTime'] - df['dateOfLastAddressChange']
        df['days_address_change'] = df['days_address_change']/np.timedelta64(1, 'D')
        df['days_expire'] = df['currentExpDate'] - df['transactionDateTime']
        df['days_expire'] = df['days_expire']/np.timedelta64(1, 'D')
        return df
    def clean_categorical_variables(self, df):
        ''' Clean the categorical variables in the dataset and create dummy variables.
            In the 'acqCountry' column, some filds don't a country input(''). They will be replace with
            'No_Country'. Do the same for 'merchantCountryCode','posConditionCode' and 'posEntryMode' and
            'transactionType'
        '''
        df['acqCountry'] = df['acqCountry'].replace('','No_Country').astype("category")
        country_dummy_variables = pd.get_dummies(df['acqCountry'])
        df['merchantCategoryCode'] = df['merchantCategoryCode'].astype("category")
        merchantCategoryCode_dummy_variables = pd.get_dummies(df['merchantCategoryCode'])
        #df['merchantCountryCode'] = df['merchantCountryCode'].replace('','No_merchantCountryCode').astype("category")
        #merchantCountryCode_dummy_variables = pd.get_dummies(df['merchantCountryCode'])
        df['posConditionCode'] = df['posConditionCode'].replace('','No_posConditionCode').astype("category")
        posConditionCode_dummy_variables = pd.get_dummies(df['posConditionCode'])
        df['posEntryMode'] = df['posEntryMode'].replace('','No_posEntryMode').astype("category")
        posEntryMode_dummy_variables = pd.get_dummies(df['posEntryMode'])
        df['transactionType'] = df['transactionType'].replace('','No_transactionType').astype("category")
        transactionType_dummy_variables = pd.get_dummies(df['transactionType'])
        df['creditLimit'] = df['creditLimit'].astype("category")
        creditLimit_dummy_variables = pd.get_dummies(df['creditLimit'])
        df_all = pd.concat([df, country_dummy_variables, merchantCategoryCode_dummy_variables,\
                            posConditionCode_dummy_variables,\
                            posEntryMode_dummy_variables, transactionType_dummy_variables ], axis=1)
        return df_all
  
    def transaction_freq(self, df):
        '''Calculate the average transaction amount and number of transactions per day in the past week and the past 4 weeks.
        And find the deviation of the current amount from the mean.
        '''
        df['transactionDate'] = df['transactionDateTime'].astype('datetime64[D]')
        grouped_data = df.groupby(['accountNumber','cardLast4Digits','transactionDate'])['transactionAmount']
        cnt_per_day = grouped_data.count()
        total_amount_per_day = grouped_data.sum()
    
        amount_dev_from_avg_list_1_week = []
        amount_dev_from_avg_list_4_weeks = []
        cnt_dev_from_avg_list_1_week = []
        cnt_dev_from_avg_list_4_weeks = []
        
        d_cnt_1_week = {}
        d_amount_1_week = {}
        
        d_cnt_4_week = {}
        d_amount_4_week = {}
        
        for index, row in df.iterrows():
            current_amount = row['transactionAmount']
            current_account = row['accountNumber']
            current_4digits = row['cardLast4Digits']
            current_date = row['transactionDate']
    
            cnt_1_week = 0
            amount_1_week = 0
            days_transaction_1_week = 0
            cnt_4_week = 0
            amount_4_week = 0
            days_transaction_4_week = 0
            
            key_ = str(current_account)+'-'+str(current_4digits)+'-'+str(current_date)
            current_cnt = cnt_per_day[current_account][current_4digits][current_date]
            
            if key_ in d_amount_1_week:
                avg_amt_1_week = d_amount_1_week[key_]
                avg_amt_4_week = d_amount_4_week[key_]
                avg_cnt_1_week = d_cnt_1_week[key_]
                avg_cnt_4_week = d_cnt_4_week[key_]
                
                cnt_dev_from_avg_list_1_week.append(current_cnt - avg_cnt_1_week)
                amount_dev_from_avg_list_1_week.append(current_amount - avg_amt_1_week)
                
                cnt_dev_from_avg_list_4_weeks.append(current_cnt - avg_cnt_4_week)                                    
                amount_dev_from_avg_list_4_weeks.append(current_amount - avg_amt_4_week)
                
            else:
            
                for i in range(1,29):
                    i_days_ago = current_date - np.timedelta64(i,'D')
    
                    try:
    
                        cnt = cnt_per_day[current_account][current_4digits][i_days_ago]
                        amount = total_amount_per_day[current_account][current_4digits][i_days_ago]
    
                        cnt_4_week += cnt
                        amount_4_week += amount
                        days_transaction_4_week += 1
                        if i <8:
                            cnt_1_week += cnt
                            amount_1_week += amount
                            days_transaction_1_week += 1
                    except:
                        pass
                 
                if cnt_1_week>0:
                    avg_cnt_1_week = cnt_1_week/days_transaction_1_week
                    avg_amt_1_week = amount_1_week/cnt_1_week
                    cnt_dev_from_avg_list_1_week.append(current_cnt - avg_cnt_1_week)
                    amount_dev_from_avg_list_1_week.append(current_amount - avg_amt_1_week)
                    d_cnt_1_week[key_] = avg_cnt_1_week
                    d_amount_1_week[key_] = avg_amt_1_week
                else:
                    cnt_dev_from_avg_list_1_week.append(0)
                    amount_dev_from_avg_list_1_week.append(0)
                    d_cnt_1_week[key_] = 0
                    d_amount_1_week[key_] = 0
    
                if cnt_4_week>0 :
                    avg_cnt_4_week = cnt_4_week/days_transaction_4_week
                    avg_amt_4_week = amount_4_week/cnt_4_week
                    cnt_dev_from_avg_list_4_weeks.append(current_cnt - avg_cnt_4_week )                                    
                    amount_dev_from_avg_list_4_weeks.append(current_amount - avg_amt_4_week )
                    d_cnt_4_week[key_] = avg_cnt_4_week
                    d_amount_4_week[key_] = avg_amt_4_week
                else:
                    cnt_dev_from_avg_list_4_weeks.append(0)
                    amount_dev_from_avg_list_4_weeks.append(0)
                    d_cnt_4_week[key_] = 0
                    d_amount_4_week[key_] = 0
            
        
        df.assign(dev_mean_1_week = amount_dev_from_avg_list_1_week)
        
        df.assign(dev_mean_4_weeks = amount_dev_from_avg_list_4_weeks)
       
        df.assign(cnt_dev_1_week = cnt_dev_from_avg_list_1_week)
        
        df.assign(cnt_dev_4_weeks = cnt_dev_from_avg_list_4_weeks)
        return df
    def bool_to_int(self, df):
        ''' Convert the boolean columns into float.'''
        df['cardPresent'] = df['cardPresent'].astype(float)
        df['CVV_match'] = df['CVV_match'].astype(float)
        df['country_match'] = df['country_match'].astype(float)
        df['expirationDateKeyInMatch'] = df['expirationDateKeyInMatch'].astype(float)
        df['isFraud'] = df['isFraud'].astype(float)
        return df 
    def merchantName_cl(self, df):
        '''Set the fields to 0 if the merchantName has no fraud.'''
        merchantName_list = df['merchantName'].unique()
        merchantName_Fraud_list = df[df['isFraud']== 1]['merchantName'].unique()
        def merchantName_clean(x):
            if x in merchantName_list and x not in merchantName_Fraud_list:
                return 0
            else:
                return 1
        df['merchantname_cl'] = df['merchantName'].apply(merchantName_clean)
        return df
    def delete_columns(self,df):
        ''' Drop off columns not needed.'''
        delete_list = ['accountNumber', 'accountOpenDate', 'acqCountry', 'cardCVV',\
                       'cardLast4Digits', 'creditLimit', 'currentExpDate', 'dateOfLastAddressChange',\
                       'enteredCVV','merchantCategoryCode','merchantCountryCode','merchantName',\
                       'posConditionCode', 'posEntryMode','transactionDateTime', 'transactionType',\
                       'transactionDate']
        df = df.drop(delete_list, axis=1)
        return df
 
if __name__ == '__main__':
    download_link = "https://github.com/CapitalOneRecruiting/DS/raw/master/transactions.zip"
    cardData = CardTransactionData(download_link)
    file_extracted = cardData.get_data()
    df = cardData.read_data(file_extracted, 0.2)
    df_datetime = cardData.datetime_transformation(df)
    df_duplicate_group = df_datetime.groupby(['accountNumber','cardLast4Digits','transactionAmount'])['isFraud'].count()
    df_duplicate_group = df_duplicate_group[df_duplicate_group>=2]
    df_dup = cardData.duplicate_identify(df_datetime, df_duplicate_group)
    df_remove_dup = cardData.remove_duplicate(df_datetime, df_dup)
    df_drop = cardData.columns_dropoff(df_remove_dup)
    df_over = cardData.over_limit(df_drop)
    df_cvv = cardData.CVV_match(df_over)
    df_country_match = cardData.country_match(df_cvv)
    df_days = cardData.add_days_features(df_country_match)
    df_categorical = cardData.clean_categorical_variables(df_days)
    df_freq = cardData.transaction_freq(df_categorical)
    df_bool = cardData.bool_to_int(df_freq)
    df_cl = cardData.merchantName_cl(df_bool)
    df_delete = cardData.delete_columns(df_cl)
    df_delete.to_csv('card_transaction.csv',index = False)
    
        
        
    
    
        