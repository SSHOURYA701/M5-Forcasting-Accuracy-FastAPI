# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:11:04 2021

@author: SHREYANSH
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pickle

def predicterFunction(test,calendar_df,prices_df):
    # We are creating new features required for the prediction for days from 1942 till 1969
    for day in range(1942,1942+28):
      test['d_' + str(day)] = np.int32(0)

    # We are transforming our Time Series Data so that we can apply supervised ml problem techniques
    data = pd.melt(test, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
              var_name='day', value_name='demand').dropna()  

    ## Merging all the files so that we have them in 1 place to make features
    data = data.merge(calendar_df, left_on='day', right_on='d')
    data = data.merge(prices_df,on=['store_id','item_id', 'wm_yr_wk'], how='left')

    ## Imputing Null in the sell price with the mean of the product id
    data['sell_price'].fillna(data.groupby('id')['sell_price'].transform('mean'), inplace=True)

    #Feature 1 :: we are removing  _ from ex d_101 to get the day number i.e 101
    data['day'] = data['day'].apply(lambda x: x.split('_')[1]).astype(np.int16)

    #since weekday's are represented as wday with numbers and d is a duplicate column.
    data.drop(['d','weekday'], axis=1, inplace=True)

    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    #Imputing Nan
    for feature in nan_features:
        data[feature].fillna('NA', inplace = True)


    ##loading the dict which contains the mappings of the Label Encoders   
    all_label_dicts = pickle.load(open("all_label_dicts.p", "rb"))

    for feature in list(all_label_dicts.keys()):
      feat_dict = all_label_dicts[feature]
      data[feature] = data[feature].map(feat_dict)

    data['lag_28'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
    data['lag_56'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(56))

    data['rolling_mean_7']   = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    data['rolling_std_7']    = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())

    data['rolling_mean_56']  = data.groupby(['id'])['demand'].transform(lambda x: x.shift(56).rolling(7).mean())
    data['rolling_std_56']    = data.groupby(['id'])['demand'].transform(lambda x: x.shift(56).rolling(7).std())

    ## Creating time based features
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['week'] = data['date'].dt.week
    data['day_of_date'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek

    data['days'] = data['day']
    data['day'] = data['day_of_date']

    ## Weekend feature
    def weekend(arg):
        if arg==5 or arg==6:
            return 1
        else:
            return 0
    data['isweekend'] = data['dayofweek'].apply(weekend)

    data.fillna(0,inplace=True)

    best_model = pickle.load(open("best_model.p", "rb"))

    features = ['days','day', 'wm_yr_wk', 'wday', 'month', 'year', 'event_name_1',
       'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX',
       'snap_WI', 'sell_price', 'week',
       'isweekend','lag_28', 'lag_56', 'rolling_mean_7', 'rolling_std_7',
       'rolling_mean_56', 'rolling_std_56']



    # We are splitting the data for validation and test and then predicting it's value
    Val = data[(data['days']>1913) & (data['days']<1942)]
    pred_val_array = best_model.predict(Val[features])

    Test = data[data['days']>1941]
    pred_test_array = best_model.predict(Test[features])
    

    # We are then reshaping the predicted value
    pred_val_array = np.reshape(pred_val_array, (-1, 28),order = 'F')
    pred_test_array = np.reshape(pred_test_array, (-1, 28),order = 'F')

    cols = ['F'+str(i) for i in range(1,29)]

    vals = pd.concat([pd.DataFrame([test['id']], index=[0]),pd.DataFrame(pred_val_array, columns=cols)],axis=1).rename(columns={0:'ID'})
    vals['ID'] = vals['ID'].apply(lambda x: x.replace('evaluation','validation'))
    tst = pd.concat([pd.DataFrame([test['id']], index=[0]),pd.DataFrame(pred_test_array, columns=cols)],axis=1).rename(columns={0:'ID'})

    return vals, tst