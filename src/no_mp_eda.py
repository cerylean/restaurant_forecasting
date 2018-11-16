import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from datetime import datetime, timedelta
from eda import clean_reshape

def clean_reshape(bid):
# Import Main Data and add concat columns
    df = pd.read_csv('CombinedBlindedData.txt',sep=",",parse_dates=['BUSINESS_DAY'])
    df.columns = [x.lower() for x in list(df.columns)]
    df = df[df['bid']==bid]
    df.reset_index(drop = True, inplace = True)
    df.drop(df[df['sky_image_desc'].str.lower()=='not used'].index,inplace = True)
    df.reset_index(drop = True, inplace = True)
    df.drop(df[df['exclude']==1].index,inplace = True)
    df.reset_index(drop = True, inplace = True)
    # df['bus_day'] = df['bid'].astype(str) + "|" + df['business_day'].astype(str)
    df['items'] = df['cat_b'].str.lower() + "|" + df['minor_b'].str.lower() + "|" + df['item_num'].astype(str)

    # file_name1 = 'dblcheck_bid_nomp' + str(bid) + '.csv'
    # df.to_csv(file_name1,index=False)

    # Import & Create Covers DataFrame
    df_covers = pd.read_csv('BlindedCovers2.csv',sep=",",parse_dates=['BUSINESS_DAY'])
    df_covers.columns = [x.lower() for x in list(df_covers.columns)]
    df_covers = df_covers[df_covers['bid']==bid]
    # df_covers['bus_day'] = df_covers['bid'].astype(str) + "|" + df_covers['business_day'].astype(str)
    df_covers = df_covers.groupby('business_day',as_index=False)['covers'].sum()
    df_covers['dow'] = df_covers['business_day'].dt.day_name()
    dummy_dow = pd.get_dummies(df_covers['dow'].str.lower())
    df_covers['month'] = df_covers['business_day'].dt.month
    dummy_month = pd.get_dummies(df_covers['month'])
    dummy_month.columns = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    df_covers_adj = pd.concat([df_covers,dummy_dow,dummy_month],axis=1)
    df_covers_adj['last_week'] =  df_covers_adj['business_day']-timedelta(days=7)
    df_covers_adj['last_year'] =  df_covers_adj['business_day']-timedelta(days=364)
    df_covers_adj.drop(['dow','month'], axis=1, inplace = True)

    # Create Weather DataFrame
    df_weather = df.copy()[['business_day','hitemp','lowtemp','humidity','precip_inch','sky_image_desc']]
    df_weather.drop_duplicates('business_day',inplace=True)
    df_weather.reset_index(drop = True, inplace = True)
    df_weather['precip_inch'].fillna(0,inplace=True)
    dummy_weather = pd.get_dummies(df_weather['sky_image_desc'])
    df_weather_adj = pd.concat([df_weather,dummy_weather],axis=1)
    df_weather_adj.drop('sky_image_desc', axis=1, inplace = True)
    df_weather_adj.columns = [x.lower() for x in list(df_weather_adj.columns)]

    # Create Items DataFrame
    df_items = df.pivot_table(values=['item_count'], index = ['business_day'], columns = ['items'],aggfunc='sum')
    df_items.columns =  df_items.columns.get_level_values(1)
    df_items.fillna(0,inplace=True)

    # Merging all the DataFrames
    df_merged = pd.merge(df_covers_adj,df_weather_adj,how= 'left', on = 'business_day')
    #merge the dataframe again, joining on the last year and last week columns, rename overlapping column names with lw('Last Week') and ly('Last Year')
    df_merged = pd.merge(df_merged,df_items, how = 'left', left_on = 'last_week',right_on = 'business_day')
    df_merged = pd.merge(df_merged,df_items, how = 'left', left_on = 'last_year',right_on = 'business_day',suffixes = ('lw','ly'))
    df_merged = pd.merge(df_merged,df_items, how= 'left', on = 'business_day')
    df_merged.drop(df_merged[df_merged['covers']==0].index,inplace = True)
    df_merged.reset_index(drop = True, inplace = True)
    file_name = 'no_mp_data_bid' + str(bid) + '.csv'
    df_merged.to_csv(file_name,index=False)

clean_reshape(2)
clean_reshape(3)
clean_reshape(4)
