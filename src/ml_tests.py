import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,  GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from eda import get_cat
# import matplotlib as mpl
# mpl.rcParams.update({
#     'font.size'           : 20.0,
#     'axes.titlesize'      : 'large',
#     'axes.labelsize'      : 'medium',
#     'xtick.labelsize'     : 'medium',
#     'ytick.labelsize'     : 'medium',
#     'legend.fontsize'     : 'large',
#     'legend.loc'          : 'upper right'
# })


def df_filter(merged_df,cat_keyword):
    # merged_df = merged_df[merged_df['bid']==bid_to_filter]
    df = merged_df.copy()
    orig_columns = list(np.arange(0,45,1))
    columns = orig_columns + [idx for idx, col in enumerate(list(df.columns)) if cat_keyword in col]
    df = df.iloc[:,columns]
    df.drop(df[df['business_day'] < (min(df['business_day'])+timedelta(days=364))].index,inplace = True)
    df.reset_index(drop = True, inplace = True)
    df.sort_values(by='business_day',ascending= True)
    df.fillna(0,inplace=True)
    df.drop(['business_day','last_week','last_year'],axis=1,inplace=True)
    return df

def gen_score(regressor,train_test, **kwargs):
    tscv = TimeSeriesSplit(n_splits=5)
    X_train, X_test, y_train, y_test = train_test
    my_cv = tscv.split(X_train,y_train)
    scores = []
    reg = regressor(**kwargs)
    for train_index, test_index in my_cv:
        X_train1, X_test1 = X_train[train_index], X_train[test_index]
        y_train1, y_test1 = y_train[train_index], y_train[test_index]
        reg.fit(X_train,y_train)
        scores.append(reg.score(X_test1, y_test1))
    name = reg.__class__.__name__
    r2 = round(np.mean(scores),3)
    # print("{0:<20s} | R2: {1:0.3f}".format(name,r2))
    return r2, reg

def create_holdout(merged_df, holdout_pct):
    split_index = int(len(merged_df)*(1-holdout_pct))
    holdout_df = merged_df.iloc[split_index:,:]
    validation_df = merged_df.iloc[:split_index,:]
    # holdout_df.to_csv('holdout.csv',index=False)
    # validation_df.to_csv('validation.csv',index=False)
    return validation_df , holdout_df

def item_test(bid,item_name,holdout_pct=.025):
    file_name = 'no_mp_data_bid' + str(bid) + ".csv"
    df_merged = pd.read_csv(file_name,parse_dates=['business_day'])
    df_filt = df_filter(df_merged,item_name)
    validation_df , holdout_df = create_holdout(df_filt, holdout_pct)
    y_train = validation_df.pop(item_name)
    X_train = validation_df.values
    y_test = holdout_df.pop(item_name)
    X_test = holdout_df.values
    return X_train, X_test, y_train, y_test ,holdout_df

def run_multiple_items(bid,item_list):
    tscv = TimeSeriesSplit(n_splits=5)
    for item in item_list:
        X_train, X_test, y_train, y_test = item_test(bid,item,.025)
        train_test_splits = [X_train, X_test, y_train, y_test]
        r2, reg = gen_score(GradientBoostingRegressor, train_test_splits, n_estimators = 1000, min_samples_split = 10,random_state=1000)
        y_pred = reg.predict(X_test)
        rmse = round((np.sum((y_pred - y_test)**2)/len(y_test))**.5)
        perc = round(rmse / (np.mean(y_test)+.00000000001),1)
        # print(f"Business:{bid}  |  {item:<30s} | GB R^2: {r2}  |  Holdout R^2: {round(reg.score(X_test, y_test),3)}")
        print(f"Business:{bid}  |  {item:<30s} | RMSE: {rmse}  | Mean: {round(np.mean(y_test))} | Perc: {round(perc*100,1)}%")
        pred_plot (y_pred, y_test, item,bid,rmse,perc)

def pred_plot(y_pred,y_test,item,bid,rmse,perc):
    file_name = 'no_mp_data_bid' + str(bid) + ".csv"
    df_merged = pd.read_csv(file_name,parse_dates=['business_day'])
    df_merged.drop(df_merged[df_merged['business_day'] < (min(df_merged['business_day'])+timedelta(days=364))].index,inplace = True)
    df_merged.reset_index(drop = True, inplace = True)
    validation_df , holdout_df = create_holdout(df_merged, .025)
    x = list(holdout_df['business_day'].dt.strftime("%m/%d"))
    ly_col = item + 'ly'
    plt.plot(x,x_test[ly_col])
    plt.plot(x, y_pred,color='green',linestyle ='--', label = 'Predicted')
    plt.plot(x, y_test,color ='blue', label = 'Actual')
    plt.xticks(rotation=45)
    plt.ylabel('Item Count')
    plt.xlabel('Dates')
    plt.title(f"{item.upper()}\nRMSE: {round(rmse)} | Mean: {round(np.mean(y_test))} | Perc: {perc*100}%", fontweight = 'bold')
    plt.legend()
    # plt.tight_layout()
    plt.show()
    # file_name = "bid" + str(bid) + "_" + item +".png"
    # plt.savefig(file_name)
if __name__ == '__main__':

    # df_merged = pd.read_csv('reformatted_data_bid4.csv',parse_dates=['business_day'])
    # df_filt = df_filter(df_merged,'apps|apps|18207704')
    # # df_filt.to_csv('app_18207704_data.csv',index=False)
    # validation_df , holdout_df = create_holdout(df_filt, .1)
    #
    # train_y = validation_df.pop('apps|apps|18207704')
    # train_X = validation_df.values

    X_train, X_test, y_train, y_test , holdout_df= item_test(3,'food|appetizers|22826289')
    # train_test_splits = [X_train, X_test, y_train, y_test]
    # tscv = TimeSeriesSplit(n_splits=5)
    #
    # rf = gen_score(RandomForestRegressor,train_test_splits,n_estimators = 100, max_depth = 3,min_samples_split = 10, random_state=1000, max_features = 'log2')
    # r2, gb = gen_score(GradientBoostingRegressor, train_test_splits, n_estimators = 100, min_samples_split = 10,random_state=1000)
    # ab = gen_score(AdaBoostRegressor, train_test_splits, n_estimators = 100, learning_rate = .5,random_state=1000)
    # lr = gen_score(LinearRegression, train_test_splits)
    # #
    # y_pred = gb.predict(X_test)
    # rmse = (np.sum((y_pred - y_test)**2)/len(y_test))**.5
    # gb.score(X_test, y_test)

    # cat4 = get_cat(4,'app',1500)
    # cat3 = get_cat(3,'app',2200)
    # cat2 = get_cat(2,'beef',1200)
    #
    # run_multiple_items(2, cat2)
    # run_multiple_items(3, cat3)
    # run_multiple_items(4, cat4)
