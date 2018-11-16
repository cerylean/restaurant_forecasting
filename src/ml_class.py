import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,  GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
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

class Forecaster(object):

    def __init__(self, bid, item_name):
        self.bid = bid
        self.item_name = item_name
        self.df_loaded = None
        self.df_filtered = None
        self.df_prepped = None
        self.holdout_pct = None
        self.validation_df = None
        self.holdout_df = None
        self.x_dates = None
        self.rmse = None
        self.score = None
        self.name = None


    def load_data(self):
        '''
        Loads data from file based on the business id

        INPUT:  None

        OUTPUT: Loaded DataFrame (df)

        '''
        file_name = 'no_mp_data_bid' + str(self.bid) + ".csv"
        self.df_loaded = pd.read_csv(file_name,parse_dates=['business_day'])
        return self.df_loaded

    def filter_sort(self):
        '''
        Filters the dataframe to only include the instantiated item
        Drops the first year of data to ensure each data point has yoy data
        Sorts the whole dataframe in chronological ascending order

        INPUT:  None

        OUTPUT: Filtered DataFrame (df)

        '''
        df = self.load_data().copy()
        last_col_index = list(df.columns).index('sunny')
        orig_columns = list(np.arange(0,last_col_index+1,1))
        columns = orig_columns + [idx for idx, col in enumerate(list(df.columns)) if self.item_name in col]
        df = df.iloc[:,columns]
        df.drop(df[df['business_day'] < (min(df['business_day'])+timedelta(days=364))].index,inplace = True)
        df.reset_index(drop = True, inplace = True)
        df.sort_values(by='business_day',ascending= True)
        self.df_filtered = df.fillna(0)
        return self.df_filtered

    def df_prep(self):
        '''
        Drops the date columns to prep for model input

        INPUT:  None

        OUTPUT: Prepped DataFrame (df)

        '''
        df = self.filter_sort()
        self.df_prepped = df.drop(['business_day','last_week','last_year'],axis=1)
        return self.df_prepped

    def create_holdout(self,holdout_pct, send_to_csv = False):
        '''
        Splits off a holdout data set based on a specified percentage

        INPUT:  Holdout Percentage (float)
                Send to CSV (bool)

        OUTPUT: Validation Dataframe (df)
                Holdout Dataframe (df)

        '''
        self.holdout_pct = holdout_pct
        split_index = int(len(self.df_prepped)*(1-self.holdout_pct))
        self.holdout_df = self.df_prepped.iloc[split_index:,:]
        self.validation_df = self.df_prepped.iloc[:split_index,:]
        if send_to_csv:
            holdout_df.to_csv('holdout.csv',index=False)
            validation_df.to_csv('validation.csv',index=False)
        return self.validation_df , self.holdout_df

    def holdout_dates(self):
        '''
        Creates the list of dates from the holdout set

        INPUT:  None

        OUTPUT: Holdout Dates (list)

        '''
        split_index = int(len(self.df_filtered)*(1-self.holdout_pct))
        holdout_df = self.df_filtered.iloc[split_index:,:]
        validation_df = self.df_filtered.iloc[:split_index,:]
        self.chart_val_dates = list(validation_df['business_day'].dt.strftime("%b '%y"))
        self.chart_hold_dates = list(holdout_df['business_day'].dt.strftime("%b '%y"))
        self.all_dates = self.chart_val_dates + self.chart_hold_dates
        self.x_dates = list(holdout_df['business_day'].dt.strftime("%m/%d"))
        return self.x_dates

    def train_test_split(self,holdout_pct=.025):

        '''
        Creates a train test split from the validation and holdout data sets

        INPUT:  Holdout Percentage (float)

        OUTPUT: X_train (arr)
                X_test (arr)
                y_train (arr)
                y_test (arr)

        '''
        self.df_prep()
        self.holdout_pct = holdout_pct
        self.validation_df , self.holdout_df = self.create_holdout(self.holdout_pct)
        y_train = self.validation_df.pop(self.item_name)
        X_train = self.validation_df.values
        y_test = self.holdout_df.pop(self.item_name)
        X_test = self.holdout_df.values
        return X_train, X_test, y_train, y_test

    def cv_fit(self,n_splits, X_train,_y_train,regressor, **kwargs):
        '''
        Cross validates the data with a Time Series Split
        Fits the training data for each split and finds the average score


        INPUT:  # of Time Series splits (int)
                X_train (arr)
                y_train (arr)
                Regression Model (reg)
                Kwargs (varies depending on model)

        OUTPUT: None

        '''
        tscv = TimeSeriesSplit(n_splits=n_splits)
        my_cv = tscv.split(X_train,y_train)
        scores = []
        reg = regressor(**kwargs)
        for train_index, test_index in my_cv:
            X_val_train, X_val_test = X_train[train_index], X_train[test_index]
            y_val_train, y_val_test = y_train[train_index], y_train[test_index]
            reg.fit(X_val_train,y_val_train)
            scores.append(reg.score(X_val_test, y_val_test))
        self.name = reg.__class__.__name__
        self.score = round(np.mean(scores),3)
        self.reg = reg


    def predict(self,X_test):
        self.y_pred = self.reg.predict(X_test)
        return self.y_pred

    def calc_rmse(self,X_test,y_test):
        self.rmse = round((np.sum((self.predict(X_test) - y_test)**2)/len(y_test))**.5)
        self.perc = round(self.rmse / (np.mean(y_test)),1)
        return self.rmse

    def pred_plot(self, X_test,y_test, last_week=False, last_year=False, send_to_png = False):
        '''
        Plots the predicted values against the actual


        INPUT:  X_test (arr)
                y_test (arr)
                Include Same Day Last Week data (bool)
                Include Same Day Last Year data (bool)
                Kwargs (varies depending on model)

        OUTPUT: None

        '''
        x = self.holdout_dates()
        rmse = self.calc_rmse(X_test,y_test)
        y_pred = self.predict(X_test)
        if last_year:
            plt.plot(x,X_test[:,-1],color='red',linestyle =':', label = 'Last Year')
        if last_week:
            plt.plot(x,X_test[:,-2],color='orange',linestyle =':', label = 'Last Week')
        plt.plot(x, y_pred,color='green',linestyle ='--', label = 'Predicted')
        plt.plot(x, y_test,color ='blue', label = 'Actual')
        plt.xticks(rotation=45)
        plt.ylabel('Item Count')
        plt.xlabel('Predicted Dates')
        plt.suptitle(f"{self.item_name.upper()}", fontweight = 'bold')
        plt.title(f"RMSE: {round(rmse)} | Mean: {round(np.mean(y_test))} | Perc: {self.perc*100}%")
        plt.legend()
        if send_to_png:
            file_name = "images/bid" + str(self.bid) + "_" + self.item_name +".png"
            plt.savefig(file_name)
        else:
            plt.show()



if __name__ == '__main__':


    app = Forecaster(3,'food|appetizers|22826289')
    app.load_data()
    X_train,X_test,y_train, y_test = app.train_test_split()
    last_week = X_test[:,-2]
    app.cv_fit(5, X_train, y_train, GradientBoostingRegressor, n_estimators = 1000, min_samples_split = 10,random_state=1000)
    # app.pred_plot(X_test,y_test,last_year=True)
    print (f"RMSE Predicted: {app.calc_rmse(X_test, y_test)}")
    print (f"RMSE Last Week: {round((np.sum((last_week - y_test)**2)/len(y_test))**.5)}")
