
## Forecasting Restaurant Items
#### *Predicting item sales at three NYC restaurants*

![Food Pic](images/food.png)

*by Lyssa Treibitz | Galvanize Data Science Immersive | November 2018*
___

### The Data

I partnered with [Avero](http://www.averoinc.com/), a restaurant software solution, to explore a topic that has been of interest to us both: forecasting. They recently released a new product focused on forecasting restaurant covers (the number of guests expected), and wanted to take the forecast a step further, to predict the number of each menu item sold per day.

I received three years of restaurant item sales and daily weather from three restaurants. The data was blinded, so I do not know the names of the restaurants nor the names of the items.  I only know which food or beverage category each item belongs to.

From the data, my model uses the following features to predict that day's item sales:

| Features       | Description      | Type
| ------------- |:-------------:| -----:|
| Covers| # of guests expected| Integer|
| DOW     | the weekday     |   Bool |
| Month     | the month    |   Bool |
| Temperature | high & low |   Integer |
| Precipitation | amount in inches  | Float |
| Sky description  | i.e. 'sunny', 'cloudy'  |  Bool |
| Humidity | percent humidity  |    Integer
| Item Last Year | item count same day last year| Integer |
| Item Last Week | item count same day last week | Integer |

___

### The Method

After experimenting with multiple regression models, I found that the best results derived from a Gradient Boosting Regressor.  

Gradient Boosting's objective is to minimize the loss of the model by adding weak learners one at a time using a gradient descent-like procedure. More specifically, each 'weak learner' represents a decision tree and the loss function to be minimized is Ordinary Least Squares regression.

I was focused on predicting one item at a time.  This means that, while the hyperparameters remained the same, each item was run through its own Gradient Boosting regression with its own unique splits.  

Additionally, since current performance is dependent on past performance, it was important to keep the time series intact. This meant that when looking at cross validation and train test splits, it was necessary to maintain chronology during cross validation.  To do so, I used the Time Series Split cross validation method, in addition to splitting off a final holdout set.  

 ![Time Series Split](images/time_series_split.png)

Given that there are a large number of features, it was important to understand which features had the greatest impact on the model.

 ![Feature Importance](images/feature_importance.png)

 Note that this graph is focused only on **one** specific item.  It is clear that only a handful of features are impacting this item's overall predictive outcome. This is led by number of covers, followed by last year and last week's item performance. Since other items might depend on some of the less important features here to account for things like time dependencies or seasonality (i.e. a weekly special or a fall menu offering), I decided to leave all the features in place for now and proceed.  
 ___

 ### The Results

 So far the results have varied greatly from item to item.  Ranging from the best case:

  ![Best Case](images/best_case.png)

To the far from ideal case:

  ![Worst Case](images/worst_case.png)

It's hard to say exactly what is going on here without a bit more context. Perhaps this item is a "daily special" button in the POS that is used for different things on different days.  Perhaps the kitchen ran out of stock of a specific item.  Maybe the server who often recommends that item took a few days off.

Below are the model's results for the top ten most popular items in my holdout set time period for one restaurant. To understand the success of the results, I look at RMSE: the spread of the predicted  vs actual results translated back into the metric of item count. The best case appetizer (first in the list) predicted on average +/- 10 items from the mean of 51 items sold. Overall, the results vary drastically from item to item and none achieve an RMSE variance below 20%.

![Other Results](images/other_results.png)

Focusing back on the positive, I wanted to get a better understanding of whether the predictions I found for our best case scenario outperformed, say, simply predicting what was ordered the week before.

![Four Graph](images/four_graphs6.png)

This plots the graphs against the same day last year, last week, and against Avero's current prediction method which takes the weighted average of the item count/cover for the same day of the last four weeks and multiplies it by the predicted covers for that day.

As you can see, simply predicting item counts from last week achieved the best results with an RMSE of 3. At least for this **one** item, for this **specific** holdout time period.


 ### Thank U, Next

 Moving forward to Capstone III, I will explore an LSTM recurrent neural net and hierarchical linear regression to see if I can improve upon these results with a goal of building a one-size-fits-all model that can output multiple menu items.  I will also look more into feature reduction as I suspect certain features will not impact the model outcome for any item.


####  References
https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/
