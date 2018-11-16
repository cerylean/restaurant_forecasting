import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from ml_class import Forecaster


def plot_cv_indices(cv, X, y, all_dates, ax, n_splits, lw=15):

    # Generate the training/testing visualizations for each CV split
    final_ii = 0

    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)
        final_ii = ii + .5
        print(final_ii)
        # print(range(len(indices)))
        # print([ii + .5] * len(indices))

    all = np.array([np.nan] * len(all_dates))
    all[range(len(X_train),len(all_dates))] = 1
    all[range(len(X))] = 0
    ax.scatter(range(len(all)), [5.5] * len(all),
               c=all, marker='_', lw=lw, cmap=cmap_data,
               vmin=-.2, vmax=1.2)
    print (range(len(all)))
    print ([-.5]*len(all))
    # Formatting
    yticklabels = [f"Split {n+1}" for n in range(n_splits)] + ['Holdout']
    xticklabels = all_dates[::95]

    ax.set(yticks=np.arange(n_splits+1) + .5,
           # xlabel='Data index',ylabel="CV iteration",
           ylim=[n_splits+1.2, -.2], xlim=[0, len(all_dates)])
    ax.set_yticklabels(yticklabels , fontsize=12)
    ax.set_xticklabels(xticklabels , rotation=45, fontsize=12)
    ax.set_title('\nN Splits = 5', fontsize=12,style='italic')
    fig.suptitle('          Time Series Split', fontsize=15)
    plt.tight_layout()
    plt.savefig('images/time_series_split.png')
    return ax

if __name__ == '__main__':

    app = Forecaster(3,'food|appetizers|22826289')
    app.load_data()
    X_train,X_test,y_train, y_test = app.train_test_split()
    app.holdout_dates()
    fig, ax = plt.subplots()
    cmap_data = plt.cm.bwr
    cmap_cv = plt.cm.coolwarm
    cv = TimeSeriesSplit(n_splits=5)
    plot_cv_indices(cv, X_train,y_train, app.all_dates, ax, 5)
