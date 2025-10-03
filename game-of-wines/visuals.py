###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


def distribution(data, feature_label, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    sns.set()
    sns.set_style("whitegrid")
    # Create figure
    fig = plt.figure(figsize = (11,5));

    # Skewed feature plotting
    for i, feature in enumerate([feature_label]):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Total Number")
        ax.set_ylim((0, 1500))
        ax.set_yticks([0, 200, 400, 600, 800])
        ax.set_yticklabels([0, 200, 400, 600, 800, ">1000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()


def visualize_classification_performance(results):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - results: a list of dictionaries of the statistic results from 'train_predict_evaluate()'
    """

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    import numpy as np

    sns.set()
    sns.set_style("whitegrid")
    
  
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 10), constrained_layout=True)


    bar_width = 0.3
    colors = ["#e55547", "#4e6e8e", "#2ecc71"]

   
    ax = np.array(ax)
    
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in range(3):
                ax[j//3, j%3].bar(i + k * bar_width, results[learner][i][metric],
                                  width=bar_width, color=colors[k])

            ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
            ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
            ax[j//3, j%3].set_xlabel("Training Set Size")
            ax[j//3, j%3].set_xlim((-0.1, 3.0))


    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")


    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")


    for col in [1, 2]:
        ax[0, col].axhline(y=1, color='k', linestyle='dashed', linewidth=1)
        ax[1, col].axhline(y=1, color='k', linestyle='dashed', linewidth=1)


    for row in [0, 1]:
        for col in [1, 2]:
            ax[row, col].set_ylim((0, 1))


    patches = [mpatches.Patch(color=colors[i], label=learner)
               for i, learner in enumerate(results.keys())]

   
    fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, fontsize='large')


    fig.suptitle("Performance Metrics for Three Supervised Learning Models",
                 fontsize=16, y=1.08)



    plt.show()


def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:11]]
    values = importances[indices][:11]

    sns.set()
    sns.set_style("whitegrid")

    # Creat the plot
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    plt.bar(np.arange(11), values, width = 0.2, align="center", label = "Feature Weight")
    # plt.bar(np.arange(11) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
    #       label = "Cumulative Feature Weight")
    plt.xticks(np.arange(11), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Feature", fontsize = 12)
    
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()  


