"""
Simple pipeline for ML projects
Author: Quinn Underriner
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set()

import sklearn
from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn import metrics
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import svm
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from sklearn import ensemble 
from sklearn import neighbors
from sklearn.grid_search import ParameterGrid
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy import optimize
#dictionary of models and parameters - inspiration from 
#https://github.com/rayidghani/magicloops/blob/master/simpleloop.py

MODELS = {
    'decision_tree': tree.DecisionTreeClassifier(),
    'logistic_regression': linear_model.LogisticRegression(), 
    'knn': neighbors.KNeighborsClassifier(),
    'random_forest': ensemble.RandomForestClassifier(), 
    'support_vector_machine': svm.SVC(), 
    'boosting': ensemble.AdaBoostClassifier(),
    'bagging': ensemble.BaggingClassifier()
}

PARAMS = {
    'decision_tree': {'max_depth': [1, 3, 5, 8, 20]},
    'logistic_regression': {'C': [0.001,0.01,0.1,1,10]}, 
    'knn': {'n_neighbors': [5, 10, 25] },
    'random_forest': {'n_estimators': [1, 2, 3,4, 10]}, 
    'support_vector_machine': {'c_values': [10**-2, 10**-1, 1 , 10, 10**2]}, 
    'boosting': {'n_estimators': [100, 50, 30]},
    'bagging': {'n_estimators': [2, 10, 20]} 
}

ID_COLS = [
    'projectid', 
    'date_posted', 
    'datefullyfunded',
    'predvar', 
    'days_2_fund']



def load_data(filename):
    """
    This function loads the dataset from a CSV 
    """
    df = pd.read_csv(filename)
    return df

"""
functions for exploration 
"""
def per_zip(df, col_name, zip_col_name):
    """
    Returns the averages a each given column per zip code 
    Inputs:
        df: dataframe
        col_name (str): column name you want to target
        zip_col_name (str): name of column in dataframe with zipcode info 
    """
    df_new = df[[col_name, zip_col_name]]
    per_z = df_new.groupby((zip_col_name)).mean().reset_index()
    per_z = per_z.round(2)
    per_z = per_z.rename(index=str, columns={0: "Count of " + col_name})
    return per_z 

def percentage_calc(df, col_name, value):
    """
    Prints percent of data over a certain threshold. 
    inputs:
        df: dataframe 
        col_name (str): column to do this for. 
        value (int): threshold to go over 
    """
    print(len(df[df[col_name]>value]) / len(df))

def find_corr(df):
    """
    Generates a heatmap of correlations 
    Inputs:
        df: dataframe 
    """
    corr = df.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns)

def describe_cols(data):
    colz = list(data.columns)
    for i in colz:
        print(data[i].describe())

"""
functions for preprocessing
"""

def impute_median(df, col):
    """
    This function takes null values in a column and fills it either with the median value
    for the column or the mean, depending on the input
    inputs 
    """
   
    df[col] = df[col].fillna(df[col].median())
    return df

def impute_mean(df, col):
    """
    This function takes null values in a column and fills it either with the median value
    for the column or the mean, depending on the input
    inputs 
    """

    df[col] = df[col].fillna(df[col].mean())
   
    return df


"""
feature generation
"""


def discretize(df, colname):
    """
    This function discretizes a continuous variable (using quartiles)
    Inputs:
        df (dataframe)
        colname (str) name of column to discretize 
    """
    df[colname] = pd.qcut(df[colname], 4, labels=[1, 2, 3, 4])
    return df


def dummy(df, colname):
    """
    Takes a categorical variable and creates binary/dummy variables
    Inputs:
        df (dataframe)
        colname (str) name of column to make dummys  
    """
    dummies = pd.get_dummies(df[colname]).rename(columns=lambda x: colname + "_" + str(x))
    df = pd.concat([df, dummies], axis=1)
    df = df.drop([colname], axis=1)
    return df


def get_xy(df, response, features):

    """
    Create data arrays for the X and Y values needed to be plugged into the model
    Inputs:
        df (dataframe) - the dataframe 
        response (str - the y value for the model 
        features (list of strings) - the x values for the model 
    """ 
    y = df[response].to_numpy()
    X = df[features].to_numpy()
    return X, y


def classify_lgreg(X, y):
    """
    Builds and returns trained logistic regression model
    Inputs:
        X (array) - Features for the model
        y (array) - What is being predicted

    """
    lg = LogisticRegression(penalty="l2")
    lg.fit(X, y)
    return lg

def choose_features(data, response, features):
    
    #delete below before shipping 

    X,y = get_xy(data, response, features)

    sel = sklearn.feature_selection.SelectFromModel(sklearn.ensemble.RandomForestClassifier(n_estimators = 100))
    sel.fit(X, y)

    chosen_features = data[features].columns[sel.get_support()]
    return chosen_features
###
###new more modular classification 
def classify(X, y, model_type, parameters):
    """
    Builds and returns trained logistic regression model
    Inputs:
        X (array) - Features for the model
        y (array) - What is being predicted

    """

    LogisticRegression

    lg = model_type(parameters)
    lg.fit(X, y)
    return lg
    #Logistic Regression, K-Nearest Neighbor, Decision Trees, SVM, Random Forests, Boosting, and Bagging

def accuracy(true_values, predicted_values):
    """
    Computes the accuracy of the prediction
    Inputs:
        true_values (array) actual predicted values from data set
        predicted_values (array) what our model predicted 
    """

    return np.mean(true_values == predicted_values)


def temporal_train_test_split(df, date_col, freq='6MS', gap_days=60):
    """
    produce six month interval splits of data
    inputs:
        df: dataframe
        date_col: column that has dates
    returns:
        list of dates
    """
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    dates = pd.date_range(start=min_date, end=max_date, freq=freq)[1:]
    train_start = min_date
    splits = []
    for i, d in enumerate(dates[:-1]):
        splits.append([[train_start, d], [d+timedelta(days=gap_days), dates[i+1]+timedelta(days=gap_days)]])
    splits.append([[train_start, dates[-1]], [dates[-1]+timedelta(days=gap_days), max_date]])
    return splits
    #currently last append will add dates i cdont have 

def split_df_by_time(df, date_col, train, test):
    """
    create training/testing splits of data, returns df 

    """
    train_df = df[(df[date_col]>=train[0])&(df[date_col]<train[1])]
    test_df = df[(df[date_col]>=test[0])&(df[date_col]<=test[1])]
    return train_df, test_df


def split_data_by_time(df, date_col, train, test, response, features):
    """
    create training/testing splits of data, returns matrix 
    """
    train_df = df[(df[date_col]>=train[0])&(df[date_col]<train[1])]
    test_df = df[(df[date_col]>=test[0])&(df[date_col]<=test[1])]
    X_train, y_train = get_xy(train_df, response, features)
    X_test, y_test = get_xy(test_df, response, features)
    return X_train, X_test, y_train, y_test

#the three fucntions below here were taken from https://github.com/dssg/
#MLforPublicPolicy/blob/master/labs/2019/lab3_lr_svm_eval_sol.ipynb
def calculate_precision_at_threshold(predicted_scores, true_labels, threshold):
    """
    calculatesd recall score
        inputs:
            predicted_scores
            true_labels
            threshold
    """
    pred_label = [1 if x > threshold else 0 for x in predicted_scores]
    _, false_positive, _, true_positives = confusion_matrix(true_labels, pred_label).ravel()
    return 1.0 * true_positives / (false_positive + true_positives)

def calculate_recall_at_threshold(predicted_scores, true_labels, threshold):
    """
    calculatesd recall score
      inputs:
        predicted_scores
        true_labels
        threshold
    """
    pred_label = [1 if x > threshold else 0 for x in predicted_scores]
    _, _, false_negatives, true_positives = confusion_matrix(true_labels, pred_label).ravel()
    return 1.0 * true_positives / (false_negatives + true_positives)

def plot_precision_recall_k(predicted_scores, true_labels):
    """
    plots precision/recall curve
    inputs:
        predicted_scores
        true_labels
    """
    precision, recall, thresholds = precision_recall_curve(true_labels, predicted_scores)
    plt.plot(recall, precision, marker='.')
    plt.show()


def calculate_precision_at_threshold_multi(predicted_scores, true_labels, thresholds):
    """
    calculatesd precision score for multiple thresholds
      inputs:
        predicted_scores
        true_labels
    """  
    z = []
    for i in thresholds:
        z.append(calculate_precision_at_threshold(predicted_scores, true_labels, i))
    return z 

def calculate_recall_at_threshold_multi(predicted_scores, true_labels, thresholds):
    """
    calculatesd recall score for multiple thresholds
      inputs:
        predicted_scores
        true_labels
    """
    z = []
    for i in thresholds:
        z.append(calculate_recall_at_threshold(predicted_scores, true_labels, i))
    return z 

def trim_dummies(df, dummies_to_trim, k):
    for col in dummies_to_trim:
        topk = list(df[col].value_counts()[:k].reset_index()['index'])
        df.loc[:, col] = df[col].apply(lambda x: x if x in topk else 'other')
    return df


def process(data, dummy_list=None, discrete_list=None, impute_median_list=None, impute_mean_list=None, k=30):
    """
    discretize data, make dummies, impute mean and median 
    """
    

    for i in impute_median_list:
        data = impute_median(data, i)

    for i in impute_mean_list:
        data = impute_mean(data, i)

    data = data.dropna()

    d = data[dummy_list].nunique().reset_index()
    d.columns = ['col', 'cnt']
    dummies_to_trim = list(d[d.cnt>k]['col'])
    data = trim_dummies(data, dummies_to_trim, k)
    for i in dummy_list:
        data = dummy(data, i)

    for j in discrete_list:
        data = discretize(data, j)

    return data 

def preprocess(data, drop_list):
    
    #drop values that wont be needed for prediction. Dropping location values here even though they likley
    #would help prediction because policy wise, we can't reconmond that schools move
    data = data.drop(drop_list, axis=1)
    data = data.replace({"t": 1, "f": 0})
    data['date_posted'] = pd.to_datetime(data['date_posted'], format='%m/%d/%y')
    data['datefullyfunded'] = pd.to_datetime(data['datefullyfunded'], format='%m/%d/%y')
    start_date = '2011-01-01'
    end = '2013-12-31'
    data["days_2_fund"] = data['datefullyfunded'] - data['date_posted']
    data["days_2_fund"] = data["days_2_fund"].dt.days
    data['predvar'] = np.where(data['days_2_fund']<= 60, 0, 1) 
    
    return data



#inspiration for below from https://github.com/rayidghani/magicloops/blob/master/simpleloop.py
def run_the_models(data, models_to_run, response, features, dummy_list, discrete_list, impute_median_list, impute_mean_list):
    """
    This runs models and produces evaluation output:
    inputs:
        data: dataframe with data
        models_to_run: list of models to run 
        response: column name of y variable
        features: list of column names for model features 
    returns:
        dataframe 
    """
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    precision_cols = ["precision_at_{}".format(str(x)) for x in thresholds]
    recall_cols = ["recall_at_{}".format(str(x)) for x in thresholds]
    cols = ['model',
            'parameters',
            'train_start',
            'train_end',
            'test_start',
            'test_end',
            'f1_score',
            'auc'] + precision_cols + recall_cols
    model_results = []
    # feature selection
    processed_data = process(data, dummy_list, discrete_list, impute_median_list, impute_mean_list)
    features = [x for x in processed_data.columns if x not in ID_COLS]
    features = choose_features(processed_data, response, features)

    splits = temporal_train_test_split(data, 'date_posted', freq='6M')
    for train, test in splits:
        train_df, test_df = split_df_by_time(data, 'date_posted', train, test)
        train_df = process(train_df, dummy_list, discrete_list, impute_median_list, impute_mean_list)
        test_df = process(test_df, dummy_list, discrete_list, impute_median_list, impute_mean_list)
        X_train, y_train = get_xy(train_df, response, features)
        X_test, y_test = get_xy(test_df, response, features)
        
        for m in models_to_run:
            if m not in MODELS:
                print(m, 'bad model')
                break
            clf = MODELS[m]
            parameter_grid = ParameterGrid(PARAMS[m])
            for p in parameter_grid:
                try:
                    # initialize list to keep track of results
                    res = [m, p, train[0], train[1], test[0], test[1]]
                    clf.set_params(**p)
                    clf.fit(X_train, y_train)
                    predicted_scores = clf.predict_proba(X_test)[:,1]
                    predicted_vals = clf.predict(X_test)
                    true_labels = y_test
                    precise = calculate_precision_at_threshold_multi(predicted_scores, true_labels, thresholds)
                    recall = calculate_recall_at_threshold_multi(predicted_scores, true_labels, thresholds)
                    auc = sklearn.metrics.roc_auc_score(true_labels, predicted_vals)
                    f1 = sklearn.metrics.f1_score(true_labels, predicted_vals)
                    # append metrics to list
                    res = res + [auc, f1] + precise + recall 
                    model_results.append(res)
                    plot_precision_recall_n(true_labels, predicted_vals, m)
                except Exception as e:
                    print(e, m, p)
        df = pd.DataFrame(model_results, columns = cols)
    return df


def choose_model(df, metric_col, asc=False):
    """
    picks best model and parameters
    df
    metric_col - eg f1 
    """
    df['parameters'] = df.parameters.astype(str)
    mean_by_group = df.groupby(['model', 'parameters']).mean().reset_index()
    best_params = mean_by_group.sort_values(metric_col, ascending=asc).drop_duplicates('model')
    best_model = best_params.iloc[0][['model', 'parameters']]
    return best_params, best_model


def plot_precision_recall_n(y_true, y_prob, model_name):
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()



