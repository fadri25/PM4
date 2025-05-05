
"""
    def save_sample(self, output_path, sample_size=1000):
        #Speichert die zufällige Stichprobe in eine neue CSV-Datei.
        sample = self.get_sample(sample_size)
        if sample is not None:
            sample.to_csv(output_path, index=False)
            print(f"Stichprobe gespeichert in: {output_path}")

# Teständerung
#if __name__ == "__main__":
#sampler.save_sample("sample_1000.csv") 



    def save_sample(self, output_path, sample_size=1000):
        #Speichert die zufällige Stichprobe in eine neue CSV-Datei.
        sample = self.get_sample(sample_size)
        if sample is not None:
            sample.to_csv(output_path, index=False)
            print(f"Stichprobe gespeichert in: {output_path}")

# Teständerung
#if __name__ == "__main__":
#sampler.save_sample("sample_1000.csv") 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import hyperopt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, precision_recall_curve, recall_score, precision_score
from functools import partial
import warnings
from sklearn.model_selection import cross_validate


MIN_PRECISION = 0.05

# The current version of XGBoost uses a conditional statement that
# the current version SciPy (internally used by XGBoost) doesn't like.
# This supresses SciPy's deprecation warning message
warnings.filterwarnings('ignore', category = DeprecationWarning)

df = pd.read_csv('C:/Users/fadri/Documents/SynologyDrive/Drive/Studium/Case Studies 4/Kreditkarten Datenset/transactions.csv', index_col = 0)
print(df.columns)
print(df.head(3))
print(df.info())
missing_values = df.isnull().sum()
missing_values

MIN_PRECISION = 0.05

# The current version of XGBoost uses a conditional statement that
# the current version SciPy doesn't like.
# This supresses SciPy's deprecation warning message
warnings.filterwarnings('ignore', category = DeprecationWarning)

def conditional_recall_score(y_true, pred_proba, min_prec = MIN_PRECISION):
    # Since the PR curve is discreet it might not contain the exact precision value given
    # So we first find the closest existing precision to the given level
    # Then return the highest recall acheiveable at that precision level
    # Taking max() helps in case PR curve is locally flat
    # with multiple recall values for the same precision
    pr, rc,_ = precision_recall_curve(y_true, pred_proba[:,1])
    return np.max(rc[pr >= min_prec])

def objective(params, X, y, X_early_stop, y_early_stop, scorer, n_folds = 10):

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    imbalance_ratio = neg_count / pos_count
    
    xgb_clf = XGBClassifier(**params, scale_pos_weight=imbalance_ratio,
                            n_estimators = 2000, n_jobs = 1)

    xgb_fit_params = {'early_stopping_rounds': 50,
                      'eval_metric': ['logloss'],
                      'eval_set': [(X_early_stop, y_early_stop)],
                      'verbose': False
                      }
    
    cv_results = cross_validate(xgb_clf, X_train, y_train, cv=n_folds,
                                fit_params= xgb_fit_params, n_jobs = -1,
                                scoring = scorer)
    cv_score = np.mean(cv_results['test_score'])


    
    # hypoeropt minimizes the loss, hence the minus sign behind cv_score
    return {'loss': -cv_score, 'status': hyperopt.STATUS_OK, 'params': params}

def tune_xgb(param_space, X_train, y_train, X_early_stop, y_early_stop, n_iter):    
    scorer = make_scorer(conditional_recall_score, needs_proba=True)

    # hyperopt.fmin will only pass the parameter values to objective. So we need to
    # create a partial function to bind the rest of the arguments we want to pass to objective
    obj = partial(objective, scorer = scorer, X = X_train, y = y_train,
                  X_early_stop = X_early_stop, y_early_stop = y_early_stop)

    # A trials object that will store the results of all iterations
    trials = hyperopt.Trials()
    
    hyperopt.fmin(fn = obj, space = param_space, algo = hyperopt.tpe.suggest,
                         max_evals = n_iter, trials = trials)
    
    # returns the values of parameters from the best trial
    return trials.best_trial['result']['params']

def optimal_threshold(estimator, X, y, n_folds = 10, min_prec = 0.05, fit_params = None):
    
    cv_pred_prob = cross_val_predict(estimator, X, y, method='predict_proba',
                                     cv = n_folds, fit_params=fit_params, n_jobs=-1)[:,1]

    # Once again, the PR curve is discreet and may not contain the exact precision level
    # we are looking for. So, we need to find the closest existing precision
    pr, _, threshold = precision_recall_curve(y, cv_pred_prob)
    # precision is always one element longer than threshold and the last one is always set to 1
    # So I drop the last element of precision so I can use it below to index threshold
    pr = pr[:-1]
    return min(threshold[pr >= min_prec])

def thresholded_predict(X, estimator, threshold):
    return np.array([1 if (p >= threshold) else 0 for p in estimator.predict_proba(X)[:,1]])

if __name__ == "__main__":    
    # Loading the data
    data = pd.read_csv('C:/Users/fadri/Documents/SynologyDrive/Drive/Studium/Case Studies 4/Kreditkarten Datenset/transactions.csv')
    X = data.drop('TX_FRAUD_SCENARIO', axis = 1)
    y = data['TX_FRAUD_SCENARIO']
    
    # Train/test split, 80/20, random_state set for reproduibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 1)

    # Further splitting the initial training set so that 10% of all data(1/8 of 80%) 
    # can be used as the evaluation set by XGBoost for early stopping
    X_train, X_early_stop, y_train, y_early_stop = train_test_split(X_train, y_train, stratify = y_train, test_size = 1/8, random_state = 1)
    
    # The prior probability distribution of parameters for Bayesian optimization
    param_space = {
            'learning_rate': hyperopt.hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'max_depth': hyperopt.hp.choice('max_depth', [2, 4, 6, 8, 10]),
            'subsample': hyperopt.hp.uniform('subsample', 0.25, 1),
            'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.7, 1.0),
            'min_child_weight': hyperopt.hp.choice('min_child_weight', [1, 3, 5, 7]),
            'reg_alpha': hyperopt.hp.uniform('reg_alhpa', 0, 1.0),
            # Avoiding lambda = 0. There is a Github issue on strange behaviour with lambda = 0
            'reg_lambda': hyperopt.hp.uniform('reg_lambda', 0.01, 1.0),
            }

    # # # # # # # # #
    # Step 1: Tuning hyper-parameters of the XGBoost classifier
    # # # # # # # # #
    print('Step 1: Tuning hyper-parameters using Bayesian Optimization\n')

    best_params = tune_xgb(param_space, X_train, y_train, X_early_stop, y_early_stop, n_iter = 150)
    
    print('\tThe best hyper-parameters found:\n')
    print(*['\t\t%s = %s' % (k, str(round(v, 4))) for k, v in best_params.items()], sep='\n')

    # # # # # # # # #
    # Step 2: Empirical thresholding: finding optimal classification threshold
    # # # # # # # # #
    print('\nStep 2: Empirical Thresholding\n')
    
    # I use 1500 trees which is very close to optimal n_trees found by early stopping while tuning
    xgboost_clf = XGBClassifier(**best_params, n_estimators=1500)
    
    classification_cutoff = optimal_threshold(xgboost_clf, X_train, y_train, min_prec = MIN_PRECISION)
    
    print('\tOptimal classification threshold = %1.3f' % classification_cutoff)
    
    # # # # # # # # #
    # Setp 3: Training and testing the model
    # # # # # # # # #
    print('\nStep 3: Training and testing the model\n')
    
    # Training on all the training data (excluding the small validation set to avoid overfitting)
    xgboost_clf.fit(X_train, y_train, verbose = False)
    
    y_pred = thresholded_predict(X_test, xgboost_clf, threshold = classification_cutoff)
    
    test_recall = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    
    print('\tTest set performance:')
    print('\tRecall    = %2.3f' % test_recall)
    print('\tPrecision = %2.3f' % test_precision)

from getdata import CSVLoader, CSVSampler
from data_transformation import DataTransformation, CustomerFeatures, Terminalriskfeatures
import xgboost
import pandas as pd
import sklearn
import numpy as np
import matplotlib as plt

class Gui:
    def testgui():
        print("test")

if __name__ == "__main__":
    #loader = CSVLoader(f"C:/PM4/transactions.csv")
    #loader.load_csv()
    begin_date = "2018-04-01"
    end_date = "2019-09-30"
    input_file = "C:/PM4/transactions.csv"
    output_folder = "C:/PM4/processed-data/"
    processor = DataTransformation(input_file, output_folder, begin_date, end_date)
    processor.load_data()
    
    if processor.process_transactions():
        processor.run_checks()
        processor.save_processed_data()
    
    print("Verarbeitung abgeschlossen.")
    
    X, y = sklearn.datasets.make_classification(n_samples=5000, n_features=2, n_informative=2,
                                            n_redundant=0, n_repeated=0, n_classes=2,
                                            n_clusters_per_class=1,
                                            weights=[0.95, 0.05],
                                            class_sep=0.5, random_state=0)
    loader = CSVLoader("C:/PM4/processed-data/")
    loader.load_csv
    dataset_df = pd.DataFrame({'X1':X[:,0],'X2':X[:,1], 'Y':y})

    def kfold_cv_with_classifier(classifier,
                             X,
                             y,
                             n_splits=5,
                             strategy_name="Basline classifier"):
    
        cv = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        
        cv_results_ = sklearn.model_selection.cross_validate(classifier,X,y,cv=cv,
                                                            scoring=['roc_auc',
                                                                    'average_precision',
                                                                    'balanced_accuracy'],
                                                            return_estimator=True)
        
        results = round(pd.DataFrame(cv_results_),3)
        results_mean = list(results.mean().values)
        results_std = list(results.std().values)
        results_df = pd.DataFrame([[str(round(results_mean[i],3))+'+/-'+
                                    str(round(results_std[i],3)) for i in range(len(results))]],
                                columns=['Fit time (s)','Score time (s)',
                                        'AUC ROC','Average Precision','Balanced accuracy'])
        results_df.rename(index={0:strategy_name}, inplace=True)
        
        classifier_0 = cv_results_['estimator'][0]
        
        (train_index, test_index) = next(cv.split(X, y))
        train_df = pd.DataFrame({'X1':X[train_index,0], 'X2':X[train_index,1], 'Y':y[train_index]})
        test_df = pd.DataFrame({'X1':X[test_index,0], 'X2':X[test_index,1], 'Y':y[test_index]})
        
        return (results_df, classifier_0, train_df, test_df)
    
    def plot_decision_boundary_classifier(ax, 
                                      classifier,
                                      train_df,
                                      input_features=['X1','X2'],
                                      output_feature='Y',
                                      title="",
                                      fs=14,
                                      plot_training_data=True):

        plot_colors = ["tab:blue","tab:orange"]

        x1_min, x1_max = train_df[input_features[0]].min() - 1, train_df[input_features[0]].max() + 1
        x2_min, x2_max = train_df[input_features[1]].min() - 1, train_df[input_features[1]].max() + 1
        
        plot_step=0.1
        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, plot_step),
                            np.arange(x2_min, x2_max, plot_step))

        Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu_r,alpha=0.3)

        if plot_training_data:
            # Plot the training points
            groups = train_df.groupby(output_feature)
            for name, group in groups:
                ax.scatter(group[input_features[0]], group[input_features[1]], edgecolors='black', label=name)
            
        ax.set_title(title, fontsize=fs)
        ax.set_xlabel(input_features[0], fontsize=fs)
        ax.set_ylabel(input_features[1], fontsize=fs)
    
    
    fig_decision_boundary, ax = plt.subplots(1, 3, figsize=(5*3,5))
    classifier = sklearn.tree.DecisionTreeClassifier(max_depth=5,class_weight={0:1,1:1},random_state=0)
    cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_results_ = sklearn.model_selection.cross_validate(classifier, X, y, cv=cv,
                                                     scoring=['roc_auc',
                                                              'average_precision',
                                                              'balanced_accuracy'],
                                                     return_estimator=True)
    # Retrieve the decision tree from the first fold of the cross-validation
    classifier_0 = cv_results_['estimator'][0]
    # Retrieve the indices used for the training and testing of the first fold of the cross-validation
    (train_index, test_index) = next(cv.split(X, y))

    # Recreate the train and test DafaFrames from these indices
    train_df = pd.DataFrame({'X1':X[train_index,0], 'X2':X[train_index,1], 'Y':y[train_index]})
    test_df = pd.DataFrame({'X1':X[test_index,0], 'X2':X[test_index,1], 'Y':y[test_index]})
    input_features = ['X1','X2']
    output_feature = 'Y'

    plot_decision_boundary_classifier(ax[0], classifier_0,
                                    train_df,
                                    title="Decision surface of the decision tree\n With training data",
                                    plot_training_data=True)

    plot_decision_boundary_classifier(ax[1], classifier_0,
                                    train_df,
                                    title="Decision surface of the decision tree\n",
                                    plot_training_data=False)


    plot_decision_boundary_classifier(ax[2], classifier_0,
                                    test_df,
                                    title="Decision surface of the decision tree\n With test data",
                                    plot_training_data=True)

    ax[-1].legend(loc='upper left', 
                bbox_to_anchor=(1.05, 1),
                title="Class")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=0, vmax=1))
    cax = fig_decision_boundary.add_axes([0.93, 0.15, 0.02, 0.5])
    fig_decision_boundary.colorbar(sm, cax=cax, alpha=0.3, boundaries=np.linspace(0, 1, 11))

    def plot_decision_boundary(classifier_0,
                           train_df, 
                           test_df):
    
        fig_decision_boundary, ax = plt.subplots(1, 3, figsize=(5*3,5))

        plot_decision_boundary_classifier(ax[0], classifier_0,
                                    train_df,
                                    title="Decision surface of the decision tree\n With training data",
                                    plot_training_data=True)

        plot_decision_boundary_classifier(ax[1], classifier_0,
                                    train_df,
                                    title="Decision surface of the decision tree\n",
                                    plot_training_data=False)


        plot_decision_boundary_classifier(ax[2], classifier_0,
                                    test_df,
                                    title="Decision surface of the decision tree\n With test data",
                                    plot_training_data=True)

        ax[-1].legend(loc='upper left', 
                    bbox_to_anchor=(1.05, 1),
                    title="Class")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=0, vmax=1))
        cax = fig_decision_boundary.add_axes([0.93, 0.15, 0.02, 0.5])
        fig_decision_boundary.colorbar(sm, cax=cax, alpha=0.3, boundaries=np.linspace(0, 1, 11))
        
        return fig_decision_boundary

    classifier = xgboost.XGBClassifier(n_estimators=100,
                                    max_depth=6,
                                    learning_rate=0.3,
                                    random_state=0)

    (results_df_xgboost, classifier_0, train_df, test_df) = kfold_cv_with_classifier(classifier, 
                                                                                    X, y, 
                                                                                    n_splits=5,
                                                                                    strategy_name="XGBoost")

    fig_decision_boundary = plot_decision_boundary(classifier_0, train_df, test_df)
    
    pd.concat([results_df_xgboost])
"""

