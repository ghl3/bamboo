
import pandas
import bamboo.modeling

import sklearn.ensemble
from sklearn import cross_validation

group = [0, 0, 0, 0,
         1, 1,
         0, 1]

feature1 = [1, 1, 1, 1,
            2, 2,
            3, 4]

feature2 = [10.0, 10.5, 9.5, 11.0,
            20.0, 20.0,
            0.0, 200.0]

df = pandas.DataFrame({'group':group,
                       'feature1':feature1,
                       'feature2':feature2})



features = df[['feature1', 'feature2']]
targets = df['group']
classifier = sklearn.ensemble.RandomForestClassifier()
fitted = classifier.fit(features, targets)

def test_balance_dataset():
    bamboo.modeling.balance_dataset(df.groupby('group'))


def test_get_prediction():
    predictions = bamboo.modeling.get_prediction(fitted, features, targets)
    print '\n'
    print predictions.head()


def test_get_scores():
    skf = cross_validation.StratifiedKFold(targets, n_folds=2)
    predictions = bamboo.modeling.get_scores(classifier, features, targets, skf)
    print '\n'
    for prediction in predictions:
        print prediction.head()


def test_plot_roc_curve():
    predictions = bamboo.modeling.get_prediction(fitted, features, targets)
    bamboo.modeling.plot_roc_curve(df['group'], predictions['predict_proba_0'])


def test_plot_precision_recall_curve():
    predictions = bamboo.modeling.get_prediction(fitted, features, targets)
    bamboo.modeling.plot_precision_recall_curve(df['group'], predictions['predict_proba_0'])


def test_plot_distribution():
    bamboo.modeling.plot_distribution(classifier, features, targets)


