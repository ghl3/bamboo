
import pandas as pd


class ScoreFeature():

    """
    A score created by a classifier using a
    specific set of features.
    Handles grabbing that set of features
    from a pandas dataset and
    scoring on those features.

    See examples in:
    /Users/george/Work/risk/IdFeatures.arff
    """

    def __init__(self, clf, names):
        self.clf = clf
        self.names = names
        self.fitted = None

    def get_features(self, df):
        return df[self.names]

    def score(self, features, class_index=1):
        scores = self.clf.predict_proba(self.get_features(features))
        class_scores = [score[class_index] for score in scores]
        return pd.Series(class_scores)
