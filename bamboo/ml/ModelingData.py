

from sklearn.preprocessing import balance_weights


class ModelingData():
    """
    A class that stores a set of features and
    targets to be used for modeling.
    Provides functionality to do simple operations
    and manipulations of that data, removing
    boilerplate for ML code.
    """

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets


    @staticmethod
    def from_dataframe(df, target, features=None):

        target_data = df[target]

        if features is None:
            features = [feature for feature in df.columns
                        if features != target]

        feature_data = df[features]

        return ModelingData(feature_data, target_data)

    def __len__(self):
        assert(len(self.features)==len(self.targets))
        return len(self.targets)
    

    def fit(self, clf, *args, **kwargs):
        return clf.fit(self.features, self.targets, *args, **kwargs)


    def train_test_split(self, *args, **kwargs):
        """
        Based on: sklearn.cross_validation.train_test_split
        Returns two ModelingData instances, the first representing
        training data and the scond representing testing data
        """
        test_size = kwargs.pop('test_size', None)
        train_size = kwargs.pop('train_size', None)
        random_state = kwargs.pop('random_state', None)

        if test_size is None and train_size is None:
                    test_size = 0.25

        n_samples = len(self.targets)

        cv = ShuffleSplit(n_samples, test_size=test_size,
                          train_size=train_size,
                          random_state=random_state)

        train, test = next(iter(cv))

        X_train = self.features[train]
        y_train = self.targets[train]

        X_test = self.features[test]
        y_test = self.features[test]

        return ModelingData(X_train, X_test), ModelingData(Y_train, Y_test)


    def _balance_by_truncation(self):
        """
        Take a modeling_data instance and return
        a new instance with the state variable
        balanced
        """

        group_size = self.targets.value_counts().min()

        grouped = self.features.groupby(self.targets)

        print group_size, grouped
        
        indices = []

        for name, group in grouped:
            indices.extend(group[:group_size].index)

        print indices
            
        return ModelingData(self.features.ix[indices], self.targets.ix[indices])


    def get_balance_weights(self):
        return balance_weights(self.targets)


    def get_balanced(self, how):
        """
        Return a ModelingData derived from this instance
        but with balanced data (balanced according to
        the supplied options)
        """


    def hist(self, var_name):
        pass


    def scatter(self):
        pass


    def numeric_features(self):
        pass


    def plot_auc_surve(self, clf):
        return plotting.plot_auc_curve(clf, self.features, self.targets)



    def score(self, clf):

        scores = features.apply(lambda x: clf.predict_proba(x), axis=1)

        return pd.merge(scores, target)

        #return [{index: index, target: target, score: score for stuff in stuff]
