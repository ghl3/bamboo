
import matplotlib.pyplot as plt

from sklearn.metrics.scorer import check_scoring
from sklearn.metrics import roc_curve, auc


def plot_auc_curve(estimator, X_test, y_test):

    y_score = estimator.predict_proba(X_test)
    scorer = check_scoring(estimator, scoring=None)

    scorer(estimator, X_test, y_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr[0], tpr[0], _ = roc_curve(y_test, y_score[:, 1])
    roc_auc[0] = auc(fpr[0], tpr[0])

    plt.figure()
    plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc[0]
