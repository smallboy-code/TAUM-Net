import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve



if __name__=='__main__':
    data = pd.read_csv('revised/BraTS_pred_Focus_loss_dual.csv')
    gt = data['mgmt_truth']
    pred = data['pred_class']
    confusion = confusion_matrix(gt, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    print('AUC:', roc_auc_score(gt, pred))
    print('Accuracy:', (TP + TN) / float(TP + TN + FP + FN))
    print('Sensitivity:', TP / float(TP + FN))
    print('Specificity:', TN / float(TN + FP))
    print('PPV:', TP / float(TP + FP))
    print('Recall:', TP / float(TP + FN))
    print('Precision:', TP / float(TP + FP))

    ax = sns.heatmap(confusion,
                     cmap="YlGnBu",
                     annot=True,
                     linewidths=2,
                     square=True,
                     xticklabels=['Negative', 'Positive'],
                     yticklabels=['Negative', 'Positive'], )
    ax.set_title('confusion matrix', family='Arial')
    plt.show()


    fpr, tpr, threshold = roc_curve(gt, data['pred'], pos_label=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('Ture Positive Rate')
    plt.title('roc curve')
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.show()


