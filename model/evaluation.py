import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def calc_auc(anomaly_scores, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, anomaly_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc, fpr, tpr


def calc_f1(predictions, labels):
    f1 = f1_score(labels, predictions)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, predictions).ravel()
    balanced_acc = metrics.balanced_accuracy_score(labels, predictions)
    acc = metrics.accuracy_score(labels, predictions)
    return f1, tn, fp, fn, tp, balanced_acc, acc


def latent_plot(X_embedding, negative_augmentation_embeddings, positive_augmentation_embeddings, label, experiment_dir):
    all_embeddings = [X_embedding]
    if negative_augmentation_embeddings is not None:
        all_embeddings.append(negative_augmentation_embeddings)
    if positive_augmentation_embeddings is not None:
        all_embeddings.append(positive_augmentation_embeddings)

    augmented_embeddings = np.concatenate(all_embeddings, axis=0)
    tsne = TSNE(n_components=2)
    tsne_embedding = tsne.fit_transform(augmented_embeddings)

    test_tsne = pd.DataFrame(tsne_embedding[:X_embedding.shape[0]])
    if negative_augmentation_embeddings is not None:
        negative_tsne = pd.DataFrame(
            tsne_embedding[X_embedding.shape[0]:X_embedding.shape[0] + negative_augmentation_embeddings.shape[0]])
    else:
        negative_tsne = None
    if positive_augmentation_embeddings is not None:
        if negative_augmentation_embeddings is not None:
            positive_tsne = pd.DataFrame(
                tsne_embedding[X_embedding.shape[0] + negative_augmentation_embeddings.shape[0]:])
        else:
            positive_tsne = pd.DataFrame(tsne_embedding[X_embedding.shape[0]:])
    else:
        positive_tsne = None

    fig, ax = plt.subplots()
    legend_list = ['normal', 'anomaly']
    test_tsne[label == 0].plot(x=0, y=1, kind='scatter', ax=ax, c='darkgray', alpha=0.7, marker='x',
                               label='normal')
    test_tsne[label == 1].plot(x=0, y=1, kind='scatter', ax=ax, c='red', alpha=0.7, marker='x',
                               label='anomaly')

    if negative_tsne is not None:
        negative_tsne.plot(x=0, y=1, kind='scatter', ax=ax, c='orange', alpha=1, marker='o', s=5,
                           label='anomaly negative transformation')
        legend_list.append('Negative Aug')

    if positive_tsne is not None:
        positive_tsne.plot(x=0, y=1, kind='scatter', ax=ax, c='green', alpha=1, marker='o', s=2,
                           label='anomaly positive transformation')
        legend_list.append('Positive Aug')

    ax.legend(legend_list)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
               fancybox=True, shadow=True, ncol=3)

    fig.savefig(f'{experiment_dir}/tsne.png')
    fig.clear()


def run_evaluation(model_name, pred, scores, labels, tau=None):
    if scores is None or None in scores:  # predictor only provide binary anomaly prediction
        auc, fpr, tpr = calc_auc(pred, labels)
        f1, tn, fp, fn, tp, balanced_acc, acc = calc_f1(pred, labels)
    else:
        if pred is None or None in pred:
            tau = np.quantile(scores, 0.9) * 1.5
            pred = [0 if a < tau else 1 for a in scores]
        auc, fpr, tpr = calc_auc(scores, labels)
        f1, tn, fp, fn, tp, balanced_acc, acc = calc_f1(pred, labels)
    eval_dict = {'model': model_name, 'f1': f1, 'auc': auc, 'tau': tau, 'fpr': fpr, 'tpr': tpr, 'tn': tn, 'fp': fp,
                 'fn': fn, 'tp': tp,
                 'b_acc': balanced_acc, 'acc': acc}
    return pred, eval_dict
