from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr

def train_probe(features, labels, is_regression=False, max_iter=1000):
    if is_regression:
        model = Ridge(alpha=1.0)
    else:
        model = LogisticRegression(max_iter=max_iter)
    model.fit(features, labels)
    return model

def eval_probe(model, features, labels, is_regression=False):
    preds = model.predict(features)
    if is_regression:
        return {
            "pearson": float(pearsonr(labels, preds)[0]),
            "spearman": float(spearmanr(labels, preds)[0])
        }
    else:
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, average="weighted")),
        }