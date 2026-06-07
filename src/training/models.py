from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


def get_models():
    return {
        "rf": RandomForestClassifier(random_state=42),
        "xgb": XGBClassifier(random_state=42),
        "svm": SVC(random_state=42, probability=True),
        "knn": KNeighborsClassifier(),
        "lm": LogisticRegression(random_state=42),
        "lgbm": LGBMClassifier(random_state=42)
    }

def get_scoring():
    return {
        "f1": "f1",
        "recall": "recall",
        "precision": "precision"
    }