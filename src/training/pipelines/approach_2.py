from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA



def get_pipeline_approach2(model):

    pca=PCA(n_components=30,random_state=42)
    preprocessor2=StandardScaler()

    pipeline2=Pipeline([
        ('prep',preprocessor2),
        ('smote',SMOTE(random_state=42)),
        ('PCA',pca),
        ('model', model)
    ])

    return pipeline2