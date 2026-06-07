from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE



def get_pipeline_approach2b(model):

    preprocessor2=StandardScaler()

    pipeline2b=Pipeline([
    ('prep',preprocessor2),
    ('smote',SMOTE(random_state=42)),
    ('model', model)
])

    return pipeline2b