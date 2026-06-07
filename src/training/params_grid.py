def get_param_grids():
    return {
    "param_grid_rf":{
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 5, 10],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
    'model__max_features': ['sqrt'],
    'model__class_weight': [None, 'balanced'] },

    "param_grid_xgb":{
    'model__n_estimators': [50, 100, 150],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.7, 0.8, 0.9],
    'model__colsample_bytree': [0.8, 0.9],
    'model__scale_pos_weight': [1,11]
},
"param_grid_svm":{
    'model__C': [0.1, 1, 10, 100],
    'model__gamma': ['scale', 0.01, 0.1, 1],
    'model__kernel': ['rbf', 'linear'],
    'model__class_weight': ['balanced'],
    'model__probability': [True]
},

"param_grid_knn":{
    'model__n_neighbors': [3, 5, 7, 10, 20],
    'model__weights': ['uniform', 'distance'],
    'model__metric': ['euclidean', 'manhattan']
},

"param_grid_lm":{
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l2','l1'],
    'model__class_weight': ['balanced'],
    'model__solver': ['liblinear','lbfgs']
},
"param_grid_lgbm":{
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__num_leaves': [7, 15, 31],
    'model__max_depth': [3, 5, -1],
    'model__is_unbalance': [True],
    'model__boost_from_average': [False]
}
}