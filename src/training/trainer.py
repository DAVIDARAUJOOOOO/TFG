from sklearn.model_selection import cross_val_predict,StratifiedKFold
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import fbeta_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


class Training:

  def __init__ (self,X_train,y_train,X_test,y_test,groups=None):
    self.X_train=X_train
    self.y_train=y_train
    self.X_test=X_test
    self.y_test=y_test
    self.results={}
    self.groups=groups

  def entrenament(self,pipeline, dict_models, scoring,params_grid,cv):

    def find_best_threshold(y_true, y_probs, beta=2):

      thresholds = np.arange(0.01, 1.0, 0.01)

      best_threshold = 0.5
      best_score = -1

      for t in thresholds:

          y_pred = (y_probs >= t).astype(int)

          score = fbeta_score(y_true, y_pred, beta=beta)

          if score > best_score:
              best_score = score
              best_threshold = t

      return best_threshold, best_score

    for name, model in dict_models.items():
      print(f"Entrenant {name}")

      pipeline.set_params(model=model)
      param_grid_model=params_grid[f"param_grid_{name}"]

      grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid_model,
        cv=cv,
        scoring=scoring,
        refit='f1',
        n_jobs=-1,
        verbose=2
        )

      # Entrenar
      if self.groups is not None:
        grid_search.fit(self.X_train, self.y_train, groups=self.groups)
      else:
        grid_search.fit(self.X_train, self.y_train)
      cv_results=grid_search.cv_results_
      best= grid_search.best_estimator_

      #Millor Threshold

      # OUT-OF-FOLD probabilities
      cv_threshold=StratifiedKFold(
      n_splits=5,
      shuffle=True,
      random_state=42
       )

      oof_probs = cross_val_predict(
          best,
          self.X_train,
          self.y_train,
          cv=cv_threshold,
          method="predict_proba",
          n_jobs=-1
      )[:,1]

      # Find optimal threshold
      best_threshold, best_f2 = find_best_threshold(
          self.y_train,
          oof_probs,
          beta=2
      )

      self.results[name] = {
            "model": best,
            "best_params": grid_search.best_params_,
            "mean_test_f1":cv_results['mean_test_f1'][grid_search.best_index_],
            "std_test_f1": cv_results['std_test_f1'][grid_search.best_index_],
            "mean_test_recall":cv_results['mean_test_recall'][grid_search.best_index_],
            "std_test_recall": cv_results['std_test_recall'][grid_search.best_index_],
            "mean_test_precision":cv_results['mean_test_precision'][grid_search.best_index_],
            "std_test_precision": cv_results['std_test_precision'][grid_search.best_index_],
            "threshold": best_threshold
        }
    return self.results


  def print_results(self):
    for name in self.results.keys():
      print(f"Model: {name}")
      print(f"Best hiperparameters: {self.results[name]['best_params']}")
      print(f"Best mean F1 in CV: {self.results[name]["mean_test_f1"]} ± {self.results[name]["std_test_f1"]}")
      print(f"Best mean Recall in CV: {self.results[name]['mean_test_recall']} ± {self.results[name]['std_test_recall']}")
      print(f"Best mean Precision in CV: {self.results[name]['mean_test_precision']} ± {self.results[name]['std_test_precision']}")
      model = self.results[name]["model"]
      y_test_probs = model.predict_proba(self.X_test)[:, 1]
      y_train_probs = model.predict_proba(self.X_train)[:, 1]
      y_pred=(y_test_probs>=self.results[name]["threshold"]).astype(int)
      y_train_pred=(y_train_probs>=self.results[name]["threshold"]).astype(int)
      print(f"Classification report test de {name} {classification_report(self.y_test, y_pred)}")
      print(f"Classification report train de {name} {classification_report(self.y_train, y_train_pred)}")
    return y_pred,y_test_probs

  def print_results_profund(self):
    tablas_modelos = {}

    for name, info in self.results.items():
        print(f"ANALITZANT MODEL: {name.upper()} amb threshold: {self.results[name]["threshold"]}")

        model_pipeline = info["model"]
        y_probs = model_pipeline.predict_proba(self.X_test)[:, 1]
        y_pred= (y_probs >=self.results[name]["threshold"]).astype(int)

        df_modelo = pd.DataFrame({
            'Realidad': self.y_test.values,
            'Prediccion': y_pred,
            'Prob_Dislexia': y_probs,
            'Error': self.y_test.values != y_pred
        }, index=self.X_test.index)

        tablas_modelos[name] = df_modelo

        solo_dislexicos = df_modelo[df_modelo['Realidad'] == 1]
        dislexicos_modelos=df_modelo[df_modelo["Prediccion"]==1]
        n_modelo=(dislexicos_modelos["Realidad"]==1).sum()
        n_detectados = (solo_dislexicos['Prediccion'] == 1).sum()
        total_dislexicos = len(solo_dislexicos)
        total_predicciones=len(dislexicos_modelos)

        print(f"\n RECALL: Detectats {n_detectados} de {total_dislexicos} nens amb dislexia.")
        print(f"\n PRECISSION: {n_modelo} dislexics eren reals dels {total_predicciones} predits.")

        print("\nDetalle de los niños con dislexia:")
        print(solo_dislexicos)
        print(dislexicos_modelos)

  def plot_metrics(self):
    n_models = len(self.results)
    fig, axes = plt.subplots(n_models, 3, figsize=(18, 5 * n_models))
    if n_models == 1:
      axes = axes.reshape(1, -1)

    for i, (name, info) in enumerate(self.results.items()):
        model = info["model"]
        y_test_probs = model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_test_probs >= self.results[name]["threshold"]).astype(int)

#Confussion matrix

        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sa", "Dislèxia"])
        disp.plot(ax=axes[i, 0], cmap='Blues', colorbar=False)
        axes[i, 0].set_title(f"Confusió: {name.upper()}")

#Especificitat
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        axes[i, 0].text(
            0.5, -0.15,
            f"Especificitat: {specificity:.2f}",
            ha='center',
            transform=axes[i, 0].transAxes,
            fontsize=12)


#ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, y_test_probs)
        roc_auc = auc(fpr, tpr)
        axes[i, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        axes[i, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[i, 1].set_title(f"ROC: {name.upper()}")
        axes[i, 1].legend(loc="lower right")

#Precission-Recall curve

        prec, rec, _ = precision_recall_curve(self.y_test, y_test_probs)
        avg_prec = average_precision_score(self.y_test, y_test_probs)
        axes[i, 2].plot(rec, prec, color='green', lw=2, label=f'AP = {avg_prec:.2f}')
        axes[i, 2].set_title(f"P-R: {name.upper()}")
        axes[i, 2].legend(loc="lower left")

    plt.tight_layout()
    plt.show()

