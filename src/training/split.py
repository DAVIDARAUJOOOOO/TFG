from sklearn.model_selection import train_test_split

def split(df,columns_no_train):
  #Estratifiquem per dislexia i curs
  df["stratification"]=df["dislexia"].astype(str)+"_"+df["curs"].astype(str)
  X=df.drop(columns=columns_no_train)
  y=df["dislexia"]

  X_train,X_test,y_train,y_test=train_test_split(
      X,y,
      test_size=0.2,
      random_state=42,
      stratify=df["stratification"]
  )
  return X_train,X_test,y_train,y_test