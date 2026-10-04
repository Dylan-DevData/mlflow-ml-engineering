#   import  librairie  et modules
import  mlflow
import  mlflow.sklearn
import  numpy as np 
import  pandas as pd
from  sklearn.ensemble import RandomForestClassifier
from  sklearn.model_selection import  train_test_split
from  sklearn.preprocessing import  StandardScaler 
from  sklearn.metrics import  *
from  sklearn.pipeline import  make_pipeline
import  sys

#   trainning  Data
try:
    df =  pd.read_csv("raw_data/heart.csv")
    print("data bien  importe")
except:
    print("path  data incorect")

X = df.iloc[:,:-1].values
y  = df.iloc[:,-1].values

#   train  test
X_train , X_test ,  y_train , y_test = train_test_split(X ,y , test_size=0.2 , random_state=42 , stratify=y)

#  set  Auto  looging  for  scikit-learn  flavor
mlflow.set_experiment("heath_prediction")

mlflow.sklearn.autolog(registered_model_name="model_heath")

#   Parameters 
max_depth_param  =  int(sys.argv[1])
n_estimators_param = int(sys.argv[2])


#  train  the  model
model = make_pipeline(StandardScaler() , RandomForestClassifier(max_depth=max_depth_param , n_estimators=n_estimators_param))
model.fit(X_train , y_train)
y_pred =  model.predict(X_test)
f1 =  f1_score(y_test , y_pred)

print("entrainement  termine ...")












