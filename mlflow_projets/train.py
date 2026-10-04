#   import  packages
import  pandas   as pd  
import numpy as np 
import  matplotlib.pyplot as plt
from  sklearn.preprocessing import  StandardScaler 
from  sklearn.model_selection import  train_test_split  , cross_val_score
from  sklearn.pipeline import  make_pipeline
from sklearn.linear_model import  LogisticRegression , SGDClassifier
from  sklearn.svm import SVC
from  sklearn.dummy import  DummyClassifier
from  sklearn.ensemble import  RandomForestClassifier
from  sklearn.metrics import *
import  mlflow


# #   load  data

# df =  pd.read_csv("../raw_data/heart.csv")
   




# print("==========data  affiche=========\n")
# print(df.head())

# #   feature  and  target
# X =  df.iloc[:,:-1].values
# y =  df.iloc[:,-1].values

# print("features and  target  ok")

# #   train test
# X_train ,   X_test , y_train , y_test =  train_test_split(X ,y , test_size=0.2 ,  random_state=42 , stratify=y)


# models = {"logistic_regresssion" :LogisticRegression() , 
#           "bete": DummyClassifier() , 
#           "sgd":SGDClassifier() , 
#           "svm" : SVC() , 
#           "rf":RandomForestClassifier(),
#           }

# mlflow.set_experiment("experiments datset heart.csv")
# mlflow.set_experiment_tag("challenge" , "models")

# for  i , v  in  models.items():
#     print(f"======model :{i} ================")
#     with mlflow.start_run(run_name=f"model:{i}") as run:
#         pipeline =  make_pipeline(StandardScaler() , v)
#         model = pipeline.fit(X_train , y_train)
#         y_pred =  model.predict(X_test)

#         accuracy = accuracy_score(y_test , y_pred)
#         f1 =  f1_score(y_test , y_pred)
#         print("======== matrice de confusion =============")
#         print(confusion_matrix(y_test , y_pred))

#         #   logging
#         mlflow.log_metrics({"accuracy" :accuracy ,
#                             "f1_score":f1
#                             })
#         mlflow.log_artifact("train.py")



print("=============== search runs ========================")

df_runs =  mlflow.search_runs(experiment_names="experiments datset heart.csv" , 
                               filter_string="metrics.f1_score" , 
                              order_by=["metrics.f1_score DESC"])

print(df_runs)



        