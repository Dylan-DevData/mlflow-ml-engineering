import  mlflow

EXPERIENCE_NAME = "cas pratique"



#  etape  1 :  train
step_1 =  mlflow.projects.run(
    uri="./" , 
    entry_point="train",
    experiment_name=EXPERIENCE_NAME,
    env_manager="local",
)
step_1.wait() #  on  attend  que l'etape 1 soit vraiment  termine avant de continuer 

run_id_train =  step_1.run_id

print(f"Entrainement  termine , run_id = {run_id_train} ")

#  etape 2: evaluation 
step_2 =  mlflow.projects.run(
    uri="./" ,
    entry_point="eval",
    experiment_name=EXPERIENCE_NAME , 
    parameters={"run_id":run_id_train},
      env_manager="local",
)

step_2.wait()

print("workflow complet  termine")
