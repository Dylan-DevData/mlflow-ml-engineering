import  mlflow

mlflow.projects.run(
    uri="./",
    entry_point="main",
    experiment_name="heath_prediction",
    env_manager="local",
    parameters={
        "max_depth_param":2,
        "n_estimators_param":10
    }
)

