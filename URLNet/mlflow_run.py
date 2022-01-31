from phishing.experiments.url.URLNet.utils import load_data
from phishing.mlflow.templates.model_template import RFHousePriceModel

data_dir = "../data/urls.txt"
X, y = load_data(data_dir)

params_list = [{"n_estimators": 75, "max_depth": 5, "random_state": 42}]
# run these experiments, each with its own instance of model with the supplied parameters.
for params in params_list:
    rfr = RFHousePriceModel.new_instance(params)
    experiment = "Experiment with {} trees".format(params['n_estimators'])
    (experimentID, runID) = rfr.mlflow_run(X, y, run_name="AirBnB House Pricing Regression Model", verbose=True)
    print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
    print("-" * 100)
