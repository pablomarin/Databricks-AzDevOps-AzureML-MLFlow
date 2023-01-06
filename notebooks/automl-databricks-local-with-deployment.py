# Databricks notebook source
# MAGIC %md
# MAGIC Copyright (c) Microsoft Corporation. All rights reserved.
# MAGIC 
# MAGIC Licensed under the MIT License.

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoML Installation
# MAGIC 
# MAGIC **Supported Databricks runtime 7.3 LTS, Install AML sdk by running the following command in the first cell of the notebook.**
# MAGIC 
# MAGIC %pip install --upgrade --force-reinstall -r https://aka.ms/automl_linux_requirements.txt

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall -r https://aka.ms/automl_linux_requirements.txt

# COMMAND ----------

# MAGIC %md
# MAGIC # AutoML : Classification with Local Compute on Azure DataBricks with deployment to ACI
# MAGIC 
# MAGIC In this example we use the scikit-learn's to showcase how you can use AutoML for a simple classification problem.
# MAGIC 
# MAGIC In this notebook you will learn how to:
# MAGIC 1. Create Azure Machine Learning Workspace object and initialize your notebook directory to easily reload this object from a configuration file.
# MAGIC 2. Create an `Experiment` in an existing `Workspace`.
# MAGIC 3. Configure AutoML using `AutoMLConfig`.
# MAGIC 4. Train the model using AzureDataBricks.
# MAGIC 5. Explore the results.
# MAGIC 6. Register the model.
# MAGIC 7. Deploy the model.
# MAGIC 8. Test the best fitted model.
# MAGIC 
# MAGIC Prerequisites:
# MAGIC Before running this notebook, please follow the readme for installing necessary libraries to your cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Machine Learning Services Resource Provider
# MAGIC Microsoft.MachineLearningServices only needs to be registed once in the subscription. To register it:
# MAGIC Start the Azure portal.
# MAGIC Select your All services and then Subscription.
# MAGIC Select the subscription that you want to use.
# MAGIC Click on Resource providers
# MAGIC Click the Register link next to Microsoft.MachineLearningServices

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check the Azure ML Core SDK Version to Validate Your Installation

# COMMAND ----------

import azureml.core

print("SDK Version:", azureml.core.VERSION)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize an Azure ML Workspace
# MAGIC ### What is an Azure ML Workspace and Why Do I Need One?
# MAGIC 
# MAGIC An Azure ML workspace is an Azure resource that organizes and coordinates the actions of many other Azure resources to assist in executing and sharing machine learning workflows.  In particular, an Azure ML workspace coordinates storage, databases, and compute resources providing added functionality for machine learning experimentation, operationalization, and the monitoring of operationalized models.
# MAGIC 
# MAGIC 
# MAGIC ### What do I Need?
# MAGIC 
# MAGIC To create or access an Azure ML workspace, you will need to import the Azure ML library and specify following information:
# MAGIC * A name for your workspace. You can choose one.
# MAGIC * Your subscription id. Use the `id` value from the `az account show` command output above.
# MAGIC * The resource group name. The resource group organizes Azure resources and provides a default region for the resources in the group. The resource group will be created if it doesn't exist. Resource groups can be created and viewed in the [Azure portal](https://portal.azure.com)
# MAGIC * Supported regions include `eastus2`, `eastus`,`westcentralus`, `southeastasia`, `westeurope`, `australiaeast`, `westus2`, `southcentralus`.

# COMMAND ----------

import os
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

sp = ServicePrincipalAuthentication(tenant_id=os.environ['TENANT_ID'], # tenantID
                                    service_principal_id=os.environ['SERVICE_PRINCIPAL_ID'], # clientId
                                    service_principal_password=os.environ['SERVICE_PRINCIPAL_SECRET']) # clientSecret

ws = Workspace.get(name=os.environ['AZML_WORKSPACE_NAME'],
                   auth=sp,
                   subscription_id=os.environ['SUBSCRIPTION_ID'],
                   resource_group=os.environ['AZML_RESOURCE_GROUP'])

ws.get_details()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating a Workspace
# MAGIC If you already have access to an Azure ML workspace you want to use, you can skip this cell.  Otherwise, this cell will create an Azure ML workspace for you in the specified subscription, provided you have the correct permissions for the given `subscription_id`.
# MAGIC 
# MAGIC This will fail when:
# MAGIC 1. The workspace already exists.
# MAGIC 2. You do not have permission to create a workspace in the resource group.
# MAGIC 3. You are not a subscription owner or contributor and no Azure ML workspaces have ever been created in this subscription.
# MAGIC 
# MAGIC If workspace creation fails for any reason other than already existing, please work with your IT administrator to provide you with the appropriate permissions or to provision the required resources.
# MAGIC 
# MAGIC **Note:** Creation of a new workspace can take several minutes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an Experiment
# MAGIC 
# MAGIC As part of the setup you have already created an Azure ML `Workspace` object. For AutoML you will need to create an `Experiment` object, which is a named object in a `Workspace` used to run experiments.

# COMMAND ----------

import logging
import os
import random
import time
import json

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun

# COMMAND ----------

# Choose a name for the experiment and specify the project folder.
experiment_name = "automl-local-classification"

experiment = Experiment(ws, experiment_name)

output = {}
output["SDK version"] = azureml.core.VERSION
output["Subscription ID"] = ws.subscription_id
output["Workspace Name"] = ws.name
output["Resource Group"] = ws.resource_group
output["Location"] = ws.location
output["Experiment Name"] = experiment.name
pd.set_option("display.max_colwidth", -1)
pd.DataFrame(data=output, index=[""]).T

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Training Data Using Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC Automated ML takes a `TabularDataset` as input.
# MAGIC 
# MAGIC You are free to use the data preparation libraries/tools of your choice to do the require preparation and once you are done, you can write it to a datastore and create a TabularDataset from it.

# COMMAND ----------

# The data referenced here was a 1MB simple random sample of the Chicago Crime data into a local temporary directory.
from azureml.core.dataset import Dataset

example_data = "https://dprepdata.blob.core.windows.net/demo/crime0-random.csv"
dataset = Dataset.Tabular.from_delimited_files(example_data)
dataset.take(5).to_pandas_dataframe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review the TabularDataset
# MAGIC You can peek the result of a TabularDataset at any range using `skip(i)` and `take(j).to_pandas_dataframe()`. Doing so evaluates only j records for all the steps in the TabularDataset, which makes it fast even against large datasets.

# COMMAND ----------

training_data = dataset.drop_columns(columns=["FBI Code"])
label = "Primary Type"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure AutoML
# MAGIC 
# MAGIC Instantiate an `AutoMLConfig` object to specify the settings and data used to run the experiment.
# MAGIC 
# MAGIC |Property|Description|
# MAGIC |-|-|
# MAGIC |**task**|classification or regression|
# MAGIC |**primary_metric**|This is the metric that you want to optimize. Classification supports the following primary metrics: <br><i>accuracy</i><br><i>AUC_weighted</i><br><i>average_precision_score_weighted</i><br><i>norm_macro_recall</i><br><i>precision_score_weighted</i>|
# MAGIC |**primary_metric**|This is the metric that you want to optimize. Regression supports the following primary metrics: <br><i>spearman_correlation</i><br><i>normalized_root_mean_squared_error</i><br><i>r2_score</i><br><i>normalized_mean_absolute_error</i>|
# MAGIC |**iteration_timeout_minutes**|Time limit in minutes for each iteration.|
# MAGIC |**iterations**|Number of iterations. In each iteration AutoML trains a specific pipeline with the data.|
# MAGIC |**spark_context**|Spark Context object. for Databricks, use spark_context=sc|
# MAGIC |**max_concurrent_iterations**|Maximum number of iterations to execute in parallel. This should be <= number of worker nodes in your Azure Databricks cluster.|
# MAGIC |**n_cross_validations**|Number of cross validation splits.|
# MAGIC |**training_data**|Input dataset, containing both features and label column.|
# MAGIC |**label_column_name**|The name of the label column.|

# COMMAND ----------

automl_config = AutoMLConfig(
    task="classification",
    debug_log="automl_errors.log",
    primary_metric="AUC_weighted",
    iteration_timeout_minutes=10,
    iterations=5,
    n_cross_validations=10,
    max_concurrent_iterations=1,  # change it based on number of worker nodes
    verbosity=logging.INFO,
    spark_context=sc,  # databricks/spark related
    training_data=training_data,
    label_column_name=label,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the Models
# MAGIC 
# MAGIC Call the `submit` method on the experiment object and pass the run configuration. Execution of local runs is synchronous. Depending on the data and the number of iterations this can run for a while.

# COMMAND ----------

local_run = experiment.submit(automl_config, show_output=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore the Results

# COMMAND ----------

# MAGIC %md
# MAGIC #### Portal URL for Monitoring Runs
# MAGIC 
# MAGIC The following will provide a link to the web interface to explore individual run details and status. In the future we might support output displayed in the notebook.

# COMMAND ----------

displayHTML(
    "<a href={} target='_blank'>Azure Portal: {}</a>".format(
        local_run.get_portal_url(), local_run.id
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy
# MAGIC 
# MAGIC ### Retrieve the Best Model
# MAGIC 
# MAGIC Below we select the best pipeline from our iterations. The `get_output` method on `automl_classifier` returns the best run and the fitted model for the last invocation. Overloads on `get_output` allow you to retrieve the best run and fitted model for *any* logged metric or for a particular *iteration*.

# COMMAND ----------

best_run, fitted_model = local_run.get_output()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download the conda environment file
# MAGIC From the *best_run* download the conda environment file that was used to train the AutoML model.

# COMMAND ----------

from azureml.automl.core.shared import constants
 
conda_env_file_name = "conda_env.yml"
best_run.download_file(
    name="outputs/conda_env_v_1_0_0.yml", output_file_path=conda_env_file_name
)
 
with open(conda_env_file_name, "r") as conda_file:
    conda_file_contents = conda_file.read()
    
# Replace the target string
conda_file_contents = conda_file_contents.replace('python=3.7.5', 'python=3.8.10')
conda_file_contents = conda_file_contents.replace('pip:',"pip:\n  - azureml-defaults")
 
# Write the file out again
with open(conda_env_file_name, 'w') as file:
    file.write(conda_file_contents)
    print(conda_file_contents)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download the model scoring file
# MAGIC From the *best_run* download the scoring file to get the predictions from the AutoML model.

# COMMAND ----------

from azureml.automl.core.shared import constants

script_file_name = "scoring_file.py"
best_run.download_file(
    name="outputs/scoring_file_v_1_0_0.py", output_file_path=script_file_name
)
with open(script_file_name, "r") as scoring_file:
    scoring_file_contents = scoring_file.read()
    print(scoring_file_contents)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the Fitted Model for Deployment
# MAGIC If neither metric nor iteration are specified in the register_model call, the iteration with the best primary metric is registered.

# COMMAND ----------

description = "AutoML Model"
tags = None
model = local_run.register_model(description=description, tags=tags)
local_run.model_id  # This will be written to the scoring script file later in the notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy the model as a Web Service on Azure Container Instance
# MAGIC 
# MAGIC Create the configuration needed for deploying the model as a web service service.

# COMMAND ----------

from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.environment import Environment

myenv = Environment.from_conda_specification(
    name="myenv", file_path=conda_env_file_name
)
inference_config = InferenceConfig(entry_script=script_file_name, environment=myenv)

aciconfig = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    tags={"area": "digits", "type": "automl_classification"},
    description="sample service for Automl Classification",
)

# COMMAND ----------

from azureml.core.webservice import Webservice
from azureml.core.model import Model

aci_service_name = "automl-databricks-local"
print(aci_service_name)
aci_service = Model.deploy(ws, aci_service_name, [model], inference_config, aciconfig)
aci_service.wait_for_deployment(True)
print(aci_service.state)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Best Fitted Model
# MAGIC 
# MAGIC #### Load Test Data - you can split the dataset beforehand & pass Train dataset to AutoML and use Test dataset to evaluate the best model.

# COMMAND ----------

dataset_test = Dataset.Tabular.from_delimited_files(
    path="https://dprepdata.blob.core.windows.net/demo/crime0-test.csv"
)

df_test = dataset_test.to_pandas_dataframe()
df_test = df_test[pd.notnull(df_test["Primary Type"])]

y_test = df_test[["Primary Type"]]
X_test = df_test.drop(["Primary Type", "FBI Code"], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Testing Our Best Fitted Model
# MAGIC We will try to predict digits and see how our model works. This is just an example to show you.

# COMMAND ----------

fitted_model.predict(X_test)

# COMMAND ----------

### Delete the service

# COMMAND ----------

aci_service.delete()

# COMMAND ----------

# MAGIC %md
# MAGIC ![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/azure-databricks/automl/automl-databricks-local-with-deployment.png)
