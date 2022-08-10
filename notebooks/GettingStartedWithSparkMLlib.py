# Databricks notebook source
# MAGIC %md
# MAGIC # Getting started with MLlib - binary classification example

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall -r https://aka.ms/automl_linux_requirements.txt

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This tutorial is designed to get you started with Apache Spark MLlib. It investigates a binary classification problem - can you predict if an individual's income is greater than $50,000 based on demographic data? The dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult) and is provided with Databricks Runtime. This notebook demonstrates some of the capabilities available in MLlib, including tools for data preprocessing, machine learning pipelines, and several different machine learning algorithms.
# MAGIC 
# MAGIC This notebook includes the following steps:
# MAGIC 
# MAGIC 0. Load the dataset
# MAGIC 0. Feature preprocessing
# MAGIC 0. Define the model
# MAGIC 0. Build the pipeline
# MAGIC 0. Evaluate the model
# MAGIC 0. Hyperparameter tuning
# MAGIC 0. Make predictions and evaluate model performance
# MAGIC 
# MAGIC ## Setup Requirements
# MAGIC Databricks Runtime 7.0 or above or Databricks Runtime 7.0 ML or above. If you are running Databricks Runtime 6.x or Databricks Runtime 6.x ML, see ([AWS](https://docs.databricks.com/getting-started/spark/machine-learning.html)|[Azure](https://docs.microsoft.com/azure/databricks/getting-started/spark/machine-learning/)) for the correct notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. Load the dataset

# COMMAND ----------

# MAGIC %md
# MAGIC Use Databricks utilities to view the first few rows of the data.

# COMMAND ----------

# MAGIC %fs head --maxBytes=1024 databricks-datasets/adult/adult.data

# COMMAND ----------

# MAGIC %md Because the dataset does not include column names, create a schema to assign column names and datatypes. 

# COMMAND ----------

schema = """`age` DOUBLE,
`workclass` STRING,
`fnlwgt` DOUBLE,
`education` STRING,
`education_num` DOUBLE,
`marital_status` STRING,
`occupation` STRING,
`relationship` STRING,
`race` STRING,
`sex` STRING,
`capital_gain` DOUBLE,
`capital_loss` DOUBLE,
`hours_per_week` DOUBLE,
`native_country` STRING,
`income` STRING"""

dataset = spark.read.csv("/databricks-datasets/adult/adult.data", schema=schema)

# COMMAND ----------

# MAGIC %md Randomly split data into training and test sets, and set seed for reproducibility.
# MAGIC 
# MAGIC It's best to split the data before doing any preprocessing. This allows the test dataset to more closely simulate new data when we evaluate the model.

# COMMAND ----------

trainDF, testDF = dataset.randomSplit([0.8, 0.2], seed=42)
print(trainDF.cache().count()) # Cache because accessing training data multiple times
print(testDF.count())

# COMMAND ----------

# MAGIC %md ## Data Exploration

# COMMAND ----------

# MAGIC %md Let's review the data.

# COMMAND ----------

display(trainDF)

# COMMAND ----------

# MAGIC %md What's the distribution of the number of `hours_per_week`?

# COMMAND ----------

display(trainDF.select("hours_per_week").summary())

# COMMAND ----------

# MAGIC %md How about `education` status?

# COMMAND ----------

display(trainDF
        .groupBy("education")
        .count()
        .sort("count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Background: Transformers, estimators, and pipelines
# MAGIC 
# MAGIC Three important concepts in MLlib machine learning that are illustrated in this notebook are **Transformers**, **Estimators**, and **Pipelines**. 
# MAGIC 
# MAGIC - **Transformer**: Takes a DataFrame as input, and returns a new DataFrame. Transformers do not learn any parameters from the data and simply apply rule-based transformations to either prepare data for model training or generate predictions using a trained MLlib model. You call a transformer with a `.transform()` method.
# MAGIC 
# MAGIC - **Estimator**: Learns (or "fits") parameters from your DataFrame via a `.fit()` method and returns a Model, which is a transformer.
# MAGIC 
# MAGIC - **Pipeline**: Combines multiple steps into a single workflow that can be easily run. Creating a machine learning model typically involves setting up many different steps and iterating over them. Pipelines help you automate this process.
# MAGIC 
# MAGIC For more information:
# MAGIC [ML Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html#ml-pipelines)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. Feature preprocessing 
# MAGIC 
# MAGIC The goal of this notebook is to build a model that predicts the `income` level from the features included in the dataset (education level, marital status, occupation, and so on). The first step is to manipulate, or preprocess, the features so they are in the format MLlib requires.

# COMMAND ----------

# MAGIC %md ### Convert categorical variables to numeric
# MAGIC 
# MAGIC Some machine learning algorithms, such as linear and logistic regression, require numeric features. The Adult dataset includes categorical features such as education, occupation, and marital status. 
# MAGIC 
# MAGIC The following code block illustrates how to use `StringIndexer` and `OneHotEncoder` to convert categorical variables into a set of numeric variables that only take on values 0 and 1. 
# MAGIC 
# MAGIC - `StringIndexer` converts a column of string values to a column of label indexes. For example, it might convert the values "red", "blue", and "green" to 0, 1, and 2. 
# MAGIC - `OneHotEncoder` maps a column of category indices to a column of binary vectors, with at most one "1" in each row that indicates the category index for that row.
# MAGIC 
# MAGIC One-hot encoding in Spark is a two-step process. You first use the StringIndexer, followed by the OneHotEncoder. The following code block defines the StringIndexer and OneHotEncoder but does not apply it to any data yet.
# MAGIC 
# MAGIC For more information:   
# MAGIC [StringIndexer](http://spark.apache.org/docs/latest/ml-features.html#stringindexer)   
# MAGIC [OneHotEncoder](https://spark.apache.org/docs/latest/ml-features.html#onehotencoder)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder

categoricalCols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex"]

# The following two lines are estimators. They return functions that we will later apply to transform the dataset.
stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=[x + "Index" for x in categoricalCols]) 
encoder = OneHotEncoder(inputCols=stringIndexer.getOutputCols(), outputCols=[x + "OHE" for x in categoricalCols]) 

# The label column ("income") is also a string value - it has two possible values, "<=50K" and ">50K". 
# Convert it to a numeric value using StringIndexer.
labelToIndex = StringIndexer(inputCol="income", outputCol="label")

# COMMAND ----------

# MAGIC %md In this notebook, we'll build a pipeline combining all of our feature engineering and modeling steps. But let's take a minute to look more closely at how estimators and transformers work by applying the `stringIndexer` estimator that we created in the previous code block.
# MAGIC 
# MAGIC You can call the `.fit()` method to return a `StringIndexerModel`, which you can then use to transform the dataset. 
# MAGIC 
# MAGIC The `.transform()` method of `StringIndexerModel` returns a new DataFrame with the new columns appended. Scroll right to see the new columns if necessary. 
# MAGIC 
# MAGIC For more information: [StringIndexerModel](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StringIndexerModel.html)

# COMMAND ----------

stringIndexerModel = stringIndexer.fit(trainDF)
display(stringIndexerModel.transform(trainDF))

# COMMAND ----------

# MAGIC %md ### Combine all feature columns into a single feature vector
# MAGIC 
# MAGIC Most MLlib algorithms require a single features column as input. Each row in this column contains a vector of data points corresponding to the set of features used for prediction. 
# MAGIC 
# MAGIC MLlib provides the `VectorAssembler` transformer to create a single vector column from a list of columns.
# MAGIC 
# MAGIC The following code block illustrates how to use VectorAssembler.
# MAGIC 
# MAGIC For more information: [VectorAssembler](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# This includes both the numeric columns and the one-hot encoded binary vector columns in our dataset.
numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
assemblerInputs = [c + "OHE" for c in categoricalCols] + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

# MAGIC %md ## Step 3. Define the model
# MAGIC 
# MAGIC This notebook uses a [logistic regression](https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression) model.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol="label", regParam=1.0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4. Connect to AZML Workspace and create and experiment to track - using the MLFlow api

# COMMAND ----------

import mlflow
import mlflow.sklearn
import azureml.core
from azureml.core import Workspace
import matplotlib.pyplot as plt

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

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

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

experiment_name = "GettingStartedWithSparkMLlib-with-mlflow"
mlflow.set_experiment(experiment_name)

# Start the mlflow run, so we can log metrics and resutls in AZML Workspace experiment
mlflow.start_run() 

# COMMAND ----------

# MAGIC %md ## Step 5. Build the pipeline
# MAGIC 
# MAGIC A `Pipeline` is an ordered list of transformers and estimators. You can define a pipeline to automate and ensure repeatability of the transformations to be applied to a dataset. In this step, we define the pipeline and then apply it to the test dataset.
# MAGIC 
# MAGIC Similar to what we saw with `StringIndexer`, a `Pipeline` is an estimator. The `pipeline.fit()` method returns a `PipelineModel`, which is a transformer.
# MAGIC 
# MAGIC For more information:   
# MAGIC [Pipeline](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Pipeline)  
# MAGIC [PipelineModel](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/PipelineModel.html)

# COMMAND ----------

from pyspark.ml import Pipeline 

# Define the pipeline based on the stages created in previous steps.
pipeline = Pipeline(stages=[stringIndexer, encoder, labelToIndex, vecAssembler, lr])

# Define the pipeline model.
pipelineModel = pipeline.fit(trainDF)

# Apply the pipeline model to the test dataset.
predDF = pipelineModel.transform(testDF)

# COMMAND ----------

# MAGIC %md Display the predictions from the model. The `features` column is a [sparse vector](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.linalg.SparseVector), which is often the case after one-hot encoding, because there are so many 0 values.

# COMMAND ----------

display(predDF.select("features", "label", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md ## Step 5. Evaluate the model
# MAGIC 
# MAGIC The `display` command has a built-in ROC curve option.

# COMMAND ----------

display(pipelineModel.stages[-1], predDF.drop("prediction", "rawPrediction", "probability"), "ROC")

# COMMAND ----------

# MAGIC %md To evaluate the model, we use the `BinaryClassificationEvaluator` to evalute the area under the ROC curve and the `MulticlassClassificationEvaluator` to evalute the accuracy.
# MAGIC 
# MAGIC For more information:  
# MAGIC [BinaryClassificationEvaluator](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.BinaryClassificationEvaluator)  
# MAGIC [MulticlassClassificationEvaluator](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.MulticlassClassificationEvaluator)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

bcEvaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
roc = bcEvaluator.evaluate(predDF)
mlflow.log_metric('ROC', roc)
print(f"Area under ROC curve: {roc}")

mcEvaluator = MulticlassClassificationEvaluator(metricName="accuracy")
acc = mcEvaluator.evaluate(predDF)
mlflow.log_metric('ACCURACY', acc)
print(f"Accuracy: {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6. Hyperparameter tuning
# MAGIC 
# MAGIC MLlib provides methods to facilitate hyperparameter tuning and cross validation. 
# MAGIC - For hyperparameter tuning, `ParamGridBuilder` lets you define a grid search over a set of model hyperparameters.
# MAGIC - For cross validation, `CrossValidator` lets you specify an estimator (the pipeline to apply to the input dataset), an evaluator, a grid space of hyperparameters, and the number of folds to use for cross validation.
# MAGIC   
# MAGIC For more information:   
# MAGIC [Model selection using cross-validation](https://spark.apache.org/docs/latest/ml-tuning.html)  
# MAGIC [ParamGridBuilder](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.tuning)  
# MAGIC [CrossValidator](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator)   

# COMMAND ----------

# MAGIC %md
# MAGIC Use `ParamGridBuilder` and `CrossValidator` to tune the model. This example uses three values for `regParam` and three for `elasticNetParam`, for a total of 3 x 3 = 9 hyperparameter combinations for `CrossValidator` to examine. 

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .build())

# COMMAND ----------

# MAGIC %md Whenever you call `CrossValidator` in MLlib, Databricks automatically tracks all of the runs using [MLflow](https://mlflow.org/). You can use the MLflow UI ([AWS](https://docs.databricks.com/applications/mlflow/index.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/)) to compare how each model performed. To look at all the MLflow metrics tracked from the run, click on the `Experiment` icon to the far right above.
# MAGIC 
# MAGIC In this example we use the pipeline we created as the estimator.

# COMMAND ----------

# Create a 3-fold CrossValidator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=bcEvaluator, numFolds=3, parallelism = 4)

# Run cross validations. This step takes a few minutes and returns the best model found from the cross validation.
cvModel = cv.fit(trainDF)

# COMMAND ----------

# MAGIC %md ## Step 7. Make predictions and evaluate model performance
# MAGIC Use the best model identified by the cross-validation to make predictions on the test dataset, and then evaluate the model's performance using the area under the ROC curve. 

# COMMAND ----------

# Use the model identified by the cross-validation to make predictions on the test dataset
cvPredDF = cvModel.transform(testDF)

# Evaluate the model's performance based on area under the ROC curve and accuracy 
cvroc = bcEvaluator.evaluate(cvPredDF)
print(f"Area under ROC curve: {cvroc}")
mlflow.log_metric('CVROC', cvroc)

cvacc = mcEvaluator.evaluate(cvPredDF)
print(f"Accuracy: {cvacc}")
mlflow.log_metric('CVACCURACY', cvacc)

# COMMAND ----------

# MAGIC %md
# MAGIC Using SQL commands, you can also display predictions grouped by age and occupation. This requires creating a temporary view of the predictions dataset. 

# COMMAND ----------

cvPredDF.createOrReplaceTempView("finalPredictions")

# COMMAND ----------

# MAGIC %md ### Use SQL queries for displaying predictions

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT occupation, prediction, count(*) AS count
# MAGIC FROM finalPredictions
# MAGIC GROUP BY occupation, prediction
# MAGIC ORDER BY occupation

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT age, prediction, count(*) AS count
# MAGIC FROM finalPredictions
# MAGIC GROUP BY age, prediction
# MAGIC ORDER BY age

# COMMAND ----------

# MAGIC %md ### Register Model in Azure ML Model Registry

# COMMAND ----------

mlflow.spark.save_model(cvModel, "MLModelCVmodel")
model_local_path = os.path.abspath("MLModelCVmodel")
mlflow.register_model(f"file://{model_local_path}", "MLModelCVmodel")

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------


