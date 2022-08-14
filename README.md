## To run the Demo

1.	In Azure DevOps: Create a new project called: MLOps-AzDevOps-Databricks-AZML
2.	In Azure DevOps: Import(clone) this github repo to Azure Devops Repo
3.  In Azure Portal: Create a Databricks instance called mlops-azdevops-db-azml, on a new resource group called mlops-azdevops-db-azml_rg (you can pick other names, just remember it)
4.  In Azure Portal: In Azure Active Directory->App Registrations, Create an App Registration (Service Principal), create a Secret (and take not of the value) and add contributor permissions to the resource group on Step 3. 
5.  In Azure Portal: Go to the Resource Group created on Step 3, Then go to Access Control (IAM)->Add->Add Role Assignments->Contributor->Members->+Select Members-> Search for your App on Step 4 -> Select -> Review/Assign
6.  In Azure Portal: Create an Azure Machine Learning Workspace, on the same resource group of Step 3.
7.	In Databricks:   Repos->Add Repo, add the Azure Devos Repo from step 2. Select as Git Provider: Azure DevOps Services
8.	In Azure Portal: Create an Azure KeyVault, on the same resource group of step 3, with the SECRETS stated below
9.	In Azure DevOps: In Pipelines->Library, Create a Variable Group in Pipeline->Library called: mlops-azdevops-db-azml-vg, and link Secrets from Azure Key Vault as variables. Add all the variables created on Step 8.
10.	In Azure DevOps: In Artifacts, Create a FEED called: mlops-azdevops-db-azml
11.	In Azure DevOps: In Pipelines, Run the Pipeline


## Pipeline will need a Azure Secret Vault with the following secrets.
<br>
***DBXInstance***: Databricks instance, eg: adb-631237481529976.16<br>
***ResourceGroup***: Resource Group where Databricks instance is<br>
***SubscriptionID***: Azure Subscription ID where Databricks instance is<br>
***SVCApplicationID***: Application (client) ID for the Service Principal (app registration in Azure AD)<br>
***SVCDirectoryID***: Directory (tenant) ID for the Service Principal<br>
***SVCSecretKey***: Secret value for the Service Principal<br>
***WorkspaceName***: Name of the Databricks Workspace<br>
***AzmlResourceGroup***: Azure Machine Learning Resource Group<br>
***AzmlWORKSPACEname***: Name of the Azure Machine Learning Workspace<br>
