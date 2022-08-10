import requests
import time
import os
import json

DBRKS_REQ_HEADERS = {
    'Authorization': 'Bearer ' + os.environ['DBRKS_BEARER_TOKEN'],
    'X-Databricks-Azure-Workspace-Resource-Id': '/subscriptions/'+ os.environ['DBRKS_SUBSCRIPTION_ID'] +'/resourceGroups/'+ os.environ['DBRKS_RESOURCE_GROUP'] +'/providers/Microsoft.Databricks/workspaces/' + os.environ['DBRKS_WORKSPACE_NAME'],
    'X-Databricks-Azure-SP-Management-Token': os.environ['DBRKS_MANAGEMENT_TOKEN']}

DBRKS_BASE_URL = "https://"+os.environ['DBRKS_INSTANCE']+".azuredatabricks.net/"
DBRKS_SUBMIT_ENDPOINT = 'api/2.0/jobs/runs/submit'

postjson = {
  "run_name": 'automated_devops_run',
  "existing_cluster_id": os.environ["DBRKS_CLUSTER_ID"],
  "notebook_task": 
    {
      "notebook_path": '/Shared/' + os.environ['NOTEBOOK_NAME']
    }
}

response = requests.post( DBRKS_BASE_URL + DBRKS_SUBMIT_ENDPOINT, headers=DBRKS_REQ_HEADERS, json=postjson)
if response.status_code != 200:
    raise Exception(response.text)

print(response.status_code)

print(response.text)

json_payload = json.loads(response.content) # run_id

# Now we wait until the job is done / Notebook is done

def get_job_info():
    DBRKS_GET_ENDPOINT = 'api/2.1/jobs/runs/get'
    response = requests.get("https://"+os.environ['DBRKS_INSTANCE']+".azuredatabricks.net/" + DBRKS_GET_ENDPOINT, headers=DBRKS_REQ_HEADERS, json=json_payload)
    if response.status_code == 200:
        return json.loads(response.content)
    else:
        raise Exception(json.loads(response.content))


def get_job_output():
    DBRKS_OUTPUT_ENDPOINT = 'api/2.1/jobs/runs/get-output'
    response = requests.post("https://"+os.environ['DBRKS_INSTANCE']+".azuredatabricks.net/" + DBRKS_OUTPUT_ENDPOINT, headers=DBRKS_REQ_HEADERS, json=json_payload)
    if response.status_code == 200:
        return json.loads(response.content)
    else:
        raise Exception(json.loads(response.content))
        

def manage_job_state():
    await_job = True
    start_time = time.time()
    loop_time = 600  # 10 Minutes
    while await_job:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > loop_time:
            raise Exception('Error: Loop took over {} seconds to run.'.format(loop_time))
        if get_job_info()['state']['life_cycle_state'] == 'TERMINATED':
            print('Job is completed')
            #print(get_job_output()['notebook_output']) ## This is throwing an endpoint doesnt exist error, don't know yet why
            await_job = False
        else:
            time.sleep(20)
            
            
manage_job_state()


