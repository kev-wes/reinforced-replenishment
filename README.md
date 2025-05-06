
# reinforced-replenishment

## Overview

This is your new Kedro project, which was generated using `kedro 0.19.6`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to set up the kedro forecast project
The follow instructions will show you how to set up the kedro forecasting template for onboarding projects.

1. Install miniconda, postgresql and azure-cli with homebrew if not already installed:

```bash
brew install --cask miniconda
brew install postgresql
brew install azure-cli
```

2. Create conda environment with python=3.12:

```bash
conda create --name reinforced-replenishment python=3.12 -y
```

3. Activate conda environment in shell:
```bash
conda activate reinforced-replenishment 
```

**Important:** Make sure that the conda environment is set up in the IDE when the code is run.

4. Install requirements:
```bash
pip install -r requirements.txt
```
> *Note:* To add or remove dependencies to a project, edit the [requirements.txt](requirements.txt) file and run the above command again.

5. Activate nbstripout to ignore notebook output cells in git:
```bash
nbstripout --install
```

6. Install pre-commit git hooks:
```bash
pre-commit install
```

7. Rename the file [template.env](conf/local/template.env) to [.env](conf/local/.env) and pass the required credentials.

| Environment Variable | Bitwarden | Description |
| -------------------- | --------- | --------- |
| FORECAST_PIPELINE_URL |  | Forecast pipeline URL. The URL is predefined in the [template.env](conf/local/template.env) |
| BATCH_API_URL |  | Batch api URL. The URL is predefined in the [template.env](conf/local/template.env)|
| REDIS_HOST |  | Redis host for the redis message stream of the forecast pipeline. You can find the information in the compose.yaml of the forecastpipeline. |
| REDIS_PORT |  | Redis port for the redis message stream of the forecast pipeline. You can find the information in the compose.yaml of the forecastpipeline. |
| REDIS_DB |  | Redis db for the redis message stream of the forecast pipeline. You can find the information in the compose.yaml of the forecastpipeline. |
| REDIS_PASSWORD |  | Redis password for the redis message stream of the forecast pipeline. You can find the information in the compose.yaml of the forecastpipeline. By default the redis has no password and the environment variable can be omitted. |
| REDIS_HASH |  | Redis hash for the redis message stream of the forecast pipeline. You can find the information in the compose.yaml of the forecastpipeline. By default the hash value is `progress`. |
| POSTGRES_CONNECTION_STRING | workbench(fs-psql-cluster) | Postgres connection String from project database. |
| MLFLOW_TRACKING_URI |  | Mlflow tracking uri. The URI is predefined in the [template.env](conf/local/template.env).|
| BLOB_STORAGE_CONNECTION_STRING | forecastfileupload | Azure connection string from forecastfileupload blob storage.|

## How to run the kedro pipeline

If you are using **vscode** you can use the [launch.json](.vscode/launch.json) to run the pipelines. You can also run the pipelines in the console with `kedro run`, `kedro run -p data_processing` or `kedro run -p modelling`.

**Important!** When you start a kedro pipeline in the console or with an other IDE (e.g pycharm), you must make sure that the .env file is loaded, otherwise kedro will give an error.


## How to run forecasts
To run forecasts, you must start the forecastpipeline and the necessary services as Docker containers on your Macbook. The docker-compose file can be found in the [forecastpipeline repository](https://github.com/westphalia-datalab/forecastpipeline/blob/main/compose.yaml) in github.

Once you have downloaded the docker-compose file, update the environment variables `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER` and `POSTGRES_PASSWORD` with your database credentials you have already added in the [.env](conf/local/.env) file. Then open the terminal and navigate to the directory in which the file is saved to start the Docker containers with the command `docker-compose up -d`. Now all necessary Docker containers are running and you can start the local forecasts with the kedro modelling pipeline. Make sure that the configuration of the forecast api is set correctly. The url of the local forecast pipeline is `http://localhost:5000`.

##### Example:

```yaml
demo.forecast_api_config:
  validate: true
```

You can use the `validate` parameter to activate validation in order to recognise errors in the forecast configuration before you start a forecast calculation.

> *Note:* Make sure that the key `batch` has been removed from the forecast api configuration, otherwise the forecast will automatically be treated as a batch forecast.

## How to run batch forecasts
To run forecasts that incur high computational costs, it can be helpful to calculate the forecasts in an Azure batch process instead of in a local instance on your MacBook. To use the Azure batch process in your kedro modelling pipeline, you need to add the batch configuration in the forecast api configuration in your modelling parameters. 

##### Example:

```yaml
demo.forecast_api_config:
  validate: true
  batch:
    company_name: demo_company
    pool_id: small
    image_tag: ${latest_image:forecastpipeline}
```

To run a forecast in a batch process, you must specify the `company_name` (usually the customer's company name), the `pool_id` and the `image_tag`. You can use the `pool_id` to select the machine size to be used for calculating the forecasts. [Here](https://pacemaker-ai.atlassian.net/wiki/spaces/INF/pages/4427907073/Batch+Pools) you can see all available pool IDs. If you pass a pool_id that does not exist, an error is triggered. Please take care to the machine size. Select the machine size depending on your data size. With the `image_tag` you can select which forecast pipeline version is used for the forecast calculation. [Here](https://portal.azure.com/#view/Microsoft_Azure_ContainerRegistries/RepositoryBlade/id/%2Fsubscriptions%2Fbae0f85e-0d3b-4b4a-93f3-3d3591f9877a%2FresourceGroups%2Fprimary%2Fproviders%2FMicrosoft.ContainerRegistry%2Fregistries%2Fwdlcontainers/repository/forecastpipeline) you can find all available forecast pipeline versions.

> *Note:* To calculate forecast in batch, you must forward the batch nodes from kubernetes to your localhost. You can find instructions on how to forward the ports [here](#how-to-forward-ports-to-calculate-forecast-in-batch) 

## How to create a forecast configuration

To start a forecast in the forecast pipeline, you need a **forecast configuration**. This configuration is defined as a Kedro parameter in **YAML format** and is typically located in the `modelling` subfolder within the base environment's parameter directory. The forecast configuration is structured as follows:

##### Example 1: Single Configuration Object for running one configuration.
```yaml
demo.forecast_config:
  data_source:
    source_type: postgres-table
    source: demo-demand-forecast
    table_name: demo_forecast_table
  target: value
  date_column: date
  resolution: daily
  horizon: 12

  ...
```

 ##### Example 2: List of Configuration Objects to execute multiple configurations simultaneously
```yaml
demo.forecast_config:
  - data_source:
      source_type: postgres-table
      source: demo-demand-forecast
      table_name: demo_forecast_table
    target: value
    date_column: date
    resolution: daily
    horizon: 12

    ...
```

For a more detailed explanation of the different parameters within the forecast configuration, please refer to the Confluence page:  
[**How To: Build a Forecast Config**](https://pacemaker-ai.atlassian.net/wiki/spaces/PBP/pages/4476764164/How+To+Build+a+forecast+config+updated+to+v7.3.0).

## How to aggregate forecast results

Sometimes it is beneficial to calculate forecasts at one level and use them at another. For example, a forecast may be calculated at a daily level but evaluated on a monthly level. Using the **`forecast_result_config`**, you can adjust the aggregation level of forecast results after they are generated. This aggregation can involve different temporal levels or group-level aggregations. The configuration is defined as a Kedro parameter in **YAML format** and is typically located in the `modelling` subfolder within the base environment's parameter directory.

#### Examples of `forecast_result_config`

##### Example 1: No Changes to Aggregation
```yaml
demo.forecast_result_config:
```
- In this case, the forecast results remain at the same level as they were produced by the forecast pipeline. No additional aggregation is applied.

##### Example 2: Filter data
```yaml
demo.forecast_result_config:
  query: horizon == 1
```
- Filter the rows in the forecast result. Filter columns that can be used are the columns `date`, `horizon` and all grouping columns of the forecast (grouping columns are start with the prefix `group.`). [Here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) you can find more informations about the query syntax.

##### Example 3: Temporal Aggregation
```yaml
demo.forecast_result_config:
  result_aggregation:
    resolution: monthly
    incomplete_periods: false
    per_benchmark: false
```
- The results are aggregated to a **monthly level** using the `result_aggregation` object with the following parameter: 
  - **`resolution: monthly`**. Aggregation of the forecast on monthly resolution. 
     - `Weekly` and `monthly` resolutions are allowed.
     - Only forecasts calculated on a `daily` basis can be aggregated on a weekly or monthly basis.  
   - **`incomplete_periods: false`**: Incomplete periods (e.g., partial months) are excluded. If the forecast begins or ends within a period (week or month), these periods are not included in the results.
      - Setting this parameter to `true`, incomplete periods are aggregated and treated as a complete period and included in the results.
   - **`per_benchmark: false`**: Aggregation is performed across all benchmarks.  
     - Setting this parameter to `true` would aggregate results separately for each benchmark. If benchmarks overlap, this parameter should always be set to `true` to avoid mixing data.

##### Example 4: Temporal and Group Aggregation
```yaml
demo.forecast_result_config:
  group_columns:
  - federal_state
  result_aggregation:
    resolution: monthly
    incomplete_periods: false
    per_benchmark: false
```
- In addition to the monthly aggregation from Example 2:  
   - The results are also aggregated by the group column **`federal_state`** using the `group_columns` parameter, ignoring all other group columns. 

##### Example 5: Disable forecast status validation
```yaml
demo.forecast_result_config:
  disable_status_validation: True
  ...
```
- In some cases it can be helpfull to disable the forecast status validation and request the forecast results immediately with the parameter `disable_status_validation`. This means that the kedro pipeline does not wait for the 100% status. The results are requested immediately after the start of the `get_forecast_result` node. If the forecast calculation is not completed, an error is raised in the kedro pipeline. If the parameter is not set or is `false` the the status validation is enabled. 

> *Note:* The **`forecast_result_config`** object must always be present. If no post-processing aggregation is required, the configuration should be passed as shown in Example 1. 

## How to configure the evaluation configuration

The **evaluation configuration** allows you to define which error metrics should be calculated for the generated forecast. Various error metrics can be selected, calculated at different aggregation levels, and summarized using different aggregation functions. The evalutions are safed in the database under the name `*namespace*_forecast_metrics`.

The configuration is defined in **YAML parameter file**, where the forecast configuration is also specified. The structure of the configuration is as follows:

Example:

```yaml
demo.evaluation_config:
- metric: wape
  agg_func: weighted_mean
- metric: wape
  agg_func: weighted_mean
  pre_aggregation_groups: []
- metric: wape
  agg_func: weighted_mean
  pre_aggregation_groups:
  - city
- metric: wape
  agg_func: weighted_mean
  pre_aggregation_groups:
  - city
  pre_aggregation_freq: Q
  query: horizon == 3
  name: WAPE Quarterly Horizon 3
  metric_decimals: 3
```

The **`evaluation_config`** object contains a list of elements, each defining an error metric to be calculated. Each element in the list includes the following parameters:

-  **`metric`** (required):  
   Specifies the error metric to be calculated.

-  **`agg_func`** (optional):  
   Defines how the calculated errors are summarized into an overall error.  
   - The aggregation functions `mean`, `weighted_mean` and `median` are allowed.
   - Default value: `mean` (if not specified).

-  **`pre_aggregation_groups`** (optional):  
   Specifies the aggregation level of the data before the error is calculated.  
   - **`None`**: The error is calculated at the level of the forecast results.  
   - **Empty list (`[]`)**: The data is aggregated to the date level, and the error is calculated on the aggregated data.  
   - **Populated list (e.g., `["Column1", "Column2"]`)**: The data is aggregated to the level defined by the specified group columns and the date column before calculating the error.
-  **`pre_aggregation_freq`** (optional):  
   Specifies the resolution of the timeseries before the error is calculated. For full specification of available frequencies, please see [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
-  **`query`** (optional):  
   Filter the rows in the forecast result that are to be used for calculating the metric. Filter columns that can be used are the columns `date`, `horizon` and all grouping columns of the forecast (grouping columns are start with the prefix `group.`). [Here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) you can find more informations about the query syntax.
-  **`name`** (optional):  
   A user-defined metric name. If no name is specified the name is automatically created from the combination of `metric`, `agg_func` and `pre_aggregation_groups`.
-  **`metric_decimals`** (optional):  
   A user-defined rounding precision defined as decimal places. If not defined, no rounding is used.
   
## How to Handle Large Forecast Configurations in Kedro

When working with Kedro pipelines, forecast configurations can sometimes become quite large and complex, especially when using models like `FallbackChain`. In such cases, using [**OmegaConf custom resolvers**](https://omegaconf.readthedocs.io/en/2.3_branch/custom_resolvers.html#custom-resolvers) can simplify and automate rule-based YAML code generation.

With OmegaConf resolvers, you can dynamically generate values by referencing resolvers in your YAML configuration using the syntax: `${resolver:input}`.

### Available Custom Resolvers

#### 1. **`range`**
Creates a list of integers defined by a start, end, and step size. Useful for generating sequences like lag lists.

**Example:**
```yaml
preprocessing:
- name: add_lag
  kwargs:
    lags: ${range:2,13,2}
```
**Equivalent to:**
```yaml
preprocessing:
- name: add_lag
  kwargs:
    lags: [2, 4, 6, 8, 10, 12]
```

#### 2. **`append`**
Appends a value to a list. Useful for combining individual values with lists.

**Example:**
```yaml
preprocessing:
- name: add_lag
  kwargs:
    lags: ${append:${range:2,13,2},52}
```
**Equivalent to:**
```yaml
preprocessing:
- name: add_lag
  kwargs:
    lags: [2, 4, 6, 8, 10, 12, 52]
```

#### 3. **`list`**
Creates a list from a comma-separated string. Often combined with other resolvers.

**Example:**
```yaml
preprocessing:
- name: drop_columns
  kwargs:
    columns: ${list:Feature_1,Feature_2,Feature_3}
```
**Equivalent to:**
```yaml
preprocessing:
- name: drop_columns
  kwargs:
    columns:
    - Feature_1
    - Feature_2
    - Feature_3
```

#### 4. **`extend`**
Extends one list with another. Useful for merging lists into a single list.

**Example:**
```yaml
preprocessing:
- name: add_lag
  kwargs:
    lags: ${extend:${list:50,51,52},${range:2,13,2}}
```
**Equivalent to:**
```yaml
preprocessing:
- name: add_lag
  kwargs:
    lags: [50, 51, 52, 2, 4, 6, 8, 10, 12]
```

#### 5. **`add_lag`**
Creates a complete `add_lag` preprocessor object for the forecast pipeline.

**Example:**
```yaml
preprocessing:
- ${add_lag:Feature_1,${range:2,13,2}}
```
**Equivalent to:**
```yaml
preprocessing:
- name: add_lag
  kwargs:
    lags: [2, 4, 6, 8, 10, 12]
    feature: Feature_1
```

#### 6. **`add_lags`**
Creates a list of `add_lag` preprocessor objects for multiple features in the forecast pipeline.

**Example:**
```yaml
preprocessing: ${add_lags:${list:Feature_1,Feature_2},${range:2,13,2}}
```
**Equivalent to:**
```yaml
preprocessing:
- name: add_lag
  kwargs:
    lags: [2, 4, 6, 8, 10, 12]
    feature: Feature_1
- name: add_lag
  kwargs:
    lags: [2, 4, 6, 8, 10, 12]
    feature: Feature_2
```

#### 7. **`create_lagged_feature_names`**
Generates lagged feature names by combining two lists. The first list contains feature names, and the second contains feature lags. A third parameter specifies the separator between the feature name and lag.

**Example:**
```yaml
explaining_prediction_models:
- name: FallbackChain
  kwargs:
    depend_columns:
    - ${create_lagged_feature_names:${list:Feature_1},${range:2,13,2},_Lag_}
```
**Equivalent to:**
```yaml
explaining_prediction_models:
- name: FallbackChain
  kwargs:
    depend_columns:
    - [Feature_1_Lag_2, Feature_1_Lag_4, Feature_1_Lag_6, Feature_1_Lag_8, Feature_1_Lag_10, Feature_1_Lag_12]
```
#### 8. **`assign_value_to_keys`**
Creates a dictionary by assigning a single value to all elements in a list. The first argument is the list of keys, and the second argument is the value assigned to all keys.

**Example:**
```yaml
aggregation_functions: ${assign_value_to_keys:${create_lagged_feature_names:${list:Feature_1},${range:2,13,2},_Lag_},sum}
```
**Equivalent to:**
```yaml
aggregation_functions:
  Feature_1_Lag_2: sum
  Feature_1_Lag_4: sum
  Feature_1_Lag_6: sum
  Feature_1_Lag_8: sum
  Feature_1_Lag_10: sum
  Feature_1_Lag_12: sum
```

#### 9. **`assign_values_to_keys`**
Creates a dictionary by pairing elements from two lists. The first list provides the keys, and the second list provides the corresponding values. Both lists must have the same length.

**Example:**
```yaml
preprocessing:
- name: set_feature_horizon
  kwargs:
    feature_horizon: ${assign_values_to_keys:${create_lagged_feature_names:${list:Feature_1},${range:2,13,2},_Lag_},${range:2,13,2}}
```
**Equivalent to:**
```yaml
preprocessing:
- name: set_feature_horizon
  kwargs:
    feature_horizon:
      Feature_1_Lag_2: 2
      Feature_1_Lag_4: 4
      Feature_1_Lag_6: 6
      Feature_1_Lag_8: 8
      Feature_1_Lag_10: 10
      Feature_1_Lag_12: 12
```

#### 10. **`generate_model_configs`**
Creates a list of forecast configurations. Generates all possible forecast configurations by applying parameter combinations to all occurrences
    of a model.

**Example:**
```yaml
demo_forecast_config:
  ...
  pipeline:
    name: DefaultGlobalPredictionModel
    kwargs:
      objective: regression
      eta_end: 0.1
      scaler: 

model_params:
- DefaultGlobalPredictionModel:
    objective: [regression, tweedie]
    num_leaves: [10, 30]

demo.forecast_config: ${generate_forecast_configs:${demo_forecast_config},${model_params}}
```
**Equivalent to:**
```yaml
demo.forecast_config:
- ...
  pipeline:
    name: DefaultGlobalPredictionModel
    kwargs:
      objective: regression
      eta_end: 0.1
      num_leaves: 10
      scaler:
- ...
  pipeline:
    name: DefaultGlobalPredictionModel
    kwargs:
      objective: regression
      eta_end: 0.1
      num_leaves: 30
      scaler: 
- ...
  pipeline:
    name: DefaultGlobalPredictionModel
    kwargs:
      objective: tweedie
      eta_end: 0.1
      num_leaves: 10
      scaler: 
- ...
  pipeline:
    name: DefaultGlobalPredictionModel
    kwargs:
      objective: tweedie
      eta_end: 0.1
      num_leaves: 30
      scaler: 
```

You can find all available custom resolvers [here](src/reinforced_replenishment/nodes/utils/yaml_resolver.py).

### Reusing YAML Code in Resolvers
You can also pass existing YAML code as input to an OmegaConf resolver. This is useful for reusing code, such as column names, in multiple configurations.

**Example:**
```yaml
column_names: [Feature_1, Feature_2]
preprocessing: ${add_lags:${column_names},${range:2,13,2}}
```

By utilizing these resolvers, you can greatly simplify and streamline your forecast pipeline configurations in Kedro. This approach reduces redundancy and enhances maintainability of your YAML configurations.


## How to work with Kedro and notebooks

### Jupyter

To get access to the kedro catalog, context, pipeline and session variables you can use the %load_ext line magic to explicitly load the Kedro IPython extension:

```python
from dotenv import load_dotenv

load_dotenv("../conf/local/.env", override=True)

%load_ext kedro.ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## How to track your forecast results with mlflow
To enable the mlflow tracking of the forecast results you must comment out the hook in row 6 and 7 in the [settings.py](src/reinforced_replenishment/settings.py). After the first run of the modelling pipeline you will find your experiment with the name reinforced-replenishment at (http://localhost:8080). 

> *Note:* If you received an azure token error when uploading the experiment data to mlflow you may not logged in with your azure account on your pc. Try to login with the command line `az login`. Use the default azure subscription (Pacemaker).  

## FAQ

### How to export a jupyter notebook file to html?

```bash
jupyter nbconvert notebooks/example.ipynb --no-input --to html --output-dir="data/08_reporting"
```

### How to run the get_forecast_results node with an old forecast_key version?

To run a node with an old versioned dataset you can add the parameter --load-versions to the run command and pass the required dataset name with the specific version. The dataset name and version is seperated by a colon. To run the get_forecast_results node with an old forecast_key you have to overwrite the dataset forecast_key@json. 

Example:
```bash
kedro run --pipeline modelling --from-nodes demo.get_forecast_results_node --load-versions demo.forecast_key@json:YYYY-MM-DDThh.mm.ss.sssZ
```

If you are using vscode you will find a placeholder run configuration in the [launch.json](.vscode/launch.json).

### How to forward ports to calculate forecast in batch?

To calculate a forecast in batch, you need a port forwarding to the necessary batch pods in Kuberenetes. To forward the ports on your localhost, you can use the following bash script:

```bash
#!/bin/bash
if ! command -v kubectl &> /dev/null; then
  echo "kubectl is not installed. Please install it and retry."
  exit 1
fi

# Initialize variables
CONTEXT="aks-prod-gwc"
NAMESPACE="projects"

# Set the kubectl context
kubectl config use-context "$CONTEXT"

echo "Forwarding to $CONTEXT instance using namespace $NAMESPACE"

POD_NAME_PIPELINE=$(kubectl get pod -n "$NAMESPACE" | grep -m1 "$NAMESPACE-pipeline" | awk '{print $1}')
POD_NAME_REDIS=$(kubectl get pod -n "$NAMESPACE" | grep -m1 "$NAMESPACE-redis" | awk '{print $1}')
POD_NAME_BATCH_API=$(kubectl get pod -n "$NAMESPACE" | grep -m1 "$NAMESPACE-batch" | awk '{print $1}')
POD_NAME_MLFLOW=$(kubectl get pod -n "$NAMESPACE" | grep -m1 "$NAMESPACE-mlflow" | awk '{print $1}')

if [[ -z "$POD_NAME_PIPELINE" || -z "$POD_NAME_REDIS" || -z "$POD_NAME_BATCH_API" || -z "$POD_NAME_MLFLOW" ]]; then
  echo "Failed to retrieve one or more pod names. Exiting..."
  exit 1
fi

trap 'kill $(jobs -p)' EXIT

echo "Port-forwarding for $POD_NAME_PIPELINE on port 5000"
kubectl port-forward pods/"$POD_NAME_PIPELINE" 5000:5000 -n "$NAMESPACE" &
echo "Port-forwarding for $POD_NAME_REDIS on port 6379"
kubectl port-forward pods/"$POD_NAME_REDIS" 6379:6379 -n "$NAMESPACE" &
echo "Port-forwarding for $POD_NAME_BATCH_API on port 8000"
kubectl port-forward pods/"$POD_NAME_BATCH_API" 8000:8000 -n "$NAMESPACE" &
echo "Port-forwarding for $POD_NAME_MLFLOW on port 8080"
kubectl port-forward pods/"$POD_NAME_MLFLOW" 8080:8080 -n "$NAMESPACE"
```

Follow these steps to run the Bash script and enable port forwarding:

1. **Create the Bash Script**
   - Save the provided script in a file (e.g., `forward-batch-ports`).

2. **Navigate to the Directory**
   - Open your terminal and change to the directory where you saved the script:
     ```sh
     cd /path/to/your/script
     ```

3. **Make the Script Executable**
   - Run the following command to make the script executable:
     ```sh
     chmod +x forward-batch-ports
     ```

4. **Move the script to a systempath**
    - Move the script in `/usr/local/bin/` to make it executable from anywhere:
      ```sh
      mv forward-batch-ports /usr/local/bin/
      ```
4. **Start Port Forwarding**
   - Start the port forwarding by running:
     ```sh
     forward-batch-ports
     ```

   - The following ports will be forwarded to your localhost:
     - (http://localhost:5000) (Forecast Pipeline)
     - (http://localhost:8000) (Batch API)
     - (http://localhost:8080) (MLflow)
     - (http://localhost:6379) (Redis)

5. **Stop Port Forwarding**
   - You can stop the forwarding anytime by pressing `CTRL + C` in the terminal.
