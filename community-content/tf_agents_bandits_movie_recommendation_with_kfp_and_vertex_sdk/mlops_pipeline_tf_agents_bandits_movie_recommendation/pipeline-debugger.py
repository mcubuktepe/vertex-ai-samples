import os
from subprocess import PIPE
import sys

from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2 import compiler, dsl
#from kfp.v2.google.client import AIPlatformClient
from google.cloud.aiplatform.pipeline_jobs import PipelineJob

# The Google Cloud Notebook product has specific requirements
IS_GOOGLE_CLOUD_NOTEBOOK = os.path.exists("/opt/deeplearning/metadata/env_version")

# Google Cloud Notebook requires dependencies to be installed with '--user'
USER_FLAG = ""
if IS_GOOGLE_CLOUD_NOTEBOOK:
    USER_FLAG = "--user"

PROJECT_ID = "silent-cider-335420"  # @param {type:"string"}

from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

from google.cloud import aiplatform

import google.auth

credentials, project_id = google.auth.default()

PROJECT_ID = "silent-cider-335420"
BUCKET_NAME = "gs://silent-cider-335420-bucket/"
REGION = "us-central1"  

request = google.auth.transport.requests.Request()

#credentials.refresh(request)
aiplatform.init(project=PROJECT_ID,location=REGION)
# BigQuery parameters (used for the Generator, Ingester, Logger)
BIGQUERY_DATASET_ID = f"{PROJECT_ID}.movielens_dataset"  # @param {type:"string"} BigQuery dataset ID as `project_id.dataset_id`.
BIGQUERY_LOCATION = "us"  # @param {type:"string"} BigQuery dataset region.
BIGQUERY_TABLE_ID = f"{BIGQUERY_DATASET_ID}.training_dataset"  # @param {type:"string"} BigQuery table ID as `project_id.dataset_id.table_id`.


# Dataset parameters
RAW_DATA_PATH = "gs://cloud-samples-data/vertex-ai/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/u.data"  # Location of the MovieLens 100K dataset's "u.data" file.

# Pipeline parameters
PIPELINE_NAME = "movielens-pipeline"  # Pipeline display name.
ENABLE_CACHING = False  # Whether to enable execution caching for the pipeline.
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline"  # Root directory for pipeline artifacts.
PIPELINE_SPEC_PATH = "metadata_pipeline.json"  # Path to pipeline specification file.
OUTPUT_COMPONENT_SPEC = "output-component.yaml"  # Output component specification file.

# BigQuery parameters (used for the Generator, Ingester, Logger)
BIGQUERY_TMP_FILE = (
    "tmp.json"  # Temporary file for storing data to be loaded into BigQuery.
)
BIGQUERY_MAX_ROWS = 5  # Maximum number of rows of data in BigQuery to ingest.

# Dataset parameters
TFRECORD_FILE = (
    f"{BUCKET_NAME}/trainer_input_path/*"  # TFRecord file to be used for training.
)

# Logger parameters (also used for the Logger hook in the prediction container)
LOGGER_PUBSUB_TOPIC = "logger-pubsub-topic"  # Pub/Sub topic name for the Logger.
LOGGER_CLOUD_FUNCTION = "logger-cloud-function"  # Cloud Functions name for the Logger.

# Trainer parameters
TRAINING_ARTIFACTS_DIR = (
    f"{BUCKET_NAME}/artifacts"  # Root directory for training artifacts.
)
TRAINING_REPLICA_COUNT = "1"  # Number of replica to run the custom training job.
TRAINING_MACHINE_TYPE = (
    "n1-standard-4"  # Type of machine to run the custom training job.
)
TRAINING_ACCELERATOR_TYPE = "ACCELERATOR_TYPE_UNSPECIFIED"  # Type of accelerators to run the custom training job.
TRAINING_ACCELERATOR_COUNT = "0"  # Number of accelerators for the custom training job.

# Deployer parameters
TRAINED_POLICY_DISPLAY_NAME = (
    "movielens-trained-policy"  # Display name of the uploaded and deployed policy.
)
ENDPOINT_DISPLAY_NAME = "movielens-endpoint"  # Display name of the prediction endpoint.
ENDPOINT_MACHINE_TYPE = "n1-standard-4"  # Type of machine of the prediction endpoint.

# Prediction container parameters
PREDICTION_CONTAINER = "prediction-container"  # Name of the container image.
PREDICTION_CONTAINER_DIR = "src/prediction_container"

cloudbuild_yaml = """steps:
- name: "gcr.io/kaniko-project/executor:latest"
  args: ["--destination=gcr.io/{PROJECT_ID}/{PREDICTION_CONTAINER}:latest",
         "--cache=true",
         "--cache-ttl=99h"]
  env: ["AIP_STORAGE_URI={ARTIFACTS_DIR}",
        "PROJECT_ID={PROJECT_ID}",
        "LOGGER_PUBSUB_TOPIC={LOGGER_PUBSUB_TOPIC}"]
options:
  machineType: "E2_HIGHCPU_8"
""".format(
    PROJECT_ID=PROJECT_ID,
    PREDICTION_CONTAINER=PREDICTION_CONTAINER,
    ARTIFACTS_DIR=TRAINING_ARTIFACTS_DIR,
    LOGGER_PUBSUB_TOPIC=LOGGER_PUBSUB_TOPIC,
)

with open(f"{PREDICTION_CONTAINER_DIR}/cloudbuild.yaml", "w") as fp:
    fp.write(cloudbuild_yaml)

print(1)

from kfp.components import load_component_from_url

generate_op = load_component_from_url(
    "https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/68d6cf46ee22a9b9295d62ea71996150baf8db94/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/mlops_pipeline_tf_agents_bandits_movie_recommendation/src/generator/component.yaml"
)
ingest_op = load_component_from_url(
    "https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/68d6cf46ee22a9b9295d62ea71996150baf8db94/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/mlops_pipeline_tf_agents_bandits_movie_recommendation/src/ingester/component.yaml"
)
train_op = load_component_from_url(
    "https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/68d6cf46ee22a9b9295d62ea71996150baf8db94/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/mlops_pipeline_tf_agents_bandits_movie_recommendation/src/trainer/component.yaml"
)


@dsl.pipeline(pipeline_root=PIPELINE_ROOT, name=f"{PIPELINE_NAME}-startup")
def pipeline(
    # Pipeline configs
    project_id: str,
    raw_data_path: str,
    training_artifacts_dir: str,
    # BigQuery configs
    bigquery_dataset_id: str,
    bigquery_location: str,
    bigquery_table_id: str,
    bigquery_max_rows: int = 10000,
    # TF-Agents RL configs
    batch_size: int = 8,
    rank_k: int = 20,
    num_actions: int = 20,
    driver_steps: int = 3,
    num_epochs: int = 5,
    tikhonov_weight: float = 0.01,
    agent_alpha: float = 10,
) -> None:
    """Authors a RL pipeline for MovieLens movie recommendation system.

    Integrates the Generator, Ingester, Trainer and Deployer components. This
    pipeline generates initial training data with a random policy and runs once
    as the initiation of the system.

    Args:
      project_id: GCP project ID. This is required because otherwise the BigQuery
        client will use the ID of the tenant GCP project created as a result of
        KFP, which doesn't have proper access to BigQuery.
      raw_data_path: Path to MovieLens 100K's "u.data" file.
      training_artifacts_dir: Path to store the Trainer artifacts (trained policy).

      bigquery_dataset: A string of the BigQuery dataset ID in the format of
        "project.dataset".
      bigquery_location: A string of the BigQuery dataset location.
      bigquery_table_id: A string of the BigQuery table ID in the format of
        "project.dataset.table".
      bigquery_max_rows: Optional; maximum number of rows to ingest.

      batch_size: Optional; batch size of environment generated quantities eg.
        rewards.
      rank_k: Optional; rank for matrix factorization in the MovieLens environment;
        also the observation dimension.
      num_actions: Optional; number of actions (movie items) to choose from.
      driver_steps: Optional; number of steps to run per batch.
      num_epochs: Optional; number of training epochs.
      tikhonov_weight: Optional; LinUCB Tikhonov regularization weight of the
        Trainer.
      agent_alpha: Optional; LinUCB exploration parameter that multiplies the
        confidence intervals of the Trainer.
    """
    # Run the Generator component.
    generate_task = generate_op(
        project_id=project_id,
        raw_data_path=raw_data_path,
        batch_size=batch_size,
        rank_k=rank_k,
        num_actions=num_actions,
        driver_steps=driver_steps,
        bigquery_tmp_file=BIGQUERY_TMP_FILE,
        bigquery_dataset_id=bigquery_dataset_id,
        bigquery_location=bigquery_location,
        bigquery_table_id=bigquery_table_id,
    )

    # Run the Ingester component.
    ingest_task = ingest_op(
        project_id=project_id,
        bigquery_table_id=generate_task.outputs["bigquery_table_id"],
        bigquery_max_rows=bigquery_max_rows,
        tfrecord_file=TFRECORD_FILE,
    )

    # Run the Trainer component and submit custom job to Vertex AI.
    train_task = train_op(
        training_artifacts_dir=training_artifacts_dir,
        tfrecord_file=ingest_task.outputs["tfrecord_file"],
        num_epochs=num_epochs,
        rank_k=rank_k,
        num_actions=num_actions,
        tikhonov_weight=tikhonov_weight,
        agent_alpha=agent_alpha,
    )

    worker_pool_specs = [
        {
            "containerSpec": {
                "imageUri": train_task.container.image,
            },
            "replicaCount": TRAINING_REPLICA_COUNT,
            "machineSpec": {
                "machineType": TRAINING_MACHINE_TYPE,
                "acceleratorType": TRAINING_ACCELERATOR_TYPE,
                "acceleratorCount": TRAINING_ACCELERATOR_COUNT,
            },
        },
    ]
    train_task.custom_job_spec = {
        "displayName": train_task.name,
        "jobSpec": {
            "workerPoolSpecs": worker_pool_specs,
        },
    }

    # Run the Deployer components.
    # Upload the trained policy as a model.
    model_upload_op = gcc_aip.ModelUploadOp(
        project=project_id,
        display_name=TRAINED_POLICY_DISPLAY_NAME,
        artifact_uri=train_task.outputs["training_artifacts_dir"],
        serving_container_image_uri=f"gcr.io/{PROJECT_ID}/{PREDICTION_CONTAINER}:latest",
    )
    # Create a Vertex AI endpoint. (This operation can occur in parallel with
    # the Generator, Ingester, Trainer components.)
    endpoint_create_op = gcc_aip.EndpointCreateOp(
        project=project_id, display_name=ENDPOINT_DISPLAY_NAME
    )
    # Deploy the uploaded, trained policy to the created endpoint. (This operation
    # has to occur after both model uploading and endpoint creation complete.)
    gcc_aip.ModelDeployOp(
        # project=project_id,
        endpoint=endpoint_create_op.outputs["endpoint"],
        model=model_upload_op.outputs["model"],
        deployed_model_display_name=TRAINED_POLICY_DISPLAY_NAME,
        # machine_type=ENDPOINT_MACHINE_TYPE,
    )

# Compile the authored pipeline.
compiler.Compiler().compile(pipeline_func=pipeline, package_path=PIPELINE_SPEC_PATH)

# Create a Vertex AI client.
#from kfp.v2.google.client import AIPlatformClient
#api_client = AIPlatformClient(project_id=PROJECT_ID, region=REGION)

api_client = PipelineJob(display_name=PIPELINE_NAME,template_path=PIPELINE_SPEC_PATH,location=REGION,project=PROJECT_ID,
    parameter_values={
        # Pipeline configs
        "project_id": PROJECT_ID,
        "raw_data_path": RAW_DATA_PATH,
        "training_artifacts_dir": TRAINING_ARTIFACTS_DIR,
        # BigQuery configs
        "bigquery_dataset_id": BIGQUERY_DATASET_ID,
        "bigquery_location": BIGQUERY_LOCATION,
        "bigquery_table_id": BIGQUERY_TABLE_ID,
    },
    job_id=f"pipeline-job-{TIMESTAMP}",
    enable_caching=ENABLE_CACHING)

# Create a pipeline run job.
response = api_client.run(
    service_account='503659053462-compute@developer.gserviceaccount.com',
)


print(1)