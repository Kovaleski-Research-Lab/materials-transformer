from kfp import dsl
from kfp.client import Client

@dsl.component(
    base_image="kovaleskilab/ml-uv:latest",
)
def training_job_component(
    # parameters that kubeflow can pass to the job i.e. learning_rate, epochs
):
    pass

@dsl.pipeline(
    name="Materials Transformer Training Pipeline",
    description="A pipeline to train the materials transformer model."
)
def training_pipeline():
    training_task = training_job_component()
    
    # 1. Set CPU and Memory requests and limits
    training_task.set_cpu_request("24")
    training_task.set_cpu_limit("24")
    training_task.set_memory_request("200Gi")
    training_task.set_memory_limit("200Gi")
    
    # 2. Request an A100 specifically
    training_task.add_resource_limit("nvidia.com/a100", "1")
    
    # 3. mount persistent volume claims
    training_task.add_pvc_volume(
        pvc_name="nfe-data",
    )
    
    # continuing...