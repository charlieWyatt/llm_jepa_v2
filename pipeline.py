from kfp.v2 import compiler
from kfp.v2.dsl import component, pipeline
from google.cloud import aiplatform


@component(
    base_image='python:3.10',
    packages_to_install=[
        '-r requirements.txt'
    ],
)
def run_training():
    import subprocess
    import sys
    import os

    sys.path.append(os.path.abspath("."))

    subprocess.run(["python", "train.py"], check=True)


@pipeline(name="my-training-pipeline")
def my_pipeline():
    run_training()


if __name__ == "__main__":
    aiplatform.init(project="innate-algebra-429308-s8", location="us-central1")

    compiler.Compiler().compile(
        pipeline_func=my_pipeline,
        package_path="pipeline.json"
    )

    aiplatform.PipelineJob(
        display_name="training-job",
        template_path="pipeline.json",
        pipeline_root="gs://llm_jepa/pipeline-root/",
    ).run()
