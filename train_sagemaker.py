from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.pytorch import PyTorch

S3_BUCKET_NAME = "bucket-name"


def start_training():
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path=f"s3://{S3_BUCKET_NAME}/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard",
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role="your-execution-role",
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={"batch-size": 32, "epochs": 25},
        tensorboard_config=tensorboard_config,
    )

    # Start training
    estimator.fit(
        {
            "training": f"s3://{S3_BUCKET_NAME}/dataset/train",
            "validation": f"s3://{S3_BUCKET_NAME}/dataset/dev",
            "test": f"s3://{S3_BUCKET_NAME}/dataset/test",
        }
    )


if __name__ == "__main__":
    start_training()
