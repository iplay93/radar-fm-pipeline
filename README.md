# radar-fm-pipeline

Foundation Model Pipeline for Wearable Sensor Data

radar-fm-pipeline is a plug and play pipeline for building, adapting, and deploying foundation models on wearable sensor data.

It is designed for environments where data characteristics, model architectures, downstream tasks, and execution environments continuously change, which is the default in real world wearable sensing.

## Why radar-fm-pipeline

Wearable sensor systems introduce challenges that conventional machine learning pipelines do not handle well.

- Heterogeneous sensors and device specific signal characteristics
- Frequent missing data and large variations in signal quality
- Sparse, weak, or task dependent labels
- Multiple downstream tasks derived from a single foundation model
- Deployment across server, mobile, and edge environments

As these factors evolve, tightly coupled codebases require constant refactoring.

radar-fm-pipeline addresses this by enforcing clear module boundaries and standard interfaces, enabling plug and play composition of data processing, model architectures, task adaptation, and deployment logic.

## Design Philosophy

- Change is expected: data, models, tasks, and environments evolve over time
- Isolate what changes: changing components are implemented as independent modules
- Foundation models are systems: training, adaptation, inference, and deployment are treated as a unified pipeline
- Plug and play by design: components can be swapped without modifying the core pipeline

## Repository Structure

```text
project-root/
├── datasets/          # Data loading, standardization, quality handling, and windowing
├── models/            # Model architecture definitions (Transformer, Mamba, multi-modal heads)
├── losses/            # Training objectives (contrastive, MSE, masked prediction)
├── metrics/           # Performance metrics (F1, AUROC, time-series metrics)
├── trainer/           # Training engine (DDP, checkpointing, logging)
├── adaptation/        # Downstream adaptation (PEFT, prompt tuning, fine-tuning)
├── inference/         # Inference pipeline and runtime policies
├── configs/           # YAML or Hydra based experiment configurations
├── scripts/           # Executable entry points (train, eval, export)
├── utils/             # Shared utilities (reproducibility, logging)
└── tests/             # Unit tests and data pipeline integrity checks
```

## Key Features

- Modular plug and play architecture for wearable foundation models
- Support for self supervised pretraining and downstream task adaptation
- Clear separation between data processing, modeling, training, and inference
- Configuration driven workflows for reproducible experimentation
- Designed to scale across datasets, tasks, and deployment targets

## Extending the Pipeline

radar-fm-pipeline is designed to be extended without refactoring the core codebase.

- Add a new dataset by extending `datasets/`
- Add a new model architecture by extending `models/`
- Add a new downstream task by extending `adaptation/`
- Customize inference and deployment logic in `inference/`

Each component follows a well defined interface, enabling efficient experimentation and system evolution.

## Intended Use Cases

- Wearable foundation model research
- Multi task learning on time series sensor data
- On device and edge AI deployment
- Rapid prototyping across heterogeneous wearable datasets

## Summary

radar-fm-pipeline provides a scalable and maintainable foundation for wearable foundation models by treating data, models, tasks, and deployment as a single system rather than isolated components.
