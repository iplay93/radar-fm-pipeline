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
