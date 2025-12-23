# SketchPlacement

This repository contains code for sketch placement algorithms.

## Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Dataset Generation

To generate synthetic flow/path datasets, use the scripts in the `dataset` directory.

### Usage

```bash
cd dataset
./run_dataset.sh [config_path]
```

- `config_path`: Path to the YAML configuration file (default: `configs/run.yaml`).

Example:
```bash
cd dataset
./run_dataset.sh configs/run_abilene.yaml
```

This will generate output in `dataset/out/`.

## Phase I: Placement

The `phase-I` directory contains algorithms for switch placement (Set Cover ILP and Greedy).

### Usage

```bash
cd phase-I
./run.sh [mode] <input_paths_file> [token]
```

- `mode`: `cover` (ILP Set Cover) or `greedy` (Greedy Set Cover). Default is `cover`.
- `input_paths_file`: Path to the generated paths JSONL file (e.g., `../dataset/out/abilene/paths.jsonl.gz`).
- `token`: Optional identifier for the run.

### Examples

Run Greedy algorithm:
```bash
cd phase-I
./run.sh greedy ../dataset/out/abilene/paths.jsonl.gz abilene_greedy
```

Run ILP Set Cover:
```bash
cd phase-I
./run.sh cover ../dataset/out/abilene/paths.jsonl.gz abilene_cover
```

## Directory Structure

- `dataset/`: Code for generating synthetic datasets.
- `phase-I/`: Phase I algorithms (ILP and Greedy).
- `requirements.txt`: Python dependencies.
