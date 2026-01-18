# Machine Learning Final Project

## Project Setup

This project uses `uv` for dependency management and Jupyter Notebooks for analysis.

### Prerequisites

- `uv` installed.

### Installation

Dependencies are managed in `pyproject.toml`. To sync/install:

```bash
uv sync
```

### Running the Notebook

To launch the Jupyter Notebook with the correct environment:

```bash
uv run jupyter notebook
```

### Environment & Caching

All caches are configured to be stored locally in the `.cache` directory within this folder.
- **Kaggle Cache**: `.cache/kagglehub`
- **Hugging Face**: `.cache/huggingface`
- **UV Cache**: `.cache/uv`
- **Jupyter**: `.cache/jupyter`

### Project Structure

- `Final_Project.ipynb`: Main notebook for the project.
- `.env`: Environment variables for cache configuration.
- `pyproject.toml`: Dependency definitions.
