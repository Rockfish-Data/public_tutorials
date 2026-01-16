# Telecom RAN Data Generation Tutorial

Learn how to build synthetic timeseries datasets with realistic patterns using Rockfish's Entity Data Generator.

This tutorial uses a telecom Radio Access Network (RAN) schema to demonstrate key concepts.

## 📖 What You'll Learn

**[Read the complete tutorial](https://docs.rockfish.ai/uc-demos/use-case-product-demo-telecom-ran.html)** to understand:
- Defining entities with cardinalities and time ranges
- Creating entity relationships with foreign keys
- Configuring timeseries measurements with seasonality
- Using categorical and uniform distribution columns
- Creating derived columns with dependencies

The tutorial is also included [in the same directory](rf_telecom_ran_tutorial.md) for offline reference. 

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- Rockfish API Key: Please contact [support@rockfish.ai](mailto:support@rockfish.ai) if you do not have an API Key yet.

### Setup

```bash
# Clone or download the tutorial files, and cd into the directory
cd telecom_ran_schema

# Create a virtual environment
python -m venv rf-venv
source rf-venv/bin/activate

# Install Rockfish SDK and other dependencies
pip install -r requirements.txt

# Set Rockfish API key as an environment variable
# Tip: You can also create an .env file in the same directory
export ROCKFISH_API_KEY="your-api-key-here"

# Run tutorial notebooks using Jupyter
# Tip: Start with 01_basic_generation.ipynb
jupyter notebook
```

## 📁 Tutorial Structure

### Notebooks

- **01_basic_generation.ipynb**: Generate baseline data (~10K rows, ~1-2 min)
- **02_scale_generation.ipynb**: Generate baseline data at scale (~2M rows, ~2-5 min)
- **03_incident_generation.ipynb**: Simulate incidents in data (~3-5 min)

### Supporting Files

- **utils.py**: Schema definitions (`create_telecom_ran_schema()`, `create_incident_telecom_ran_schema()`) and plotting helpers
- **requirements.txt**: Dependencies to run the tutorial
- **rf_telecom_ran_tutorial.md**: Tutorial documentation (offline reference, useful for AI assistance)

## 📚 Learn More

- [Entity Data Generation API Reference](https://docs.rockfish.ai/sdk/actions-ent.html)
