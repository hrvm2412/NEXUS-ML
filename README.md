# Setup Guide (Windows 11)

## Prerequisites

*   **Python 3.12**: Ensure Python 3.12 is installed and added to your PATH.
*   **NVIDIA GPU**: This project requires an NVIDIA GPU with CUDA support.

### NVIDIA CUDA Installation

Please download and install the NVIDIA CUDA Toolkit. Based on the project dependencies, **CUDA 13** is required.

*   [Download NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

## Installation Commands

Open your terminal (PowerShell or Command Prompt) and run the following commands in order:

```powershell
# 1. Create a virtual environment
python -m venv .env

# 2. Activate the virtual environment
.env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Updated Usage

