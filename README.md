# Leaves Life 🌿
Leaf Disease Detection on Images with Neural Networks Project.

## Prerequisites
- Python 3.8 or higher
- Docker

## Setup for simple leaf detection
If you only want to use the web application to detect the disease or healthiness of a leaf.

Create a Virtual environment
```
python -m venv .venv
```
Activate the virtual environment (Windows):
```
.venv\Scripts\activate
```
Activate the virtual environment (macOS/Linux):
```
source .venv/bin/activate
```
Install the required Python packages:
```
pip install -r requirements.txt
```

Run the Application
```
cd src
python app.py
```

The application will be available at ```http://localhost:5000```.


## Setup for training on CPU
Complete the simple setup first, then:

Build the docker image
```
docker compose up --build
```
This will start the MLflow server at ```http://localhost:5001```.

Run the training script
```
cd src
python train_model.py
```

## Setup for training on GPU
Install Anaconda here : <a href="https://www.anaconda.com/download/">anaconda.com/download</a>

Install CUDA 11.2 here : <a href="https://developer.nvidia.com/cuda-11.2.2-download-archive">developer.nvidia.com/cuda-11.2.2-download-archive</a> 

Download CUDNN 8.1.0 here : <a href="https://developer.nvidia.com/rdp/cudnn-archive">developer.nvidia.com/rdp/cudnn-archive</a>

Create a new Python 3.8 enrionment
```
conda create -n leaveslife python=3.8
conda activate leaveslife
```

Install CUDA and CUDNN in the environment
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Install TensorFlow after CUDA and CUDNN
```
python -m pip install "tensorflow==2.10"
```

Install other dependencies
```
pip install -r requirements.txt
```

Build the docker image
```
docker compose up --build
```
This will start the MLflow server at ```http://localhost:5001```.

Run the training script
```
cd src
python train_model.py
```
