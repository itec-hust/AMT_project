
# AMTPro-Project

## Project Overview
This project includes deep learning architectures for audio models, video models, and fusion models, designed for research and applications in multimodal tasks. The code structure is modular and well-organized, making it easy to extend and debug.

---

## Code Structure

### 1. **Model Folder**
This folder contains the main network architectures, including:
- **`hppt`**: Audio model
- **`conformer`**: Video model
- **`fusion`**: Fusion model

Each model's subfolder contains the following files:
- **`run.py`**: The main script to run the program, initiating training or testing.
- **`train.py`**: Defines the training process.
- **`test.py`**: Defines the testing process.

> Configuration parameters are set using JSON files specific to each model, loaded in `run.py`.

### 2. **data Folder**
Stores datasets and data loaders for all models.

### 3. **evaluate Folder**
Contains the following modules:
- **Loss functions**
- **Accuracy calculation methods during training**

### 4. **feature_extract Folder**
Handles the first stage of feature extraction for audio and video models.

### 5. **utils Folder**
Contains utility functions for common tasks across the project.






---

## File Description
| Folder/File Name       | Description                            |
|------------------------|----------------------------------------|
| `Model/hppt`          | Audio model network architecture       |
| `Model/conformer`     | Video model network architecture       |
| `Model/fusion`        | Fusion model network architecture      |
| `data`                | Datasets and data loaders              |
| `evaluate`            | Loss functions and accuracy calculation|
| `feature_extract`     | Feature extraction for audio and video |
| `utils`               | Common utility functions               |

