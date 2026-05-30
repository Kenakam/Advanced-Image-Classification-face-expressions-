# Advanced Image Classification Pipeline using CNN + RNN + ANN Architecture with Optuna Optimization

## 📌 Project Overview

This repository hosts a production-grade, deep learning-driven image classification solution implemented inside a comprehensive Jupyter Notebook (`Advanced Image Classification (face expressions).ipynb`). The project addresses a highly intricate **7-class categorical image classification challenge**.

Unlike conventional visual models that rely exclusively on Convolutional architectures, this pipeline implements a state-of-the-art hybrid sequential design combining **Spatial Features (CNN)**, **Temporal Sequence Processing (RNN)**, and **Dense Fully-Connected Classification Topologies (ANN)**.

To maximize predictive power, the pipeline fully integrates an automated hyperparameter tuning suite driven by the **Optuna** framework. This setup dynamically scales, tests, and finds the most mathematically optimal combination of multi-layered architectural configurations, regularizations, dropout percentages, and adaptive backpropagation gradient algorithms.

---

## 🛠️ Unified System Architecture

The neural network implements a sequential, cross-paradigm architecture specifically engineered to translate spatial high-dimensional array tensors into recurrent sequences, and subsequently into discrete categorical probability distributions:

```
[ Input Tensor: 64x64x3 ] 
          │
          ▼
┌────────────────────────────────────────┐
│  1. Convolutional Neural Network (CNN) │  --> Extracts multi-scale visual features
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│  2. Sequence Reshaping Transition      │  --> Maps spatial maps into sequential timesteps
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│  3. Recurrent Neural Network (RNN)     │  --> Models deep sequential/temporal logic
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│  4. Artificial Neural Network (ANN)    │  --> Learns highly non-linear decision boundaries
└────────────────────────────────────────┘
          │
          ▼
[ Softmax Classification: 7 Classes ]

```

### 1. Convolutional Neural Network (Spatial Feature Extraction)

* **Dual Conv2D Hidden Layers**: Learns local and global image patterns through parameterized dynamic filters.
* **MaxPool2D Layers**: Reduces spatial footprint while maintaining feature dominance and spatial invariance.
* **SpatialDropout2D**: A crucial architectural regularization choice that drops entire 2D feature maps instead of individual pixels to mitigate high spatial correlation overfitting.
* **Flatten Layer**: Flattens remaining multi-channel feature maps into highly dense 1D vector streams.

### 2. Temporal Mapping Transition

* **Reshape Layer**: Transforms the static 1D flattened feature vector into a multi-dimensional sequential matrix structure configured as `(1, -1)`. This forces the pipeline to treat spatial representations as single-timestep data sequences, setting up compatibility with recurrent network layers.

### 3. Recurrent Neural Network (Sequential Logic Processing)

* **LSTM / GRU Blocks**: Dynamically selects between Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) networks to parse sequence streams.
* **Multi-Layer Stacking**: Configurable to stack up to 3 layers deep, with downstream layers receiving structural sequence outputs via `return_sequences=True`.
* **Standard Dropout**: Embedded behind every recurrent block to prevent co-adaptation of sequential parameters.

### 4. Artificial Neural Network & Softmax Decoupling

* **Dense Rectified Linear Unit (ReLU) Layer**: Aggregates processed spatial-recurrent arrays into a non-linear dense layout.
* **Softmax Output Discriminator**: Converts incoming inputs into an explicit probability distribution spread across 7 categorical labels (`class_mode='categorical'`).

---

## 🚀 Hyperparameter Search Space (Optuna Optimization)

The framework optimizes system weights using a automated Tree-structured Parzen Estimator (`TPESampler`) search space:

| Hyperparameter Target | Optimization Search Space Range |
| --- | --- |
| **Conv2D Layer 1 Filter Volume** | `[16, 64]` (Integer Steps) |
| **Conv2D Layer 2 Filter Volume** | `[16, 64]` (Integer Steps) |
| **Kernel Feature Window Size** | `[2, 5]` (Symmetric Window Filters) |
| **MaxPooling Subsampling Window** | `[2, 3]` (Window Dimensions) |
| **Strides Subsampling Progression** | `[1, 2]` (Pixel Multipliers) |
| **Spatial Dropout Ratios (Layers 1-2)** | `[0.1, 0.5]` (Continuous Floating Points) |
| **Recurrent Engine Selection Type** | `['LSTM', 'GRU']` (Categorical Structs) |
| **Stacked Recurrent Depth Limit** | `[1, 3]` (Layer Bounds) |
| **Hidden Recurrent Unit Bounds** | `[32, 128]` (Per-layer unit ranges) |
| **Dense ANN Node Layout Structure** | `[32, 128]` (Fully Connected Layout) |
| **Backpropagation Gradient Engines** | `['adam', 'sgd', 'adagrad', 'rmsprop']` (Categorical Optimizers) |
| **Stochastic Adaptive Learning Rate** | `[1e-5, 1e-3]` (Logarithmic Continuous Range) |

---

## 📈 Computational Constraints & Threshold Considerations

### ⚠️ Infrastructure Resource Boundaries

> [!IMPORTANT]
> This deep learning project was constructed, executed, and validated within local hardware configurations and the shared execution limits of **Google Colab**. Due to limited access to enterprise-grade compute structures (such as distributed high-end multi-GPU cluster networks), training configurations were necessarily bounded.
> As a result, model performance across certain complex trials may not reach maximum theoretical capabilities. In an unconstrained hardware environment with dedicated compute resources, expanding the optimization trial budget and relaxing dataset bottlenecks would lead to superior global convergence and enhanced testing accuracy metrics.

### ⚙️ Task-Specific Threshold Configurations

During optimization trials, certain specific parameter constraints were configured:

* **Epoch Limit**: Fixed at `10` iterations per trial search phase.
* **Early Stopping Patience**: Hardcoded to `5` iterations tracking `val_loss`.
* **Gradient Clipping**: Strict hard limit of `clipvalue=1.0` applied across all chosen optimizer engines.

While standard best practices for training highly complex hybrid neural networks often recommend extended training periods (e.g., hundreds of epochs per model architecture configuration), the thresholds used here are **task-specific boundaries**. They were purposefully implemented to allow the automated Optuna framework to evaluate 20 distinct multi-layer neural architectures within standard notebook timeouts, while successfully avoiding vanishing or exploding gradient behaviors during backpropagation.

---

## 💻 Technical Stack & Ecosystem

* **Language Platform**: Python 3.x
* **Deep Learning Framework**: TensorFlow 2.x / Keras (`Sequential`, `Conv2D`, `LSTM`, `GRU`, `Dense`)
* **Data Transformation Engine**: Keras `ImageDataGenerator` (Real-time image scaling, rotation, shearing, zooming, and automated color normalization)
* **Optimization Framework**: Optuna Suite (`TPESampler` Study Management)
* **Statistical Performance Analysis**: Scikit-Learn (`roc_auc_score`, `mean_squared_error`, `r2_score`)
* **Data Pipelines**: NumPy, Pandas
* **Data Visualization**: Matplotlib, IPython Display HTML formatting

---

## 📦 Pipeline Execution Blueprint

### 1. Repository Setup & Dependencies

Ensure your environment satisfies the library requirements. Install the mandatory frameworks using your terminal or a code cell:

```bash
pip install tensorflow numpy pandas scikit-learn optuna matplotlib

```

### 2. Dataset Hierarchy Layout

For proper asset loading by the `flow_from_directory` input pipelines, extract and organize your data assets exactly as follows:

```
📂 project_root/
 ├── 📄 Final_task_CNN+RNN+ANN_questions.ipynb
 ├── 📦 train.zip
 ├── 📦 test.zip
 ├── 📦 single_prediction.zip
 └── 📂 content/
      └── dataset/
           ├── 📂 train/
           │    ├── 📂 class_1/
           │    └── 📂 class_7/
           └── 📂 test/
                ├── 📂 class_1/
                └── 📂 class_7/

```

### 3. Execution Pipeline Sequence

1. **Initialize Workspace Environment**: Open the Jupyter Notebook file inside Google Colab or your local notebook engine and execute the initial setup cells to pull in libraries and unzip data assets.
2. **Execute Ingestion Data-Streams**: Run the `ImageDataGenerator` code block. This extracts the image assets, reshapes them to target dimensions of `64x64` pixels, scales color spaces down to `1./255`, and applies data augmentation parameters (including horizontal flips, zoom modifications, and shear mappings).
3. **Trigger Automated Search Grid**: Run the objective optimization trial loop. Optuna will orchestrate 20 custom structural trials, outputting the highest performing parameters (`study.best_params`) alongside the top validation accuracy metrics.
4. **Compile the Optimized Model**: Instantiate and train the optimized model architecture using the discovered hyperparameter configurations to achieve reliable evaluation performance.
