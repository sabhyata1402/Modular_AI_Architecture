# Multi-Label Chained AI Classification Architecture

## 📖 Overview
This repository contains the implementation for **CA1: Engineering and Evaluating Artificial Intelligence**. The goal of this project was to refactor an existing single-label text classification architecture into a **Multi-label Chained Classifier** using Object-Oriented and Modular AI principles.

It employs **Design Choice 1: Chained Multi-outputs** to cascade customer service email classifications across three deepening levels of granularity (Type 2 -> Type 3 -> Type 4).

## 🏗️ Architecture Design Principles
This project implements strong software engineering principles for machine learning:
1. **Abstraction**: The `BaseModel` handles a uniform interface (`train()`, `predict()`, `print_results()`) so the coordinator script doesn't need to know the inner workings of `ChainedRandomForest`.
2. **Encapsulation**: The `Data` class hides complex test/train split logic and the concatenation of chained labels (e.g., `Type 2 + Type 3`).
3. **Modularity**: Data preprocessing, embedding (TF-IDF), model definitions, and orchestration are entirely decoupled into separate specific files.

## 🗂️ Project Structure
```text
skeleton/
├── main.py                    # The central controller and entry point
├── Config.py                  # Global constant mapping
├── preprocess.py              # Data deduplication, cleaning, and text translation via Google Translate
├── embeddings.py              # Translates text into mathematical TF-IDF Vectors
├── utils.py                   # Reusable helper functions (like execution timers)
├── data/                      # Raw datasets (e.g., AppGallery.csv, Purchasing.csv)
├── model/
│   ├── base.py                # Abstract Base Class for ML models
│   ├── randomforest.py        # Baseline ML model
│   └── chained_randomforest.py# Custom Chained Multi-output ML model
└── modelling/
    ├── data_model.py          # Data Encapsulation Object
    └── modelling.py           # Coordinator script calling the models uniformly
```

## 🚀 How to Run

1. **Prerequisites:** You need Python 3 installed. We recommend using a `conda` environment.
2. **Install Dependencies:**
   Ensure you have `pandas`, `scikit-learn`, `numpy`, and `googletrans==4.0.0-rc1` installed.
   ```bash
   pip install pandas scikit-learn numpy googletrans==4.0.0-rc1
   ```
3. **Execution:**
   Navigate your terminal into the root folder (`skeleton/`) and execute the controller:
   ```bash
   python main.py
   ```

## 📊 Expected Output
Upon running `main.py`, the system will load the data, execute TF-IDF feature extractions, and launch the models. You will see performance metric reports print to your terminal for:
- **Baseline**: Singe-label `Type 2` only (RandomForest)
- **Level 1**: Chained Type 2 only
- **Level 2**: Chained concatenated `Type 2 + Type 3`
- **Level 3**: Chained concatenated `Type 2 + Type 3 + 4`

*(Note: As the chained granularity deepens and adds more complex unique combinations, accuracy naturally and expectedly drops).*
