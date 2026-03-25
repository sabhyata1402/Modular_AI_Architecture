# Email Classification Project

## Overview

This project implements a multi label email classification system. Built atop the
provided skeleton file structure, the architecture was designed and extended with
adherence to principles of modularity, encapsulation and abstraction. 

It classifies customer emails across three dependent label types — 
Type 2, Type 3, and Type 4 with the Chained Multi-Output Classifier


## How It Works

Messages are loaded from two datasets, AppGallery and Purchasing, they are cleaned, 
and converted into numerical vectors using TF-IDF. Data is then grouped by Type 
1 and passed through two classifiers:

- **RandomForest (the baseline)** classifies Type 2 only
- **ChainedRandomForest (for Design Choice 1)** — classifies all three label sets:
  - Type 2 only (such as `Suggestion`)
  - Type 2 + Type 3 combined (concat results - `Suggestion_Payment`)
  - Type 2 + Type 3 + Type 4 combined (eg. `Suggestion_Payment_Subscription cancellation`)

Accuracy is measured on each level. Since the labels are dependent, accuracy cannot 
increase as the chain deepens. So if Type 2 is predicted incorrectly, the full combined 
label is wrong regardless of Type 3 or Type 4.


## Architecture

The code is structured around the three architectural principles:

**Separation of Concerns** — involves preprocessing, vectorisation, data encapsulation, 
modelling, and orchestration each living in their own dedicated file. The `main.py` file
only coordinates the pipeline, it contains no implementation logic

**Encapsulation** — the `Data` class in `data_model.py` has all train test splits 
and chained label arrays in one object. Every model receives this single object rather 
than raw arrays, keeping the input format consistent across all the models

**Abstraction** — `BaseModel` in `base.py` defines a common interface that every model 
must implement. These are `train(data)`, `predict(data)`, `print_results(data)`, and 
`data_transform()`. The modelling coordinator calls these methods without knowing which 
model it's talking to


## Project Structure
```
Code_Architecture/
   main.py                     # Entry point, coordinats the full pipeline
   Config.py                   # Shared constants (eg. column names, group key)
   preprocess.py               # Data loading, deduplication, cleaning
   embeddings.py               # TF-IDF vectorisation
   utils.py                    # Helper utilities (sucha s time_it decorator)
   data/
      AppGallery.csv
      Purchasing.csv
   model/
      __init__.py             # Exposes public model interface for the package
      base.py                 # Abstract BaseModel class
      randomforest.py         # Baseline single label classifier
      chained_randomforest.py # Chained multi-label classifier
   modelling/
      data_model.py           # Data class — encapsulates all train test splits
      modelling.py            # Runs each model through the uniform interface
```


## Setup and Running

You will need Python 3 and the following packages:
```
pip install pandas scikit-learn numpy
```

To run the pipeline, navigate to the project root (`Code_Architecture`) and execute:
```
python main.py
```


## Expected Output

The output will print timing information for each pipeline step, followed by results 
for each data group (AppGallery & Games + In-App Purchase) For each group you will see:

- A baseline classification report for Type 2 only
- A chained accuracy summary showing all three levels
- Full classification reports per chain level

Accuracy will decrease at each chain level — which is anticipated and correct behaviour, 
and is the key result this architecture is to demonstrate


## Team

Team members include Sabhyata Kumari and Patrick Tsouganov. Both team members contributed 
commits — see repository history for individual contributions