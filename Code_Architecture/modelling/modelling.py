# Modelling co ordinator function
# Calls model instances and runs each one through the abstract
# interface defined by BaseModel
# 
# 1. RandomForest — baseline, Type 2 only
# 2. ChainedRandomForest — Design 1, Type 2 + 3 + 4
#
from model.randomforest import RandomForest
from model.chained_randomforest import ChainedRandomForest

# Run all models with given Data object
def model_predict(data, name):

    # skip if Data object has no training data
    if data.X_train is None:
        print(f"Skipping group '{name}' : insufficient data")
        return

    # 1. RandomForest
    # Single-label classifier, predicts Type 2 only
    print(f"\n{'#'*65}")
    print(f"    GROUP : {name}")
    print(f"{'#'*65}")

    model = RandomForest("RandomForest")
    model.train(data)
    model.predict(data)
    model.print_results(data)

    # 2. ChainedRandomForest
    # Multi-label classifier
    chained_model = ChainedRandomForest("ChainedRandomForest")
    chained_model.train(data)
    chained_model.predict(data)
    chained_model.print_results(data)

# Re-evaluate and print results for a trained model
# *Print again without retraining* 
def model_evaluate(model, data):
    model.print_results(data)