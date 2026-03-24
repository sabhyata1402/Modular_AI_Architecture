# PURPOSE:
#   The modelling coordinator — creates model instances and runs each one
#   through the uniform abstract interface provided by BaseModel.
#
# ARCHITECTURAL PRINCIPLE:
#   Abstraction (Feature 3).
#   This file calls EXACTLY the same three methods on EVERY model:
#       model.train(data)
#       model.predict(data)
#       model.print_results(data)
#   The coordinator never knows (or needs to know) the internal details
#   of any model. Adding a new model only requires:
#     1. Adding one import line at the top of this file
#     2. Adding three lines (instantiate, train, predict, print_results)
#   main.py never changes.
#
# MODELS RUN:
#   1. RandomForest (baseline — classifies Type 2 only)
#
#   2. ChainedRandomForest (CA1 Design Choice 1)
#      Classifies Type 2, Type2+Type3, Type2+Type3+Type4.
#      Demonstrates multi-label chained classification.

from model.randomforest import RandomForest
from model.chained_randomforest import ChainedRandomForest


def model_predict(data, df, name):

    if data.X_train is None:
        print(f"  Skipping group '{name}' — insufficient data.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL 1: RandomForest (Baseline — Type 2 only)
    # ══════════════════════════════════════════════════════════════════════════


    print("\n" + "-" * 65)
    print("  BASELINE — RandomForest (Type 2 only)")
    print("-" * 65)

    # Instantiate the model — same __init__ signature as all lecturer models:
    # (model_name: str, embeddings: np.ndarray, y: np.ndarray)
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())

    # Train the model — uses data.X_train and data.y_train internally
    model.train(data)

    # Generate predictions — uses data.X_test internally,
    # stores results on self.predictions
    model.predict(data)

    # Print classification report — uses data.y_test and self.predictions
    model.print_results(data)

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL 2: ChainedRandomForest (CA1 Design Choice 1)
    # ══════════════════════════════════════════════════════════════════════════
    # This is the new model added for CA1.
    # It classifies all three chain levels:
    #   Level 1: Type 2 only             (uses data.y_train)
    #   Level 2: Type 2 + Type 3         (uses data.y2_3_train)
    #   Level 3: Type 2 + Type 3 + Type 4 (uses data.y2_3_4_train)

    print("\n" + "-" * 65)
    print("  CA1 DESIGN CHOICE 1 — ChainedRandomForest")
    print("  Levels: Type2  |  Type2 + Type3  |  Type2 + Type3 + Type4")
    print("-" * 65)
  

    # Instantiate the chained model — same signature as RandomForest
    chained_model = ChainedRandomForest("ChainedRandomForest", data.get_embeddings(), data.get_type())

    # Train all three internal classifiers — uses X_train, y_train, y2_3_train, y2_3_4_train from the Data object
    chained_model.train(data)

    # Generate predictions at all three levels — uses X_test internally, stores results on self.predictions, self.predictions_y2_3, self.predictions_y2_3_4
    chained_model.predict(data)

    # Print accuracy summary + classification reports for all three levels uses y_test, y23_test, y234_test and stored predictions
    chained_model.print_results(data)

# Utility function to re-evaluate and print results for an already-trained model.
def model_evaluate(model, data):
    model.print_results(data)
