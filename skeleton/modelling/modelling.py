from model.randomforest import RandomForest


def model_predict(data, df, name):

    print("BASELINE — RandomForest (Type 2 only)")

    # Initialize the RandomForest model
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)  # Train the model — uses data.X_train and data.y_train internally

    model.predict(data) # Predict
    model.print_results(data) # Evaluate

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL 2: ChainedRandomForest (CA1 Design Choice 1)
    # ══════════════════════════════════════════════════════════════════════════
    # This is the new model added for CA1.
    # It classifies all three chain levels:
    #   Level 1: Type 2 only             (uses data.y_train)
    #   Level 2: Type 2 + Type 3         (uses data.y23_train)
    #   Level 3: Type 2 + Type 3 + Type 4 (uses data.y234_train)
    #
    # NOTICE: The three method calls below are IDENTICAL to RandomForest above.
    # This is Abstraction (Feature 3) working correctly — the coordinator
    # does not change when a new model type is added.

  
    print("  CA1 DESIGN CHOICE 1 — ChainedRandomForest")
    print("  Levels: Type2 | Type2 + Type3 | Type2 + Type3 + Type4")

    chained_model = ChainedRandomForest("ChainedRandomForest", data.get_embeddings(), data.get_type())
    chained_model.train(data)
    chained_model.predict(data)
    chained_model.print_results(data)


def model_evaluate(model, data):
    model.print_results(data)
