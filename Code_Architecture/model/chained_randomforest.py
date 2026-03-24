# PURPOSE:
#   CA1 Design Choice 1 — Chained Multi-Output Classification.
#   The core new file added for CA1 to extend the architecture from
#   single-label (Type 2 only) to multi-label (Type 2 + Type 3 + Type 4).
#
# WHAT DOES IT DO?
#   Manages THREE RandomForest classifiers inside ONE class instance.
#   Each classifier is trained on a different chained label level:
#
#   Level 1 — model_y2:
#       Predicts Type 2 only.
#       Real classes: 'Problem/Fault', 'Suggestion'
#       (after NaN filtering, 'Others' class is excluded)
#
#   Level 2 — model_y2_3:
#       Predicts Type 2 + "_" + Type 3 as a single combined string.
#       Real examples from Purchasing group:
#           'Suggestion_Payment'
#           'Problem/Fault_Payment issue'
#           'Suggestion_Invoice'
#       Real examples from AppGallery group:
#           'Problem/Fault_Coupon/Gifts/Points Issues'
#           'Problem/Fault_AppGallery-Install/Upgrade'
#           'Suggestion_VIP / Offers / Promotions'
#
#   Level 3 — model_y2_3_4:
#       Predicts Type 2 + "_" + Type 3 + "_" + Type 4 as a single string.
#       Real examples from Purchasing group:
#           'Suggestion_Payment_Subscription cancellation'
#           'Problem/Fault_Payment issue_Risk Control'
#           'Suggestion_Invoice_Invoice related request'
#       Real examples from AppGallery group:
#           "Problem/Fault_Coupon/Gifts/Points Issues_Can't use or acquire"
#           'Suggestion_VIP / Offers / Promotions_Offers / Vouchers / Promotions'
#
# WHY ACCURACY DROPS AT EACH LEVEL (from CA1 brief):
#   If a model predicts Type 2 = 'Others' but the true answer is
#   Type 2 = 'Suggestion', Type 3 = 'Payment', Type 4 = 'Subscription cancellation':
#       Level 1 result: WRONG  (0/1 correct)
#       Level 2 result: WRONG  (combined label mismatch, even if Type 3 matched)
#       Level 3 result: WRONG  (combined label mismatch)
#   → Score: 0% (model D example from the CA brief)
#
#   Therefore:
#       Level 2 accuracy CANNOT exceed Level 1 accuracy
#       Level 3 accuracy CANNOT exceed Level 2 accuracy
#   This is EXPECTED and CORRECT behaviour.
#
# ARCHITECTURAL PRINCIPLES:
#   Abstraction (Feature 3) — inherits from BaseModel (base.py).
#   All four abstract methods implemented: train(), predict(),
#   print_results(), data_transform().
#   The __init__ signature matches ALL other lecturer models.
#
#   Encapsulation (Feature 2) — receives ONE Data object with all arrays.
#   Accesses data.X_train, data.y2_3_train, data.y2_3_4_test etc. directly.
#
# TEAMMATE: Teammate 2
# =============================================================================

import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import random

# Fix random seeds — ensures all three classifiers produce reproducible results
seed = 0
np.random.seed(seed)
random.seed(seed)


class ChainedRandomForest(BaseModel):
    """
    CA1 Design Choice 1: Chained Multi-Output Classification.

    Three RandomForest classifiers — one per chain level:
        model_y2   — Level 1: Type 2 only
        model_y2_3  — Level 2: Type2 + Type3 combined
        model_y2_3_4 — Level 3: Type2 + Type3 + Type4 combined

    Inherits from BaseModel. Implements all four abstract methods.
    __init__ signature: (model_name: str, embeddings: ndarray, y: ndarray)
    """

    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        # Call the parent BaseModel constructor
        super(ChainedRandomForest, self).__init__()

        # Store model metadata (matching the lecturer's model pattern)
        self.model_name = model_name
        self.embeddings = embeddings   # full TF-IDF matrix before split
        self.y = y                     # all Type 2 labels before split

        # ── Level 1 Classifier ─────────────────────────────────────────────────
        # Predicts Type 2 only: 'Problem/Fault' or 'Suggestion'
        # This is the simplest prediction — only 2 classes after NaN filtering.
        # Accuracy here is the ceiling for Levels 2 and 3.
        # n_estimators=1000: 1000 trees for stable, robust predictions
        # class_weight='balanced_subsample': adjusts for class imbalance
        #   e.g. Purchasing has 66 Suggestion vs 10 Problem/Fault — very imbalanced
        self.model_y2   = RandomForestClassifier(
            n_estimators=1000, random_state=seed, class_weight='balanced_subsample'
        )

        # ── Level 2 Classifier ─────────────────────────────────────────────────
        # Predicts Type2_Type3 combined strings.
        # More classes = harder problem = lower accuracy than Level 1.
        # Purchasing group: 4 unique combos (e.g. 'Suggestion_Payment')
        # AppGallery group: 9 unique combos (e.g. 'Problem/Fault_AppGallery-Install/Upgrade')
        self.model_y2_3  = RandomForestClassifier(
            n_estimators=1000, random_state=seed, class_weight='balanced_subsample'
        )

        # ── Level 3 Classifier ─────────────────────────────────────────────────
        # Predicts full Type2_Type3_Type4 combined strings.
        # Most classes = hardest problem = lowest accuracy.
        # Purchasing group: 6 unique combos
        #   e.g. 'Suggestion_Payment_Subscription cancellation' (most common)
        # AppGallery group: 16 unique combos
        #   e.g. "Problem/Fault_Coupon/Gifts/Points Issues_Can't use or acquire"
        self.model_y2_3_4 = RandomForestClassifier(
            n_estimators=1000, random_state=seed, class_weight='balanced_subsample'
        )

        # ── Prediction storage ─────────────────────────────────────────────────
        # Stored after predict() so print_results() can access them.
        # As per base.py comment: "Predictions should be stored on self.predictions"
        self.predictions      = None   # Level 1 predicted Type 2 labels
        self.predictions_y2_3  = None   # Level 2 predicted Type2_Type3 labels
        self.predictions_y2_3_4 = None   # Level 3 predicted Type2_Type3_Type4 labels

        # Apply data transformations (none needed — Data class handles label building)
        self.data_transform()

    # ==========================================================================
    def train(self, data) -> None:
        """
        Train all three classifiers using the chained labels in the Data object.

        All three classifiers use the SAME X_train (TF-IDF features).
        Only the target label (y) differs between levels.

        data.X_train     — same TF-IDF features for all three classifiers
        data.y_train     — Type 2 labels (e.g. 'Suggestion', 'Problem/Fault')
        data.y2_3_train   — Type2_Type3 labels (e.g. 'Suggestion_Payment')
        data.y2_3_4_train  — Type2_Type3_Type4 labels (e.g. 'Suggestion_Payment_Subscription cancellation')
        """
        print("\n [ChainedRF] Training Level 1 — Type 2 only")
        # Level 1: learn to distinguish 'Problem/Fault' from 'Suggestion'
        self.model_y2.fit(data.X_train, data.y_train)

        print("  [ChainedRF] Training Level 2 — Type 2 + Type 3")
        # Level 2: learn to predict combined label e.g. 'Suggestion_Payment'
        self.model_y2_3.fit(data.X_train, data.y2_3_train)

        print("  [ChainedRF] Training Level 3 — Type 2 + Type 3 + Type 4")
        # Level 3: learn to predict full combined label
        # e.g. 'Suggestion_Payment_Subscription cancellation'
        self.model_y2_3_4.fit(data.X_train, data.y2_3_4_train)

        print("  [ChainedRF] All three classifiers trained successfully.")

    # Run predictions at all three chain levels.
    def predict(self, data) -> None:

        # Level 1: predict Type 2 for each test email
        # e.g. 'Suggestion' or 'Problem/Fault'
        self.predictions      = self.model_y2.predict(data.X_test)

        # Level 2: predict Type2_Type3 for each test email
        # e.g. 'Suggestion_Payment' or 'Problem/Fault_Payment issue'
        self.predictions_y2_3  = self.model_y2_3.predict(data.X_test)

        # Level 3: predict Type2_Type3_Type4 for each test email
        # e.g. 'Suggestion_Payment_Subscription cancellation'
        self.predictions_y2_3_4 = self.model_y2_3_4.predict(data.X_test)

    # Print accuracy at each chain level and detailed classification reports.
    def print_results(self, data) -> None:
        """
        Print accuracy at each chain level and detailed classification reports.

        Compares stored predictions against true labels in the Data object:
            data.y_test    — true Type 2 labels
            data.y2_3_test  — true Type2_Type3 labels
            data.y2_3_4_test — true Type2_Type3_Type4 labels
        """
        # ── Compute accuracy at each level ─────────────────────────────────────
        # accuracy_score = (number of exactly correct predictions) / (total predictions)
        # For chained labels, the ENTIRE combined string must match exactly.
        # 'Suggestion_Payment_Subscription cancellation' ≠ 'Suggestion_Payment_Risk Control'
        acc_y2   = accuracy_score(data.y_test,    self.predictions)
        acc_y2_3  = accuracy_score(data.y2_3_test,  self.predictions_y2_3)
        acc_y2_3_4 = accuracy_score(data.y2_3_4_test, self.predictions_y2_3_4)

        # ── Print accuracy summary ─────────────────────────────────────────────
        print("\n" + "-" * 65)
        print("  CHAINED MULTI-OUTPUT RESULTS")
        print("-" * 65)
        print(f"  Level 1  Type 2 only               : {acc_y2:.4f}  ({acc_y2*100:.2f}%)")
        print(f"  Level 2  Type 2 + Type 3            : {acc_y2_3:.4f}  ({acc_y2_3*100:.2f}%)")
        print(f"  Level 3  Type 2 + Type 3 + Type 4   : {acc_y2_3_4:.4f}  ({acc_y2_3_4*100:.2f}%)")
    

        # ── Detailed classification report for each level ──────────────────────
        # Shows precision, recall, F1-score, and support for each class.
        # zero_division=0 suppresses divide-by-zero warnings when a class
        # has no predictions (can happen with rare chained label combos).

        print("\n" + "-" * 65)
        print("\n--- Level 1: Type 2 only ---")
        print("Classes: Problem/Fault, Suggestion\n")
        print(classification_report(
            data.y_test, self.predictions, zero_division=0
        ))

        print("\n" + "-" * 65)
        print("\n--- Level 2: Type 2 + Type 3 ---")
        print("  e.g. 'Suggestion_Payment', 'Problem/Fault_AppGallery-Install/Upgrade'")
        print(classification_report(
            data.y2_3_test, self.predictions_y2_3, zero_division=0
        ))

        print("\n" + "-" * 65)
        print("\n--- Level 3: Type 2 + Type 3 + Type 4 ---")
        print("  e.g. 'Suggestion_Payment_Subscription cancellation'")
        print(classification_report(
            data.y2_3_4_test, self.predictions_y2_3_4, zero_division=0
        ))

    
    def data_transform(self) -> None:
        pass
