import os
import numpy as np
from datasets import load_from_disk

GLUE_ROOT = "/g/data/oy87/cw9909/glue_data"

GLUE_TASKS = [
    "cola", "sst2", "mrpc", "qqp",
    "stsb", "mnli", "qnli", "rte", "wnli"
]

# Mapping from task → (field1, field2) or None for single text
FIELD_MAP = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),

    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),

    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2")
}

class GLUETask:
    def __init__(self, name: str):
        self.name = name
        task_dir = os.path.join(GLUE_ROOT, name)
        self.ds = load_from_disk(task_dir)

    def get(self):
        # MNLI special-case because it has matched & mismatched
        if self.name == "mnli":
            train = self.ds["train"]
            test = self.ds["validation_matched"]
        else:
            train = self.ds["train"]
            test = self.ds["validation"]

        field1, field2 = FIELD_MAP[self.name]

        # ---------- Single-sentence tasks ----------
        if field2 is None:
            X_train = train[field1]
            X_test  = test[field1]
            y_train = np.array(train["label"])
            y_test  = np.array(test["label"])
            return X_train, y_train, X_test, y_test

        # ---------- Regression (STS-B) ----------
        if self.name == "stsb":
            X_train = list(zip(train[field1], train[field2]))
            X_test  = list(zip(test[field1],  test[field2]))
            y_train = np.array(train["label"], dtype=float)
            y_test  = np.array(test["label"],  dtype=float)
            return X_train, y_train, X_test, y_test

        # ---------- All other sentence-pair tasks ----------
        X_train = list(zip(train[field1], train[field2]))
        X_test  = list(zip(test[field1],  test[field2]))
        y_train = np.array(train["label"])
        y_test  = np.array(test["label"])
        return X_train, y_train, X_test, y_test