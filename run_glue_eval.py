import json
from eval.tasks.glue_dataset import GLUETask, GLUE_TASKS
from eval.probes.linear_probe import train_probe, eval_probe

def run_glue(model_wrapper, config):
    results = {}

    for task_name in GLUE_TASKS:
        print(f"\n========== {task_name.upper()} ==========")

        task = GLUETask(task_name)
        x_train, y_train, x_test, y_test = task.get()

        feats_train = model_wrapper.extract_features(x_train, config)
        feats_test = model_wrapper.extract_features(x_test, config)

        is_reg = task_name == "stsb"
        probe = train_probe(feats_train, y_train, is_reg, config.max_iter)
        metrics = eval_probe(probe, feats_test, y_test, is_reg)

        results[task_name] = metrics
        print(metrics)

    return results