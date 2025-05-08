import os
import re
import subprocess

import optuna

# 确保 logs 目录存在
os.makedirs("logs", exist_ok=True)


def objective(trial):
    dataset = "squirrel"  # 可以替换为变量
    source = "critical"
    gpu = "6"
    trial_id = trial.number  # Optuna 会自动给每个 trial 分配唯一 ID

    # 采样参数
    hid = trial.suggest_categorical("hid", [128, 256, 512])
    nlayers = trial.suggest_categorical("nlayers", [3, 4])
    lr1 = trial.suggest_categorical("lr1", [0.1, 0.15, 0.01, 0.05])
    lr2 = trial.suggest_categorical("lr2", [0.5, 0.005])
    wd1 = trial.suggest_categorical("wd1", [0, 5e-4, 5e-5])
    wd2 = trial.suggest_categorical("wd2", [0, 5e-4, 1e-4])
    dpC = trial.suggest_categorical("dpC", [0, 0.3, 0.6, 0.9])
    dpM = trial.suggest_categorical("dpM", [0, 0.4, 0.6, 0.8])
    tau = trial.suggest_categorical("tau", [1, 0.3, 0.6, 0.9])
    dropout = trial.suggest_categorical("dropout", [0.2, 0.4, 0.6, 0.8])

    # 构建命令
    cmd = [
        "python3",
        "training.py",
        "--dataset",
        dataset,
        "--source",
        source,
        "--epochs",
        "1000",
        "--seed",
        "51290",
        "--hid",
        str(hid),
        "--nlayers",
        str(nlayers),
        "--K",
        "10",
        "--patience",
        "200",
        "--lr1",
        str(lr1),
        "--lr2",
        str(lr2),
        "--wd1",
        str(wd1),
        "--wd2",
        str(wd2),
        "--dpC",
        str(dpC),
        "--dpM",
        str(dpM),
        "--tau",
        str(tau),
        "--dropout",
        str(dropout),
        "--dev",
        gpu,
    ]

    # 执行命令并捕获输出
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # 保存日志到 logs/{dataset}_{trial_id}.log
    log_path = f"logs/{dataset}_{trial_id}.log"
    with open(log_path, "w") as f:
        f.write("CMD:\n" + " ".join(cmd) + "\n\n")
        f.write("STDOUT:\n" + result.stdout + "\n\n")
        f.write("STDERR:\n" + result.stderr)

    # 提取输出中的 ACCURACY
    match = re.search(r"optuna_avg_f1_score:\s*([0-9.]+)", result.stdout)
    if match:
        acc = float(match.group(1))
        return -acc  # Optuna 默认最小化，这里负数表示最大化 accuracy
    else:
        # 如果没有正确输出 accuracy，记录失败
        raise ValueError(f"Trial {trial_id} failed to return accuracy.")


# 启动搜索
study = optuna.create_study()
study.optimize(objective, n_trials=50, n_jobs=4)

print("Best trial:")
print(" Value (accuracy):", -study.best_value)
print(" Params:", study.best_params)
