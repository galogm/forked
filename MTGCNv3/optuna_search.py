import os
import sys

sys.path.append("../../../..")
import time

import numpy as np
import optuna
import torch

data_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(data_root)

from args import args
from data.data_process import load_data
from the_utils import save_to_csv_files, set_device, set_seed, split_train_test_nodes
from train import train
from train_prior import train_prior


def get_splits_mask(
    num_nodes,
    name,
    train_ratio,
    valid_ratio,
    repeat,
    split_id,
    SPLIT_DIR,
    mask=True,
):
    train_idx, val_idx, test_idx = split_train_test_nodes(
        num_nodes=num_nodes,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        data_name=name,
        split_id=split_id,
        split_times=repeat,
        fixed_split=True,
        split_save_dir=SPLIT_DIR,
    )
    if not mask:
        return train_idx, val_idx, test_idx
    train_mask = (
        torch.zeros(num_nodes)
        .scatter_(0, torch.tensor(train_idx, dtype=torch.int64), 1)
        .bool()
    )
    val_mask = (
        torch.zeros(num_nodes)
        .scatter_(0, torch.tensor(val_idx, dtype=torch.int64), 1)
        .bool()
    )
    test_mask = (
        torch.zeros(num_nodes)
        .scatter_(0, torch.tensor(test_idx, dtype=torch.int64), 1)
        .bool()
    )

    return train_mask, val_mask, test_mask


# def seed_torch(seed=1029):
#     os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     print('seed init successful')


def update_pseudo(label, predict, easy_idx):
    label[easy_idx] = predict[easy_idx]
    return label


def mutil_stage_train(args, trial):
    print(args)
    # seed_torch()
    # set_seed(42)
    # data = load_data(args.dataset, args.new_split, args.train_spilt, args.device)
    # if len(data.train_mask.shape) > 1:
    #     train_mask, val_mask, test_mask = data.train_mask[:,2], data.val_mask[:,2], data.test_mask[:,2]
    # else:
    #     train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    # label = data.y.clone()

    import torch_geometric.transforms as T
    from graph_datasets import load_data
    from the_utils import save_to_csv_files, set_device, set_seed

    from ignn.modules import DataConf
    from ignn.utils import read_configs

    DATA = DataConf(**read_configs("data"))
    DEVICE = set_device(args.device.split(':')[1])
    set_seed(42)
    # graph_dgl, labels, class_num = load_data(
    #     dataset_name=args.dataset,
    #     directory=DATA.DATA_DIR,
    #     source=args.source,
    #     row_normalize=True,
    #     rm_self_loop=False,
    #     add_self_loop=True,
    #     to_simple=True,
    #     verbosity=3,
    # )
    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor(), T.ToDevice(DEVICE)])
    data = transform(load_data(
        dataset_name=args.dataset,
        directory=DATA.DATA_DIR,
        source=args.source,
        row_normalize=True,
        rm_self_loop=False,
        add_self_loop=True,
        to_simple=True,
        verbosity=3,
        return_type="pyg",
    ))
    label = data.y.clone().to(DEVICE)

    num_k_dict = {
        "wisconsin": [5, 0, 0, 0],
        "cora": [100, 50, 20, 1],
        "citeseer": [100, 50, 20, 1],
        "pubmed": [1000, 500, 300, 1],  # 400,50,20,1
        "cs": [50, 10, 5, 1],
        "physics": [1500, 1500, 700, 10],
        "texas": [5, 2, 2, 1],
        "cornell": [5, 2, 2, 1],
        "squirrel": [100, 50, 20, 1]
    }
    num_k_list = num_k_dict[args.dataset]

    run = 10

    res_m = []
    training_time = []

    for idx in range(run):
        args.id = f"{trial.number}_{idx}"

        if data.name == "arxiv_ogb":
            train_mask, val_mask, test_mask = (
                data["train_mask"],
                data["val_mask"],
                data["test_mask"],
            )
        else:
            train_mask, val_mask, test_mask = get_splits_mask(
                data.num_nodes,
                data.name,
                train_ratio=48,
                valid_ratio=32,
                repeat=run,
                split_id=idx,
                SPLIT_DIR=DATA.SPLIT_DIR,
                mask=True,
            )
        train_mask, val_mask, test_mask = train_mask.to(DEVICE), val_mask.to(DEVICE), test_mask.to(DEVICE)

        t_s = time.time()
        res, p_res = [], []
        pseudo_mask = torch.zeros_like(train_mask, dtype=bool, device=args.device)

        for s in range(4):
            print(f"------------------stage {s} begin -------------------")
            num_k = num_k_list[s]
            train_num = torch.sum(pseudo_mask).item() + 0.0001
            label_acc = label[pseudo_mask].eq(data.y[pseudo_mask]).sum() / train_num
            print(f"train node num:{train_num}, pseudo node acc{label_acc.item()}")

            p_max_acc = train_prior(
                data,
                label,
                train_mask,
                val_mask,
                test_mask,
                pseudo_mask,
                num_k,
                args=args,
            )

            if s == 0:
                pseudo_mask, easy_idx, predict, max_acc = train(
                    data,
                    label,
                    train_mask,
                    val_mask,
                    test_mask,
                    pseudo_mask,
                    num_k,
                    stage=s,
                    args=args,
                )
            else:
                pseudo_mask, easy_idx, predict, max_acc = train(
                    data,
                    label,
                    train_mask,
                    val_mask,
                    test_mask,
                    pseudo_mask,
                    num_k,
                    s,
                    predict,
                    args=args,
                )

            label = update_pseudo(label, predict, easy_idx)
            print(f"------------------stage {s} end -------------------")
            res.append(max_acc)
            p_res.append(p_max_acc)
        res_m.append(max(res))
        training_time.append(time.time() - t_s)
        print('time', training_time[-1])

    print("avg_train_time: {:.4f} s".format(np.mean(training_time)))
    print("std_train_time: {:.4f} s".format(np.std(training_time)))
    print("optuna_avg_f1_score: {:.4f}".format(np.mean(res_m)))
    print("optuna_std_f1_score: {:.4f}".format(np.std(res_m)))

    save_to_csv_files(
        results={
            "acc": f"{np.mean(res_m)}±{np.std(res_m)}",
            "time": f"{np.mean(training_time)}±{np.std(training_time)}",
        },
        insert_info={
            "dataset": data.name,
            "model": "MTGCN",
        },
        append_info={
            "args": args.__dict__,
            "source": args.source,
        },
        csv_name="baselines_ex.csv",
    )

    return res_m.mean()


def set_search_space(trial):
    args.lr = trial.suggest_float("lr", 0.001, 0.05, log=True)
    args.tblr = trial.suggest_float("tblr", 0.001, 0.05, log=True)
    args.tbwd = trial.suggest_float("tbwd", 1e-6, 1e-4)
    args.tpwd = trial.suggest_float("tpwd", 1e-6, 1e-4)
    args.lw = trial.suggest_float("lw", 0.01, 0.7)
    args.a = trial.suggest_float("a", -0.9, 0.9)
    args.layer_num = trial.suggest_categorical("layer_num", [8, 16])
    # args.layer_num = trial.suggest_categorical("layer_num", [64])

    args.dr = trial.suggest_float("dropout", 0.1, 0.8)
    res = mutil_stage_train(args, trial)
    return res


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(set_search_space, n_trials=20)
    for item in study.trials:
        with open("result_" + str(args.dataset) + ".txt", "a") as f:
            f.write(f"{item}\n")

    print("Best trial:")
    print(" Value (accuracy):", study.best_value)
    print(" Params:", study.best_params)
