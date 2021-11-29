import optuna
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import create_dataloader
from src.model import Model
from src.utils.torch_utils import model_info, check_runtime, autopad
from src.trainer import TorchTrainer, count_model_params
from typing import Any, Dict, List, Tuple
from subprocess import _args_from_interpreter_flags
import argparse
import yaml
import os
import math

DATA_PATH = "/opt/ml/data"  # type your data path here that contains test, train and val directories
RESULT_MODEL_PATH = "./result_model.pt" # result model will be saved in this path

MAX_NUM_POOLING = 3
MAX_DEPTH = 6
CLASSES = 6
DEFAULT = {
        "EPOCHS": 10,
        "IMG_SIZE": 96,
        "n_select": 2,
        "BATCH_SIZE": 16,
        "optimizer" : None,
        "scheduler" : None,
    }
    
BEST_MODEL_SCORE = 0 # f1 score threshold

def calculate_feat_size(image_size:int, kernel_size:int, stride:int, padding:int =None) -> int :
    if padding is None:
        padding = int(*autopad(kernel_size, padding))
    after_size = math.floor((image_size - kernel_size + 2*padding) / stride) + 1
    return after_size

def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparam from user-specified search space."""
    learning_rate = trial.suggest_categorical("lr", [0.1, 0.5, 0.01, 0.05, 0.001, 0.005])
    epochs = trial.suggest_int("epochs", low=50, high=150, step=50)
    img_size = trial.suggest_categorical("img_size", [96, 112, 168, 224])
    n_select = trial.suggest_int("n_select", low=0, high=6, step=2)
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "adamw"])
    batch_size = trial.suggest_int("batch_size", low=16, high=32, step=16)
    scheduler = trial.suggest_categorical("scheduler", ["reduce", "cosine", "onecycle", "None"])
    return {
        "INIT_LR" : learning_rate,
        "EPOCHS": epochs,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "BATCH_SIZE": batch_size,
        "optimizer" : optimizer,
        "scheduler" : scheduler,
    }

def add_module(trial, depth, n_pooling, image_size):
    m_name = 'm'+str(depth)
    if depth >= 6 and image_size <= 48 :
        module_list = ["Conv", "DWConv", "MBConv", "InvertedResidualv2","InvertedResidualv3", "Fire", "Bottleneck", "BottleneckAttn", "ECAInvertedResidualv2", "ECAInvertedResidualv3", "Pass"] 
    else :
        module_list = ["Conv", "DWConv", "MBConv", "InvertedResidualv2","InvertedResidualv3", "Fire", "Bottleneck", "ECAInvertedResidualv2", "ECAInvertedResidualv3", "Pass"]

    m_args = []

    # default 설정
    m_padding = None
    m_kernel = 1 

    #depth = depth-1
    if n_pooling < MAX_NUM_POOLING:
        m_stride = trial.suggest_int(m_name+"/stride", low=1, high=2, step=1)
    else:
        m_stride = 1
        
    if m_stride==1:
        m_repeat = trial.suggest_int(m_name+"/repeat", low=1, high=8, step=1)
    else:
        m_repeat = 1
        
    # Module
    if depth >= 6 and image_size <= 48:
        module_idx = trial.suggest_int(m_name+"/module_name", low=1, high=11)
    else:
        module_idx = trial.suggest_int(m_name+"/module_name", low=1, high=10)
    m = module_list[module_idx-1]
    if m == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m_out_channel = trial.suggest_int(m_name+"/out_channel", low=16, high=64, step=16)
        m_kernel = trial.suggest_int(m_name+"/kernel_size", low=1, high=5, step=2)
        m_activation = trial.suggest_categorical(
            m_name+"/activation", ["ReLU", "Hardswish"]
        )
        m_args = [m_out_channel, m_kernel, m_stride, None, 1, m_activation]
    elif m == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m_out_channel = trial.suggest_int(m_name+"/out_channel", low=16, high=64, step=16)
        m_kernel = trial.suggest_int(m_name+"/kernel_size", low=1, high=5, step=2)
        m_activation = trial.suggest_categorical(
            m_name+"/activation", ["ReLU", "Hardswish"]
        )
        m_args = [m_out_channel, m_kernel, m_stride, None, m_activation]
    elif m == "MBConv":
        m_kernel = 5
        # m_kernel = trial.suggest_int(m_name+"/kernel_size", low=3, high=5, step=2)
        m_out_channel = trial.suggest_int(m_name+"/out_channel_mb", low=16*depth, high=32*depth, step=16)
        m_exp_ratio = trial.suggest_int(m_name+"/exp_ratio_mb", low=1, high=4)
        m_args = [m_exp_ratio, m_out_channel, m_stride, m_kernel]
    elif m == "InvertedResidualv2":
        m_out_channel = trial.suggest_int(m_name+"/out_channel_v2", low=16*depth, high=32*depth, step=16)
        m_exp_ratio = trial.suggest_int(m_name+"/exp_ratio_v2", low=1, high=4)
        m_args = [m_out_channel, m_exp_ratio, m_stride]
    elif m == "InvertedResidualv3":
        m_kernel = trial.suggest_int(m_name+"/kernel_size", low=3, high=5, step=2)
        m_exp_ratio = round(trial.suggest_float(m_name+"/exp_ratio_v3", low=1.0, high=6.0, step=0.1), 1)
        m_out_channel = trial.suggest_int(m_name+"/out_channel_v3", low=16*depth, high=32*depth, step=16)
        m_se = trial.suggest_int(m_name+"/se_v3", low=0, high=1, step=1)
        m_hs = trial.suggest_int(m_name+"/hs_v3", low=0, high=1, step=1)
        m_args = [m_kernel, m_exp_ratio, m_out_channel, m_se, m_hs, m_stride]
    elif m == "Fire":
        # Fire args: [squeeze_planes, expand1x1_planes, expand3x3_planes]
        m_sqz = trial.suggest_int(m_name+"/sqz", low=16, high=64, step=16)
        m_exp1 = trial.suggest_int(m_name+"/exp1", low=64, high=256, step=64)
        m_stride = 1
        m_args = [m_sqz, m_exp1, m_exp1]
    elif m == 'Bottleneck':
        m_out_channel = trial.suggest_int(m_name+"/out_channel_b", low=16*depth, high=32*depth, step=16)
        m_stride = 1
        m_args = [m_out_channel]
    elif m == "BottleneckAttn":
        m_out_channel = trial.suggest_int(m_name+"/out_channel_bt", low=1024, high=2048, step=1024)
        print('#'*100, '\nBottleneckAttn input size : ', image_size)
        m_feature_size = image_size
        m_num_heads = trial.suggest_int("m/num_heads", low=4, high=8, step=4)
        m_args = [m_out_channel, m_feature_size, m_stride, m_num_heads]
    elif m == "ECAInvertedResidualv2":
        m_out_channel = trial.suggest_int(m_name+"/out_channel_v2", low=16, high=32, step=16)
        m_exp_ratio = trial.suggest_int(m_name+"/exp_ratio_v2", low=1, high=4)
        m_k_eca = trial.suggest_int(m_name+"/v2_k_eca", low=3, high=9, step=2)
        m_args = [m_out_channel, m_exp_ratio, m_stride, m_k_eca]
    elif m == "ECAInvertedResidualv3":
        m_kernel = trial.suggest_int(m_name+"/kernel_size", low=3, high=5, step=2)
        m_exp_ratio = round(trial.suggest_float(m_name+"/exp_ratio_v3", low=1.0, high=6.0, step=0.1), 1)
        m_out_channel = trial.suggest_int(m_name+"/out_channel_v3", low=16, high=40, step=8)
        m_k_eca = trial.suggest_int(m_name+"/v3_k_eca", low=3, high=9, step=2)
        m_hs = trial.suggest_categorical(m_name+"/v3_hs", [0, 1])
        m_args = [m_kernel, m_exp_ratio, m_out_channel, m_k_eca, m_hs, m_stride]
    

    if m_padding is None:
        m_padding = int(*autopad(m_kernel, m_padding))

    print('# module block : ', m)
    print(f"before ####### image_size : {image_size}, m_kernel : {m_kernel}, m_stride : {m_stride}, m_padding : {m_padding}")
    if m != "Pass":
        image_size = calculate_feat_size(image_size, m_kernel, m_stride, m_padding)
    print(f"after ####### image_size : {image_size}")

    if not m == "Pass":
        if m_stride==1:
            return [m_repeat, m, m_args], True, image_size
        else:
            return [m_repeat, m, m_args], False, image_size
    else:
        return None, None, image_size

def add_pooling(trial, depth, image_size):
    m_name = 'mp'+str(depth)
    m = trial.suggest_categorical(
        m_name,
        ["MaxPool",
         "AvgPool",
         "Pass"])
    if not m == "Pass":
        feat_size = calculate_feat_size(image_size, 3, 2, 1)
        return [1, m, [3,2,1]], feat_size
    else:
        return None, image_size
        
def search_model(trial: optuna.trial.Trial, image_size : int) -> List[Any]:
    """Search model structure from user-specified search space."""
    model = []
    n_pooling = 0 # Modify 1 -> 0
    # Example) ImageSize with downsampling
    # 32 -> 16 -> 8 -> 4 -> 2 (need 3 times downsampling(=stride2)) <- Competition size
    # 128 -> 64 -> 32 -> 16 -> 8 -> 4 (need 5 times downsampling(=stride2)) <- General size
    # 224 -> 112 -> 56 -> 28 -> 14 -> 7

    feat_size = image_size
    
    # Start Conv (or Depthwise)
    m1 = trial.suggest_categorical("m1", ["Conv", "DWConv"])
    m1_args = []
    m1_repeat = 1
    
    if m1=="Conv":
        m1_out_channel = trial.suggest_int("m1/out_channels", low=16, high=24, step=8)
    elif m1=="DWConv":
        m1_out_channel = trial.suggest_int("m1/out_channels", low=15, high=24, step=3)
    
    if MAX_NUM_POOLING==3:
        m1_stride = 1
    else:
        m1_stride = 2

    m1_activation = trial.suggest_categorical(
        "m1/activation", ["ReLU", "Hardswish"]
        )
    if m1 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m1_args = [m1_out_channel, 3, m1_stride, None, 1, m1_activation]
    elif m1 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m1_args = [m1_out_channel, 3, m1_stride, None, m1_activation]
    model.append([m1_repeat, m1, m1_args])
    
    feat_size = calculate_feat_size(feat_size, 3, m1_stride)
        
    # Module Layers (depths = max_depth)
    for depth in range(2,MAX_DEPTH+3):
        module_args, use_stride, feat_size = add_module(trial, depth, n_pooling, feat_size)
        if module_args is not None:
            model.append(module_args)
            if use_stride:
                if n_pooling<MAX_NUM_POOLING:
                    pool_args, feat_size = add_pooling(trial, depth, feat_size)
                    if pool_args is not None:
                        model.append(pool_args)
                        n_pooling+=1
            else:
                n_pooling+=1
    
    last_dim = trial.suggest_int("last_dim", low=512, high=1024, step=256)
    model.append([1, "Conv", [last_dim, 1, 1]])
        
    # GAP -> Classifier
    last_layer = trial.suggest_categorical("last", ["Linear", "Conv"])
    if last_layer == 'Linear':
        model.append([1, "GlobalAvgPool", []])
        model.append([1, "Flatten", []])
        model.append([1, "Linear", [CLASSES]])
    else:
        model.append([1, "Conv", [last_dim, 1, 1]])
        model.append([1, "GlobalAvgPool", []])
        model.append([1, "FixedConv", [6, 1, 1, None, 1, None]]) 
    
    return model

def objective(trial: optuna.trial.Trial, args, device) -> Tuple[float, int, float]:
    """Optuna objective.
    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    global BEST_MODEL_SCORE
    
    if args.model_name is None : 
        hyperparams = DEFAULT
        model_config: Dict[str, Any] = {}
        model_config["input_channel"] = 3
        model_config["depth_multiple"] = trial.suggest_categorical(
            "depth_multiple", [0.25, 0.5, 0.75, 1.0]
        )
        model_config["width_multiple"] = trial.suggest_categorical(
            "width_multiple", [0.25, 0.5, 0.75, 1.0]
        )
        model_config["backbone"] = search_model(trial, hyperparams["IMG_SIZE"])

    else :
        model_config_path = f"./configs/model/{args.model_name}.yaml"
        with open(model_config_path) as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        hyperparams = search_hyperparam(trial)
    
    model_config["INPUT_SIZE"] = [hyperparams["IMG_SIZE"], hyperparams["IMG_SIZE"]]

    model = Model(model_config, verbose=True)
    model.to(device)
    model.model.to(device)

    # check ./data_configs/data.yaml for config information
    data_config: Dict[str, Any] = {}
    data_config["DATA_PATH"] = DATA_PATH
    data_config["DATASET"] = "TACO"
    data_config["FP16"] = True
    data_config["AUG_TRAIN"] = "randaugment_train"
    data_config["AUG_TEST"] = "simple_augment_test"
    data_config["AUG_TRAIN_PARAMS"] = {
        "n_select": hyperparams["n_select"],
    }
    data_config["AUG_TEST_PARAMS"] = None
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["VAL_RATIO"] = 0.8
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]

    mean_time = check_runtime(
        model.model,
        [model_config["input_channel"]] + model_config["INPUT_SIZE"],
        device,
    )
    # model_info(model, verbose=True)
    train_loader, val_loader, _ = create_dataloader(data_config)

    criterion = nn.CrossEntropyLoss()

    if args.model_name is None :    # default hyperparams
        hyperparams["optimizer"] = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        hyperparams["scheduler"] = torch.optim.lr_scheduler.OneCycleLR(
            hyperparams["optimizer"],
            max_lr=0.1,
            steps_per_epoch=len(train_loader),
            epochs=hyperparams["EPOCHS"],
            pct_start=0.05,
        )

    trainer = TorchTrainer(
        model,
        criterion,
        hyperparams,
        device=device,
        verbose=1,
        model_path=RESULT_MODEL_PATH,
        trial=trial if args.model_name else None
    )
    trainer.train(train_loader, hyperparams["EPOCHS"], val_dataloader=val_loader)
    loss, f1_score, acc_percent = trainer.test(model, test_dataloader=val_loader)
    params_nums = count_model_params(model)

    # model_info(model, verbose=True)
    if f1_score > BEST_MODEL_SCORE:
        BEST_MODEL_SCORE = f1_score
        file_name = f"{f1_score:.2%}_{params_nums}_{mean_time:.3f}.yaml"
        if args.model_name: # Search hyperparams
            save_path = os.path.join('./configs/data', file_name)
            for key, value in data_config.items():
                if key in hyperparams.keys():
                    continue
                else :
                    hyperparams[key] = value
            with open(save_path, 'w') as f:
                yaml.dump(hyperparams, f)
            print("Complete saving hyperparameter yaml file to ", save_path)
        else: # Search model
            save_path = os.path.join('./configs/model', file_name)
            with open(save_path, 'w') as f:
                yaml.dump(model_config, f)
            print("Complete saving model yaml file to ", save_path)
    
    if args.model_name: # single-objective
        return f1_score
    else: # multi-objective
        return f1_score, params_nums, mean_time


def get_best_trial_with_condition(optuna_study: optuna.study.Study) -> Dict[str, Any]:
    """Get best trial that satisfies the minimum condition(e.g. accuracy > 0.8).
    Args:
        study : Optuna study object to get trial.
    Returns:
        best_trial : Best trial that satisfies condition.
    """
    df = optuna_study.trials_dataframe().rename(
        columns={
            "values_0": "acc_percent",
            "values_1": "params_nums",
            "values_2": "mean_time",
        }
    )
    ## minimum condition : accuracy >= threshold
    threshold = 0.7
    minimum_cond = df.acc_percent >= threshold

    if minimum_cond.any():
        df_min_cond = df.loc[minimum_cond]
        ## get the best trial idx with lowest parameter numbers
        best_idx = df_min_cond.loc[
            df_min_cond.params_nums == df_min_cond.params_nums.min()
        ].acc_percent.idxmax()

        best_trial_ = optuna_study.trials[best_idx]
        print("Best trial which satisfies the condition")
        print(df.loc[best_idx])
    else:
        print("No trials satisfies minimum condition")
        best_trial_ = None

    return best_trial_


def tune(gpu_id, args, storage: str = None):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    sampler = optuna.samplers.MOTPESampler()
    if storage is not None:
        print(f"****** storage url is {storage} ******")
        rdb_storage = optuna.storages.RDBStorage(url=storage)
        print(rdb_storage)
    else:
        rdb_storage = None
    
    if args.model_name:
        study_name = "automl_hyparams"
        directions = ["maximize"]
        pruner = optuna.pruners.HyperbandPruner()
    else:
        study_name = "automl1130"
        directions = ["maximize", "minimize", "minimize"]
        pruner = None # multi-objective cannot use pruner
    study = optuna.create_study(
        directions=directions,
        pruner = pruner,
        study_name=study_name,
        sampler=sampler,
        storage=rdb_storage,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, args, device), n_trials=100) # original: 500

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    best_trial = get_best_trial_with_condition(study)
    print(best_trial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    # parser.add_argument("--storage", default="sqlite:///automl.db", type=str, help="Optuna database storage path.") # make local db
    parser.add_argument("--model_name", default=None, type=str, help="Model config file name (if not None, search hyperparams)")
    parser.add_argument("--storage", default=f"mysql://metamong:{input('DB password: ')}@34.82.27.63/test", type=str, help="Optuna database storage path.")
    
    args = parser.parse_args()

    tune(args.gpu, args, storage=args.storage if args.storage != "" else None)
