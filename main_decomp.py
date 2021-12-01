"""
main code to model decompositions (tensor decompositions)
base code https://github.com/jacobgil/pytorch-tensor-decompositions
"""

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import yaml
import time
import os
from datetime import datetime

import tensorly as tl

from src.model import Model
from src.loss import CustomCriterion
from src.trainer import TorchTrainer
from src.dataloader import create_dataloader
from src.utils.torch_utils import calculate_macs
from src.utils.common import get_label_counts, read_yaml
from src.decompositions.decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer


def conv_decompositions(is_cp, model):
    """
    if the layer has Conv2d, do decomposition
    """
    for i, key in enumerate(model._modules):
        if isinstance(model._modules[key], torch.nn.modules.conv.Conv2d):
            print(i, model._modules[key])
            if is_cp:
                rank = max(model._modules[key].weight.data.numpy().shape)//3
                model._modules[key] = cp_decomposition_conv_layer(model._modules[key], rank)
            else:
                model._modules[key] = tucker_decomposition_conv_layer(model._modules[key])
        else:
            conv_decompositions(is_cp, model._modules[key])
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Decomposition model.")
    parser.add_argument("--cp", dest="cp", action="store_true")
    parser.add_argument(
        "--model",
        default="mobilenetv3",
        type=str,
        help="model config file name",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="model checkpoint path (required)",
        required=True
    )
    parser.add_argument(
        "--data", default="taco", type=str, help="data config file name"
    )
    parser.set_defaults(cp=False)
    args = parser.parse_args()

    model_cfg_path = os.path.join("configs/model", args.model + ".yaml")
    data_cfg_path = os.path.join("configs/data", args.data + ".yaml")

    model_config = read_yaml(cfg=model_cfg_path)
    data_config = read_yaml(cfg=data_cfg_path)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])
    model_config["CHECKPOINT_PATH"] = args.checkpoint

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp-decomp", 'latest'))

    if os.path.exists(log_dir): 
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)

    model_instance = Model(model_config, verbose=True)
    checkpoint_path = os.path.join(args.checkpoint, "best.pt")
    print(f"Checkpoint path: {checkpoint_path}")
    if os.path.isfile(checkpoint_path):
        model_instance.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
    model_instance.model.to(device)

    # Create dataloader
    # train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create criterion
    # criterion = CustomCriterion(
    #     samples_per_cls=get_label_counts(data_config["DATA_PATH"])
    #     if data_config["DATASET"] == "TACO"
    #     else None,
    #     device=device,
    # )

    # trainer = TorchTrainer(
    #     model_instance.model,
    #     criterion,
    #     data_config,
    #     device=device,
    #     verbose=1,
    #     model_path=None,
    #     trial=None,
    # )

    before_macs = calculate_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    new_model = conv_decompositions(args.cp, model_instance.model.cpu()) # tensor->numpy 라서 cuda X
    after_macs = calculate_macs(new_model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    
    print(new_model)
    # model.eval()
    # model.cpu()
    # N = len(model.features._modules.keys())
    # for i, key in enumerate(model.features._modules.keys()):

    #     if i >= N - 2:
    #         break
    #     if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
    #         conv_layer = model.features._modules[key]
    #         if args.cp:
    #             rank = max(conv_layer.weight.data.numpy().shape)//3
    #             decomposed = cp_decomposition_conv_layer(conv_layer, rank)
    #         else:
    #             decomposed = tucker_decomposition_conv_layer(conv_layer)

    #         model.features._modules[key] = decomposed

    #     torch.save(model, 'decomposed_model')


    # elif args.fine_tune:
    #     base_model = torch.load("decomposed_model")
    #     model = torch.nn.DataParallel(base_model)

    #     for param in model.parameters():
    #         param.requires_grad = True

    #     print(model)
    #     model.cuda()        

    #     if args.cp:
    #         optimizer = optim.SGD(model.parameters(), lr=0.000001)
    #     else:
    #         # optimizer = optim.SGD(chain(model.features.parameters(), \
    #         #     model.classifier.parameters()), lr=0.01)
    #         optimizer = optim.SGD(model.parameters(), lr=0.001)


    #     trainer = Trainer(args.train_path, args.test_path, model, optimizer)

    #     trainer.test()
    #     model.cuda()
    #     model.train()
    #     trainer.train(epoches=100)
    #     model.eval()
    #     trainer.test()