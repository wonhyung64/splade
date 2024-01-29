#%%
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

# import sys
import yaml
# import hydra
import torch

from tqdm import tqdm
# from omegaconf import open_dict
from omegaconf import OmegaConf
# from omegaconf import DictConfig
# from torch.utils import data


# import json
# from collections import defaultdict

# from omegaconf import open_dict
# from tqdm.auto import tqdm

# from splade.tasks.base.trainer import TrainerIter
# from splade.utils.metrics import init_eval

from splade.tasks import amp
from splade.utils.utils import parse
from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
# from splade.datasets.dataloaders import CollectionDataLoader
from splade.datasets.dataloaders import  SiamesePairsDataLoader, DistilSiamesePairsDataLoader
from splade.datasets.datasets import PairsDatasetPreLoad, DistilPairsDatasetPreLoad, MsMarcoHardNegatives
# from splade.datasets.datasets import CollectionDatasetPreLoad
from splade.losses.regularization import init_regularizer, RegWeightScheduler
from splade.models.models_utils import get_model
from splade.optim.bert_optim import init_simple_bert_optim
# from splade.tasks.transformer_evaluator import SparseApproxEvalWrapper
# from splade.tasks.transformer_trainer import SiameseTransformerTrainer
from splade.utils.utils import get_loss, set_seed_from_config
# from splade.utils.utils import set_seed, restore_model, get_initialize_config


class MakeObj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [MakeObj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, MakeObj(b) if isinstance(b, dict) else b)


def read_yaml(path):
    x_dict = OmegaConf.load(path)
    x_yaml_str = OmegaConf.to_yaml(x_dict)
    x_obj = MakeObj(x_dict)
    
    return x_dict, x_yaml_str, x_obj


#%%
CONFIG_PATH = '/root/won/splade/splade/conf/'
CONFIG_NAME = 'config_splade_base'
# CONFIG_NAME = 'config_default'
_, __, x_obj = read_yaml(f"{CONFIG_PATH}{CONFIG_NAME}.yaml")
x_obj.config
config = {
    "data": {},
    "init_dict": {}
    }
exp_dict = {"data": {}}
init_dict = {}

for yaml_dict in x_obj.defaults:
    yaml_name = f'{"/".join(yaml_dict.items()[0])}.yaml'
    yaml_ = yaml.load(open(f"{CONFIG_PATH}{yaml_name}"), Loader=yaml.Loader)

    if yaml_.get("config"):
        config.update(yaml_.get("config"))

    if yaml_.get("data"):
        exp_dict["data"].update(yaml_.get("data"))
        config["data"].update(yaml_.get("data"))

    if yaml_.get("init_dict"):
        init_dict.update(yaml_.get("init_dict"))
        config.update(yaml_.get("init_dict"))
    
    else:
        config.update(yaml_)

del config["data"]
del config["init_dict"]
del config["config"]
config.update(x_obj.config)

init_dict["fp16"] = config["fp16"]
# config["train_batch_size"] = 32
# config["max_length"] = 10

#%%
# @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
# def train(exp_dict: DictConfig):
    # exp_dict, config, init_dict, _ = get_initialize_config(exp_dict, train=True)
model = get_model(config, init_dict)

random_seed = set_seed_from_config(config)

optimizer, scheduler = init_simple_bert_optim(model, lr=float(config["lr"]), warmup_steps=config["warmup_steps"],
                                                weight_decay=config["weight_decay"],
                                                num_training_steps=config["nb_iterations"])
config["warmup_steps"]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

################################################################
# CHECK IF RESUME TRAINING
################################################################
iterations = (1, config["nb_iterations"] + 1)  # tuple with START and END
regularizer = None
# if os.path.exists(os.path.join(config["checkpoint_dir"], "model_ckpt/model_last.tar")):
#     print("@@@@ RESUMING TRAINING @@@")
#     print("WARNING: change seed to change data order when restoring !")
#     set_seed(random_seed + 666)
#     if device == torch.device("cuda"):
#         ckpt = torch.load(os.path.join(config["checkpoint_dir"], "model_ckpt/model_last.tar"))
#     else:
#         ckpt = torch.load(os.path.join(config["checkpoint_dir"], "model_ckpt/model_last.tar"), map_location=device)
#     print("starting from step", ckpt["step"])
#     print("{} remaining iterations".format(iterations[1] - ckpt["step"]))
#     iterations = (ckpt["step"] + 1, config["nb_iterations"])
#     restore_model(model, ckpt["model_state_dict"])
#     optimizer.load_state_dict(ckpt["optimizer_state_dict"])
#     if device == torch.device("cuda"):
#         for state in optimizer.state.values():
#             for k, v in state.items():
#                 if torch.is_tensor(v):
#                     state[k] = v.cuda()
#     scheduler.load_state_dict(ckpt["scheduler_state_dict"])
#     if "regularizer" in ckpt:
#         print("loading regularizer")
#         regularizer = ckpt.get("regularizer", None)

if torch.cuda.device_count() > 1:
    print(" --- use {} GPUs --- ".format(torch.cuda.device_count()))
    # model = torch.nn.DataParallel(model)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

model.to(device)

loss_fn = get_loss(config)

# initialize regularizer dict
if "regularizer" in config and regularizer is None:  # else regularizer is loaded
    output_dim = model.module.output_dim if hasattr(model, "module") else model.output_dim
    regularizer = {
        "eval": {
            "L0": {"loss": init_regularizer("L0")},
            "sparsity_ratio": {"loss": init_regularizer("sparsity_ratio", output_dim=output_dim)}
            },
        "train": {}}
    if config["regularizer"] == "eval_only":
        # just in the case we train a model without reg but still want the eval metrics like L0
        pass
    else:
        for reg in config["regularizer"]:
            temp = {"loss": init_regularizer(config["regularizer"][reg]["reg"]),
                    "targeted_rep": config["regularizer"][reg]["targeted_rep"]}
            d_ = {}
            if "lambda_q" in config["regularizer"][reg]:
                d_["lambda_q"] = RegWeightScheduler(float(config["regularizer"][reg]["lambda_q"]),
                                                    config["regularizer"][reg]["T"])
            if "lambda_d" in config["regularizer"][reg]:
                d_["lambda_d"] = RegWeightScheduler(float(config["regularizer"][reg]["lambda_d"]),
                                                    config["regularizer"][reg]["T"])
            temp["lambdas"] = d_  # it is possible to have reg only on q or d if e.g. you only specify lambda_q
            # in the reg config
            # targeted_rep is just used to indicate which rep to constrain (if e.g. the model outputs several
            # representations)
            # the common case: model outputs "rep" (in forward) and this should be the value for this targeted_rep
            regularizer["train"][reg] = temp

# fix for current in batch neg losses that break on last batch
if config["loss"] in ("InBatchNegHingeLoss", "InBatchPairwiseNLL"):
    drop_last = True
else:
    drop_last = False


# if exp_dict["data"].get("type", "") == "triplets":
if config.get("type", "") == "triplets":
    # data_train = PairsDatasetPreLoad(data_dir=exp_dict["data"]["TRAIN_DATA_DIR"])
    data_train = PairsDatasetPreLoad(data_dir=config["TRAIN_DATA_DIR"])
    train_mode = "triplets"
# elif exp_dict["data"].get("type", "") == "triplets_with_distil":
elif config.get("type", "") == "triplets_with_distil":
    data_train = DistilPairsDatasetPreLoad(data_dir=config["TRAIN_DATA_DIR"])
    train_mode = "triplets_with_distil"
# elif exp_dict["data"].get("type", "") == "hard_negatives":
elif config.get("type", "") == "hard_negatives":
    data_train = MsMarcoHardNegatives(
        dataset_path=config["TRAIN"]["DATASET_PATH"],
        document_dir=config["TRAIN"]["D_COLLECTION_PATH"],
        query_dir=exp_dict["data"]["TRAIN"]["Q_COLLECTION_PATH"],
        qrels_path=exp_dict["data"]["TRAIN"]["QREL_PATH"])
    train_mode = "triplets_with_distil"
else:
    raise ValueError("provide valid data type for training")


# val_loss_loader = None  # default
# if "VALIDATION_SIZE_FOR_LOSS" in config:
#     print("initialize loader for validation loss")
#     print("split train, originally {} pairs".format(len(data_train)))
#     data_train, data_val = torch.utils.data.random_split(data_train, lengths=[
#         # len(data_train) - exp_dict["data"]["VALIDATION_SIZE_FOR_LOSS"],
#         len(data_train) - config["VALIDATION_SIZE_FOR_LOSS"],
#         # exp_dict["data"]["VALIDATION_SIZE_FOR_LOSS"]])
#         config["VALIDATION_SIZE_FOR_LOSS"]])
#     print("train: {} pairs ~~ val: {} pairs".format(len(data_train), len(data_val)))
#     if train_mode == "triplets":
#         val_loss_loader = SiamesePairsDataLoader(dataset=data_val, batch_size=config["eval_batch_size"],
#                                                     shuffle=False,
#                                                     num_workers=4,
#                                                     tokenizer_type=config["tokenizer_type"],
#                                                     max_length=config["max_length"], drop_last=drop_last)
#     elif train_mode == "triplets_with_distil":
#         val_loss_loader = DistilSiamesePairsDataLoader(dataset=data_val, batch_size=config["eval_batch_size"],
#                                                         shuffle=False,
#                                                         num_workers=4,
#                                                         tokenizer_type=config["tokenizer_type"],
#                                                         max_length=config["max_length"], drop_last=drop_last)
#     else:
#         raise NotImplementedError


if train_mode == "triplets":
    train_loader = SiamesePairsDataLoader(dataset=data_train, batch_size=config["train_batch_size"], shuffle=True,
                                            num_workers=4,
                                            tokenizer_type=config["tokenizer_type"],
                                            max_length=config["max_length"], drop_last=drop_last)
elif train_mode == "triplets_with_distil":
    train_loader = DistilSiamesePairsDataLoader(dataset=data_train, batch_size=config["train_batch_size"],
                                                shuffle=True,
                                                num_workers=4,
                                                tokenizer_type=config["tokenizer_type"],
                                                max_length=config["max_length"], drop_last=drop_last)
else:
    raise NotImplementedError

# val_evaluator = None
# if "VALIDATION_FULL_RANKING" in exp_dict["data"]:
#     with open_dict(config):
#         config["val_full_rank_qrel_path"] = exp_dict["data"]["VALIDATION_FULL_RANKING"]["QREL_PATH"]
#     full_ranking_d_collection = CollectionDatasetPreLoad(
#         data_dir=exp_dict["data"]["VALIDATION_FULL_RANKING"]["D_COLLECTION_PATH"], id_style="row_id")
#     full_ranking_d_loader = CollectionDataLoader(dataset=full_ranking_d_collection,
#                                                     tokenizer_type=config["tokenizer_type"],
#                                                     max_length=config["max_length"],
#                                                     batch_size=config["eval_batch_size"],
#                                                     shuffle=False, num_workers=4)
#     full_ranking_q_collection = CollectionDatasetPreLoad(
#         data_dir=exp_dict["data"]["VALIDATION_FULL_RANKING"]["Q_COLLECTION_PATH"], id_style="row_id")
#     full_ranking_q_loader = CollectionDataLoader(dataset=full_ranking_q_collection,
#                                                     tokenizer_type=config["tokenizer_type"],
#                                                     max_length=config["max_length"], batch_size=1,
#                                                     # TODO fix: bs currently set to 1
#                                                     shuffle=False, num_workers=4)
#     val_evaluator = SparseApproxEvalWrapper(model,
#                                             config={"top_k": exp_dict["data"]["VALIDATION_FULL_RANKING"]["TOP_K"],
#                                                     "out_dir": os.path.join(config["checkpoint_dir"],
#                                                                             "val_full_ranking")
#                                                     },
#                                             collection_loader=full_ranking_d_loader,
#                                             q_loader=full_ranking_q_loader,
#                                             restore=False)


#%%
# #################################################################
# # TRAIN
# #################################################################
print("+++++ BEGIN TRAINING +++++")


# trainer = SiameseTransformerTrainer(model=model, iterations=iterations, loss=loss, optimizer=optimizer,
#                                     config=config, scheduler=scheduler,
#                                     train_loader=train_loader, validation_loss_loader=val_loss_loader,
#                                     validation_evaluator=val_evaluator,
#                                     regularizer=regularizer)
# trainer.train()
fp16 = config["fp16"]
nb_iterations = config["nb_iterations"]

# model.to(device)
moving_avg_ranking_loss = 0
mpm = amp.MixedPrecisionManager(config["fp16"])
scaler = torch.cuda.amp.GradScaler(enabled=True)
# optimizer.zero_grad()

# for epoch in range(epochs):
#     for input, target in zip(data, targets):
#         with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
#             output = net(input)
#             loss = loss_fn(output, target)
#         scaler.scale(loss).backward()
#         scaler.step(opt)
#         scaler.update()
#         opt.zero_grad() # set_to_none=True here can modestly improve performance
# end_timer_and_print("Mixed precision:")


for i in tqdm(range(1, nb_iterations + 1)):
    optimizer.zero_grad()
    train_iterator = iter(train_loader)

    model.train()  # train model
    try:
        batch = next(train_iterator)
    except StopIteration:
        # when nb_iterations > len(data_loader)
        train_iterator = iter(train_loader)
        batch = next(train_iterator)

    with mpm.context():
        for k, v in batch.items():
            batch[k] = v.to(device)

        q_kwargs = parse(batch, "q")
        d_pos_kwargs = parse(batch, "pos")
        d_neg_kwargs = parse(batch, "neg")
        d_pos_args = {"q_kwargs": q_kwargs, "d_kwargs": d_pos_kwargs}
        d_neg_args = {"q_kwargs": q_kwargs, "d_kwargs": d_neg_kwargs}

        if "augment_pairs" in config:
            if config["augment_pairs"] == "in_batch_negatives":
                d_pos_args["score_batch"] = True  # meaning that for the POSITIVE documents in the batch, we will
                # actually compute all the scores w.r.t. the queries in the batch
            else:
                raise NotImplementedError


        ###ModelPredict###
        # with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
        with torch.cuda.amp.autocast(enabled=True):
            out_pos = model(**d_pos_args)
            out_neg = model(**d_neg_args)
        ##################

            out = {}
            for k, v in out_pos.items():
                out["pos_{}".format(k)] = v
            for k, v in out_neg.items():
                out["neg_{}".format(k)] = v
            if "teacher_p_score" in batch:  # distillation pairs dataloader
                out["teacher_pos_score"] = batch["teacher_p_score"]
                out["teacher_neg_score"] = batch["teacher_n_score"]


            loss = loss_fn(out).mean()  # we need to average as we obtain one loss per GPU in DataParallel
            moving_avg_ranking_loss = 0.99 * moving_avg_ranking_loss + 0.01 * loss.item()
            # training moving average for logging
            if regularizer is not None:
                if "train" in regularizer:
                    regularization_losses = {}
                    for reg in regularizer["train"]:
                        if "lambda_q" in regularizer["train"][reg]["lambdas"]:
                            lambda_q = regularizer["train"][reg]["lambdas"]["lambda_q"].step()
                        else:
                            lambda_q = False

                        if "lambda_d" in regularizer["train"][reg]["lambdas"]:
                            lambda_d = regularizer["train"][reg]["lambdas"]["lambda_d"].step()
                        else:
                            lambda_d = False

                        targeted_rep = regularizer["train"][reg]["targeted_rep"]  # used to select the "name"
                        # of the representation to regularize (for instance the model could output several
                        # representations e.g. a semantic rep and a lexical rep) => this is just a general case
                        # for the Trainer
                        regularization_losses[reg] = 0
                        if lambda_q:
                            regularization_losses[reg] += (regularizer["train"][reg]["loss"](
                                out["pos_q_{}".format(targeted_rep)]) * lambda_q).mean()
                        if lambda_d:
                            regularization_losses[reg] += ((regularizer["train"][reg]["loss"](
                                out["pos_d_{}".format(targeted_rep)]) * lambda_d).mean() +
                                                            (regularizer["train"][reg]["loss"](
                                                                out["neg_d_{}".format(
                                                                    targeted_rep)]) * lambda_d).mean()) / 2
                        # NOTE: we take the rep of pos q for queries, but it would be equivalent to take the neg
                        # (because we consider triplets, so the rep of pos and neg are the same)
                        loss += sum(regularization_losses.values())
            # with torch.no_grad():
            #     monitor_losses = {}
            #     for reg in regularizer["eval"]:
            #         monitor_losses["{}_q".format(reg)] = regularizer["eval"][reg]["loss"](
            #             out["pos_q_rep"]).mean()
            #         # again, we can choose pos_q_rep or neg_q_rep indifferently
            #         monitor_losses["{}_d".format(reg)] = (regularizer["eval"][reg]["loss"](
            #             out["pos_d_rep"]).mean() + regularizer["eval"][reg]["loss"](
            #             out["neg_d_rep"]).mean()) / 2
            # when multiple GPUs, we need to aggregate the loss from the different GPUs (that's why the .mean())
            # see https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
            # for gradient accumulation  # TODO: check if everything works with gradient accumulation
            loss = loss / config["gradient_accumulation_steps"]


    # loss.backward()
    # optimizer.step()
    # scaler.scale(loss).backward()
    # scaler.step(optimizer) #infinite loop error
    # scaler.update()
    # perform gradient update:
    mpm.backward(loss)

    if i % config["gradient_accumulation_steps"] == 0:
        mpm.step(optimizer)
    if scheduler is not None:
        scheduler.step()
            # scheduler.get_last_lr()[0] #wandb
            # self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], i - 1)
    optimizer.zero_grad() # set_to_none=True here can modestly improve performance

    # if i % config["train_monitoring_freq"] == 0:
        # training_res_handler.write("{},{}\n".format(i, loss.item()))
        # writer.add_scalar("batch_train_loss", loss.item(), i) #wandb
        # self.writer.add_scalar("moving_avg_ranking_loss", moving_avg_ranking_loss, i) #wandb
        # print("+batch_loss_iter{}: {}".format(i, round(loss.item(), 4)))
        # if regularizer is not None:
        #     if "train" in regularizer:
        #         for reg_loss in regularization_losses:
        #             pass
                    # self.writer.add_scalar("batch_{}".format(reg_loss), regularization_losses[reg_loss].item(), i) # wandb
        #     for monitor_loss in monitor_losses:
        #         pass
            #     self.writer.add_scalar("batch_{}".format(monitor_loss),
            #                             monitor_losses[monitor_loss].item(), i)
# save_checkpoint(step=i, perf=loss, is_best=False, final_checkpoint=True)  # save the last anyway



#         model_to_save = model.module if hasattr(self.model, "module") else self.model  # when using DataParallel
#         # it is practical (although redundant) to save model weights using huggingface API, because if the model has
#         # no other params, we can reload it easily with .from_pretrained()
#         output_dir = os.path.join(self.config["checkpoint_dir"], "model")
#         model_to_save.transformer_rep.transformer.save_pretrained(output_dir)
#         tokenizer = model_to_save.transformer_rep.tokenizer
#         tokenizer.save_pretrained(output_dir)
#         if model_to_save.transformer_rep_q is not None:
#             output_dir_q = os.path.join(self.config["checkpoint_dir"], "model_q")
#             model_to_save.transformer_rep_q.transformer.save_pretrained(output_dir_q)
#             tokenizer = model_to_save.transformer_rep_q.tokenizer
#             tokenizer.save_pretrained(output_dir_q)
#         super().save_checkpoint(**kwargs)

# %%
