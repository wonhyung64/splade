import json
import os

from omegaconf import OmegaConf

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from splade.evaluation.eval import load_and_evaluate
from splade.utils.utils import get_dataset_name
from splade.utils.hydra import hydra_chdir


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




CONFIG_PATH = '/root/won/splade/splade/conf/'
CONFIG_NAME = 'config_splade_base'

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
# config["record_frequency"] = 100

# for dataset EVAL_QREL_PATH
# for metric of this qrel
# config["EVAL_QREL_PATH"]
# eval_qrel_path = exp_dict.data.EVAL_QREL_PATH
eval_qrel_path = ["data/msmarco/dev_qrel.json"]
# eval_metric = exp_dict.config.eval_metric
eval_metric = ["mrr_10", "recall"]
# dataset_names = exp_dict.config.retrieval_name
dataset_names = ["MSMARCO"]
# out_dir = exp_dict.config.out_dir
out_dir = config["out_dir"]

res_all_datasets = {}
for i, (qrel_file_path, eval_metrics, dataset_name) in enumerate(zip(eval_qrel_path, eval_metric, dataset_names)):
    print(qrel_file_path, eval_metrics, dataset_name)

    if qrel_file_path is not None:
        res = {}
        print(eval_metrics)
        for metric in eval_metrics:
            qrel_fp=qrel_file_path
            res.update(load_and_evaluate(qrel_file_path=qrel_fp,
                                            run_file_path=os.path.join(out_dir, dataset_name, 'run.json'),
                                            metric=eval_metric[1]))
                                        #  metric=metric))
        if dataset_name in res_all_datasets.keys():
            res_all_datasets[dataset_name].update(res)
        else:
            res_all_datasets[dataset_name] = res
        out_fp = os.path.join(out_dir, dataset_name, "perf.json")
        json.dump(res, open(out_fp,"a"))
out_all_fp= os.path.join(out_dir, "perf_all_datasets.json")
json.dump(res_all_datasets, open(out_all_fp, "a"))
