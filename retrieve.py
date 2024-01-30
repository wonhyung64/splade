#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import yaml
from omegaconf import OmegaConf

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from splade.datasets.dataloaders import CollectionDataLoader
from splade.datasets.datasets import CollectionDatasetPreLoad
# from splade.evaluate import evaluate
from splade.models.models_utils import get_model
from splade.tasks.transformer_evaluator import SparseRetrieval
from splade.utils.utils import get_dataset_name, get_initialize_config


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
# @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")

# def index(exp_dict: DictConfig):
CONFIG_PATH = '/root/won/splade/splade/conf/'
CONFIG_NAME = 'config_splade_base'
checkpoint = "/root/won/splade/models/splade_max_base/checkpoint/expt_240129_033537_243073/model/iter_50000"

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

config["hf_training"] = 1
init_dict["fp16"] = config["fp16"]
init_dict["model_type_or_dir"] = checkpoint
# config["train_batch_size"] = 32
# config["max_length"] = 10
# config["record_frequency"] = 100


#%%
model = get_model(config, init_dict)

batch_size = 1
# NOTE: batch_size is set to 1, currently no batched implem for retrieval (TODO)
for data_dir in set(exp_dict["data"]["Q_COLLECTION_PATH"]):
    q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
    q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=config["tokenizer_type"],
                                    max_length=config["max_length"], batch_size=batch_size,
                                    shuffle=False, num_workers=1)
    evaluator = SparseRetrieval(config=config, model=model, dataset_name=get_dataset_name(data_dir),
                                compute_stats=True, dim_voc=model.output_dim)
    evaluator.retrieve(q_loader, top_k=config["top_k"], threshold=config["threshold"])
