# Based on jupyter notebok ColorizeTrainingStable.ipynb

import os
os.environ['CUDA_VISIBLE_DEVICES']='0' 

import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks.tensorboard import *
from fastai.vision.gan import *
from fasterai.generators import *
from fasterai.critics import *
from fasterai.dataset import *
from fasterai.loss import *
from fasterai.save import *
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFile
from torch.utils.data.sampler import RandomSampler, SequentialSampler


# W&B related imports
import wandb
from wandb.fastai import WandbCallback
from functools import partialmethod


path = Path('/data/Open_Images')
path_hr = path
path_lr = path/'bandw'

proj_id = 'StableModel'

gen_name = proj_id + '_gen'
pre_gen_name = gen_name + '_0'
crit_name = proj_id + '_crit'

name_gen = proj_id + '_image_gen'
path_gen = path/name_gen

TENSORBOARD_PATH = path / ('tensorboard/' + proj_id)


def get_data(bs:int, sz:int, keep_pct:float, random_seed=None, samplers=None):
    return get_colorize_data(sz=sz, bs=bs, crappy_path=path_lr, good_path=path_hr, 
                             random_seed=random_seed, keep_pct=keep_pct, samplers=samplers)

def get_crit_data(classes, bs, sz):
    src = ImageList.from_folder(path, include=classes, recurse=True).random_split_by_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=sz)
           .databunch(bs=bs).normalize(imagenet_stats))
    return data

def create_training_images(fn,i):
    dest = path_lr/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn).convert('LA').convert('RGB')
    img.save(dest)  
    
def save_preds(dl):
    i=0
    names = dl.dataset.items
    
    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen/names[i].name)
            i += 1
    
def save_gen_images():
    if path_gen.exists(): shutil.rmtree(path_gen)
    path_gen.mkdir(exist_ok=True)
    data_gen = get_data(bs=bs, sz=sz, keep_pct=0.085)
    save_preds(data_gen.fix_dl)
    PIL.Image.open(path_gen.ls()[0])

# Reduce quantity of samples per training epoch
# Adapted from https://forums.fast.ai/t/epochs-of-arbitrary-length/27777/10

@classmethod
def create(cls, train_ds:Dataset, valid_ds:Dataset, test_ds:Optional[Dataset]=None, path:PathOrStr='.', bs:int=64,
            val_bs:int=None, num_workers:int=defaults.cpus, dl_tfms:Optional[Collection[Callable]]=None,
            device:torch.device=None, collate_fn:Callable=data_collate, no_check:bool=False, sampler=None, **dl_kwargs)->'DataBunch':
    "Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds` with a batch size of `bs`. Passes `**dl_kwargs` to `DataLoader()`"
    datasets = cls._init_ds(train_ds, valid_ds, test_ds)
    val_bs = ifnone(val_bs, bs)
    if sampler is None: sampler = [RandomSampler] + 3*[SequentialSampler]
    dls = [DataLoader(d, b, sampler=sa(d), drop_last=sh, num_workers=num_workers, **dl_kwargs) for d,b,sh,sa in
            zip(datasets, (bs,val_bs,val_bs,val_bs), (True,False,False,False), sampler) if d is not None]
    return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)

ImageDataBunch.create = create
ImageImageList._bunch = ImageDataBunch

class FixedLenRandomSampler(RandomSampler):
    def __init__(self, data_source, epoch_size):
        super().__init__(data_source)
        self.epoch_size = epoch_size
        self.not_sampled = np.array([True]*len(data_source))
    
    @property
    def reset_state(self): self.not_sampled[:] = True
        
    def __iter__(self):
        ns = sum(self.not_sampled)
        idx_last = []
        if ns >= len(self):
            idx = np.random.choice(np.where(self.not_sampled)[0], size=len(self), replace=False).tolist()
            if ns == len(self): self.reset_state
        else:
            idx_last = np.where(self.not_sampled)[0].tolist()
            self.reset_state
            idx = np.random.choice(np.where(self.not_sampled)[0], size=len(self)-len(idx_last), replace=False).tolist()
        self.not_sampled[idx] = False
        idx = [*idx_last, *idx]
        return iter(idx)
    
    def __len__(self):
        return self.epoch_size


# W&B config
wandb.init(project="DeOldify")
config = wandb.config  # for shortening
config.epoch_size = 25000
config.nf_factor = 2
config.pct_start = 0.3
config.step1_bs = 12
config.step1_sz = 64
config.step1a_epochs = 10
config.step1a_pct_start = 0.8
config.step1a_lr = 1e-3
config.step1b_epochs = 10
config.step1b_pct_start = 0.3
config.step1b_lr_min = 3e-7
config.step1b_lr_max = 3e-4
config.step2_bs = 2
config.step2_sz = 128
config.step2_epochs = 10
config.step2_pct_start = 0.3
config.step2_lr_min = 1e-7
config.step2_lr_max = 1e-4
config.step3_bs = 1
config.step3_sz = 192
config.step3_epochs = 10
config.step3_pct_start = 0.3
config.step3_lr_min = 5e-8
config.step3_lr_max = 5e-5
random_seed = 1

# Load data
train_sampler = partial(FixedLenRandomSampler, epoch_size=config.epoch_size // config.step1_bs * config.step1_bs)
samplers = [train_sampler, SequentialSampler, SequentialSampler, SequentialSampler]

# Step 1a: 64 px, pre-trained

data_gen = get_data(bs=config.step1_bs, sz=config.step1_sz, keep_pct=1., random_seed=random_seed, samplers=samplers)
print(data_gen)
learn_gen=gen_learner_wide(data=data_gen, gen_loss=FeatureLoss(), nf_factor=config.nf_factor)
learn_gen.callback_fns.append(partial(WandbCallback, input_type='images'))  # log prediction samples
learn_gen.fit_one_cycle(config.step1a_epochs, pct_start=config.step1a_pct_start, max_lr=slice(config.step1a_lr))
learn_gen.save(pre_gen_name)

# Step 1b: 64 px, unfreeze

learn_gen.unfreeze()
learn_gen.fit_one_cycle(config.step1b_epochs, pct_start=config.step1b_pct_start, max_lr=slice(config.step1b_lr_min, config.step1b_lr_max))
learn_gen.save(pre_gen_name)

# Step 2: 128 px, unfreeze

learn_gen.data = get_data(bs=config.step2_bs, sz=config.step2_sz, keep_pct=1., random_seed=random_seed, samplers=samplers)
learn_gen.unfreeze()
learn_gen.fit_one_cycle(config.step2_epochs, pct_start=config.step2_pct_start, max_lr=slice(config.step2_lr_min, config.step2_lr_max))
learn_gen.save(pre_gen_name)

# Step 3: 192 px, unfreeze

learn_gen.data = get_data(bs=config.step3_bs, sz=config.step3_sz, keep_pct=1., random_seed=random_seed, samplers=samplers)
learn_gen.unfreeze()
learn_gen.fit_one_cycle(config.step3_epochs, pct_start=config.step3_pct_start, max_lr=slice(config.step3_lr_min, config.step3_lr_max))
learn_gen.save(pre_gen_name)


