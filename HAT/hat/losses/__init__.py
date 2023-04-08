import importlib
from os import path as osp

from basicsr.utils import scandir, get_root_logger

# automatically scan and import model modules for registry
# scan all the files that end with '_model.py' under the model folder
loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_loss.py')]
# import all the model modules
_loss_modules = [importlib.import_module(f'hat.losses.{file_name}') for file_name in loss_filenames]