#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                # _source_path2 ���� �ڿ� ���ڰ� ���� �̸����� ���� �ɼ� OFF
                raw = key[1:]
                if not raw.endswith("2"):
                    shorthand = True
                key = raw
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._source_path2 = ""      # �� ��° �����ͼ� ���(EDINA)
        self._model_path2 = ""       # �� ��° pretrained model ���(EDINA)
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    # 0) Parser �⺻�� �б�
    args_defaults = parser.parse_args([])

    # 1) ���� CLI ���� �б�
    args_cmd = parser.parse_args(sys.argv[1:])

    # 2) config ���� �б�
    cfgfile_string = "Namespace()"
    try:
        cfgfilepath = os.path.join(args_cmd.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as f:
            cfgfile_string = f.read()
            print("Loaded cfg_args:", cfgfile_string)
    except (TypeError, FileNotFoundError):
        print("No config file found, skipping")

    args_cfg = eval(cfgfile_string)

    # 3) Merge: defaults �� config �� CLI(�⺻���� �ƴ� ��츸)
    merged = vars(args_defaults).copy()
    merged.update(vars(args_cfg))
    for k, v in vars(args_cmd).items():
        if v != getattr(args_defaults, k):
            merged[k] = v

    # 4) ���� Namespace ����
    final_args = Namespace(**merged)

    return final_args