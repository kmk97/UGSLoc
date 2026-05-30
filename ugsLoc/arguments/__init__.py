# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr

from argparse import ArgumentParser, Namespace
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
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
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0

        self.appearance_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False
        
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
        super().__init__(parser, "Pipeline Parameters")

class VirtualPipelineParams2():
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")

class LocalizationParams(ParamGroup):
    def __init__(self, parser):
        self.use_weighted_average = True
        self.num_test_cams = 10
        super().__init__(parser, "Localization Parameters")

# Paper CPR defaults (not exposed on CLI)
CPR_DEFAULTS = dict(
    new_uncertainty=True,
    robust_loss='soft_l1',
    alpha=1.0,
    sigma_min=1.0,
    clip_range=1e3,
    crop_query=False,
    weighted_epnp=False,
    max_weighted_pts=512,
    use_ransac=True,
    uncertainty_only_epnp=False,
    iterative_pnp=False,
    iterative_steps=3,
    importance_ransac=True,
    ransac_importance_iters=1000,
    ransac_importance_sample_size=6,
    importance_alpha=1.0,
    lo_refine=True,
    weighted_refine=False,
)

def get_appearance_args(scene_name, ft_path):
    """Fixed appearance-model settings for Cambridge eval."""
    return Namespace(
        multires=10,
        multires_views=4,
        i_embed=0,
        reduce_embedding=-1,
        use_viewdirs=True,
        netdepth=8,
        netwidth=128,
        use_fusion_res=False,
        no_fusion_BN=False,
        multi_gpu=False,
        N_importance=64,
        N_samples=64,
        NeRFW=True,
        in_channels_a=50,
        in_channels_t=20,
        netchunk=2097152,
        no_grad_update=True,
        basedir='./logs/',
        expname='paper_models',
        act_itr='30000',
        ft_path=ft_path,
        scene_name=scene_name,
        render_scene=None,
        perturb=1.0,
        white_bkgd=False,
        raw_noise_std=0.0,
        dataset_type='Cambridge',
        lindisp=False,
        encode_hist=True,
        no_reload=False,
        no_ndc=True,
        hist_bin=10,
        lrate=5e-4,
    )

class CambridgeLocParams(ParamGroup):
    def __init__(self, parser):
        self.iteration = -1
        self.num_particles = 8
        self.batch_size_arg = 8
        self.num_iterations = 2
        self.pose_estimator = 'dfnet'
        self.scene_name = ''
        self.test_all_cams = True
        self.shuffle = False
        self.best_particle = True
        self.pertub_std_pos = 0.1
        self.pertub_std_rot = 0.01
        self.pertub_std_pos_init = 0.1
        self.pertub_std_rot_init = 0.01
        self.main_output_dir = './outputs'
        self.output_dir = 'final_Camb_8_2'
        self.ft_path = '/path/to/appearance_models/'
        self.original_gs = False
        super().__init__(parser, "Cambridge Localization")
        parser.add_argument("--no_appearance", action="store_true", help="disable appearance color transform")
        parser.add_argument("--test_cams_index", nargs='+', type=int, default=None,
                            help="optional subset of test camera indices")

class SevenScenesLocParams(ParamGroup):
    def __init__(self, parser):
        self.iteration = -1
        self.num_particles = 8
        self.batch_size_arg = 8
        self.num_iterations = 2
        self.pose_estimator = 'dfnet'
        self.scene_name = ''
        self.test_all_cams = True
        self.shuffle = False
        self.best_particle = False
        self.pertub_std_pos = 0.01
        self.pertub_std_rot = 0.01
        self.pertub_std_pos_init = 0.1
        self.pertub_std_rot_init = 0.01
        self.main_output_dir = './outputs'
        self.output_dir = '7scenes_paper_8_2_1000'
        self.original_gs = False
        super().__init__(parser, "7-Scenes Localization")
        parser.add_argument("--test_cams_index", nargs='+', type=int, default=None,
                            help="optional subset of test camera indices")

def build_loc_parser(description, loc_params_cls):
    parser = ArgumentParser(description=description)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    localization = LocalizationParams(parser)
    loc = loc_params_cls(parser)
    return parser, model, pipeline, localization, loc


@dataclass
class LocConfig:
    name: str
    description: str
    loc_params_cls: Type
    dataset_type: str
    coarse_pose_subdir: str
    coarse_pose_template: str
    coarse_pose_loader: str  # 'cambridge' | '7scenes'
    hessian_files: List[str]
    img_reso: Tuple[int, int]
    reprojection_error: float
    use_appearance: bool
    test_source: str  # 'colmap' | '7scenes'
    fixed_img_size: Optional[Tuple[int, int]] = None
    focal_lengths: Optional[Dict[str, float]] = None
    supports_original_gs: bool = False
    hessian_original_gs_file: Optional[str] = None


CAMBRIDGE_CFG = LocConfig(
    name='cambridge',
    description='UGSLoc — Cambridge Landmarks localization',
    loc_params_cls=CambridgeLocParams,
    dataset_type='cambridge',
    coarse_pose_subdir='Cambridge',
    coarse_pose_template='poses_Cambridge_{scene_name}_.txt',
    coarse_pose_loader='cambridge',
    hessian_files=['hessian_color_semantic.pt', 'hessian_color2.pt'],
    hessian_original_gs_file='_hessian_color.pt',
    img_reso=(512, 512),
    reprojection_error=2.5,
    use_appearance=True,
    test_source='colmap',
    supports_original_gs=True,
)

SEVENSCENES_CFG = LocConfig(
    name='7scenes',
    description='UGSLoc — 7-Scenes localization',
    loc_params_cls=SevenScenesLocParams,
    dataset_type='7scenes',
    coarse_pose_subdir='7Scenes_pgt',
    coarse_pose_template='poses_pgt_7scenes_{scene_name}_.txt',
    coarse_pose_loader='7scenes',
    hessian_files=['hessian_color2.pt', 'hessian_color_semantic.pt', 'hessian_color_no_scaling.pt'],
    img_reso=(256, 512),
    reprojection_error=1.0,
    use_appearance=False,
    test_source='7scenes',
    fixed_img_size=(480, 640),
    focal_lengths={
        'chess': 526.22, 'fire': 526.903, 'heads': 527.745, 'office': 525.143,
        'pumpkin': 525.647, 'redkitchen': 525.505, 'stairs': 525.505,
    },
)


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    print(cmdlne_string)
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
