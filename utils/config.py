import yaml
import argparse
from types import SimpleNamespace

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def config_to_namespace(config):
    """Convert dictionary config to SimpleNamespace for easier access"""
    return SimpleNamespace(**config)

def update_args_with_config(args, config):
    """Update argparse args with config values"""
                    
    if 'dataset' in config:
        args.dataset = config['dataset'].get('name', args.dataset)
        args.stereo = config['dataset'].get('stereo', args.stereo)
        args.preprocessed = config['dataset'].get('preprocessed', args.preprocessed)
                                    
        if 'augmentation' in config['dataset']:
            args.augmentation = config['dataset']['augmentation']
        args.clamp_gt = config['dataset'].get('clamp_gt', getattr(args, 'clamp_gt', False))
        args.crop_to_road = config['dataset'].get('crop_to_road', getattr(args, 'crop_to_road', False))
        args.y_range = config['dataset'].get('y_range', getattr(args, 'y_range', None))
        args.num_grids_y = config['dataset'].get('num_grids_y', getattr(args, 'num_grids_y', None))

                  
    if 'model' in config:
        args.backbone = config['model'].get('backbone', args.backbone)
        args.regression = config['model'].get('regression', args.regression)
        args.normalize = config['model'].get('normalize', args.normalize)
        args.pred_head_dim = config['model'].get('pred_head_dim', args.pred_head_dim)
        args.cla_res = config['model'].get('cla_res', args.cla_res)
        args.dino = config['model'].get('dino_size', args.dino)
        args.train_encoder = config['model'].get('train_encoder', getattr(args, 'train_encoder', False))
        if 'dinov2_layers' in config['model']:
            args.dinov2_layers = list(config['model']['dinov2_layers'])
        else:
            args.dinov2_layers = getattr(args, 'dinov2_layers', [5, 7, 9, 11])
        args.upsampler_kind = config['model'].get('upsampler_kind', getattr(args, 'upsampler_kind', 'patch2feature'))

                     
    if 'training' in config:
        args.batch_size = config['training'].get('batch_size', args.batch_size)
        print(f"Batch size from config: {args.batch_size}")
        args.epochs = config['training'].get('epochs', args.epochs)
        args.seed = config['training'].get('seed', args.seed)

                      
    if 'optimizer' in config:
        args.lr = float(config['optimizer'].get('lr', args.lr))
        args.lr_encoder = float(config['optimizer'].get('lr_encoder', getattr(args, 'lr_encoder', 1e-5)))

                 
    if 'loss' in config:
        args.loss = config['loss'].get('type', args.loss)
        args.pixel_type = config['loss'].get('pixel_type', getattr(args, 'pixel_type', 'MSE'))
        args.gradient_weight = float(config['loss'].get('gradient_weight', getattr(args, 'gradient_weight', 0.01)))
        cw = config['loss'].get('composite_weights', {}) or {}
        args.w_pixel = float(cw.get('pixel', getattr(args, 'w_pixel', 0.3)))
        args.w_gradient = float(cw.get('gradient', getattr(args, 'w_gradient', 1.0)))
        args.w_structure = float(cw.get('structure', getattr(args, 'w_structure', 0.0)))
        args.w_normal = float(cw.get('normal', getattr(args, 'w_normal', 1.0)))
        args.w_smoothness = float(cw.get('smoothness', getattr(args, 'w_smoothness', 0.0)))

                      
    if 'scheduler' in config:
        args.scheduler = config['scheduler'].get('type', args.scheduler)

                    
    if 'logging' in config:
        args.logdir = config['logging'].get('logdir', args.logdir)
        args.summary_freq = config['logging'].get('summary_freq', args.summary_freq)

                  
    if 'wandb' in config:
        args.name_run = config['wandb'].get('name_run', args.name_run)
        args.notes = config['wandb'].get('notes', args.notes)

                       
    if 'checkpoint' in config:
        args.loadckpt = config['checkpoint'].get('load_path', args.loadckpt)

    return args

def create_parser():
    """Create argument parser with config file option"""
    parser = argparse.ArgumentParser(description='RoadBEV: Road Surface Reconstruction in Bird\'s Eye View')

                              
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration YAML file')

                                    
    parser.add_argument('--dataset', help='dataset to use: add it to wandb runs')
    parser.add_argument('--stereo', action='store_true', help='if yes, use RoadBEV-stereo; otherwise, RoadBEV-mono')
    parser.add_argument('--cla_res', type=float, default=0.5, help='class resolution for elevation classification')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='maximum learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--logdir', default='/data/rhf/checkpoints/', help='the directory to save logs and checkpoints')
    parser.add_argument('--loadckpt', default=None, help='load the weights from a specific checkpoint')
    parser.add_argument('--summary_freq', type=int, default=10, help='summary_freq')
    parser.add_argument('--seed', type=int, default=307, metavar='S', help='random seed')
    parser.add_argument('--regression', action='store_true', help='regression or classification')
    parser.add_argument('--backbone', default='efficientnet', help='Use DepthAnything3 backbone or EfficientNet')
    parser.add_argument('--gradient_weight', type=float, default=0.01, help='weight for gradient loss in regression')
    parser.add_argument('--notes', type=str, default='', help='notes for wandb run')
    parser.add_argument('--scheduler', type=str, default='onecycle', help='type of lr scheduler to use: onecycle or reduceonplateau')
    parser.add_argument('--loss', type=str, default='L1', help='type of loss to use if regression: L1, gaussian NLL')
    parser.add_argument('--normalize', action='store_true', help='disable normalization')
    parser.add_argument('--name_run', type=str, default=' ', help='give the name of the wandb run')
    parser.add_argument('--pred_head_dim', type=int, default=128, help='define the bottleneck between the transformer encoder and the CNN prediction head')
    parser.add_argument('--preprocessed', action='store_true', help='if yes, the dataloader will load preprocessed data')
    parser.add_argument('--augmentation', action='store_true', help='enable data augmentation for training')
    parser.add_argument('--load_pt', default=None, help='load weights, optimizer, start_idx to resume run')
    parser.add_argument('--dino', default="small", help='ViT encoder size')
    parser.add_argument('--lr_encoder', type=float, default=1e-5, help='peak LR for the encoder param group (only relevant when --train_encoder is set; encoder is frozen otherwise).')
    parser.add_argument('--clamp_gt', action='store_true', help='clamp GT elevation values to [-y_range*100, y_range*100] cm in the dataloader')
    parser.add_argument('--crop_to_road', action='store_true', help='crop each image to the projected voxel ROI bbox (+10%% padding) and resize to 560x560')
    parser.add_argument('--train_encoder', action='store_true', help='unfreeze the backbone so its weights are updated during training')
    parser.add_argument('--w_pixel', type=float, default=0.3, help='composite loss: pixel term weight')
    parser.add_argument('--w_gradient', type=float, default=1.0, help='composite loss: gradient term weight')
    parser.add_argument('--w_structure', type=float, default=0.0, help='composite loss: SSIM-like structural term weight')
    parser.add_argument('--w_normal', type=float, default=1.0, help='composite loss: surface-normal term weight')
    parser.add_argument('--w_smoothness', type=float, default=0.0, help='composite loss: edge-aware smoothness term weight')
    parser.add_argument('--dinov2_layers', type=int, nargs='+', default=[5, 7, 9, 11], help='which DINOv2 transformer block indices to extract intermediate features from')
    parser.add_argument('--upsampler_kind', type=str, default='patch2feature', choices=['patch2feature', 'dino'], help='upsampler after the DINOv2 encoder: patch2feature or dino')
    parser.add_argument('--pixel_type', type=str, default='MSE', choices=['L1', 'MSE'], help='pixel-term loss inside CompositeLoss: L1 or MSE.')
    parser.add_argument('--y_range', type=float, default=None, help='vertical half-range in meters; voxel ROI spans [-y_range, +y_range] around base_height. If None, dataset default (0.2) is used.')
    parser.add_argument('--num_grids_y', type=int, default=None, help='if set, pin the number of vertical voxel bins; grid_res_y is derived as 2*y_range/num_grids_y so the head architecture stays fixed regardless of y_range. If None, num_grids_y is derived from y_range/grid_res_y as before.')

    return parser

def parse_args_with_config():
    """Parse arguments and load config file"""
    parser = create_parser()
    args = parser.parse_args()

                      
    if args.config:
        config = load_config(args.config)
        args = update_args_with_config(args, config)
                                    
        args.config_dict = config

    return args