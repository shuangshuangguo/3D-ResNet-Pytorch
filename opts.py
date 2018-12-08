import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', default='input', type=str, help='train list file path')
    parser.add_argument('--val_list', default='', type=str, help='val list file path')
    parser.add_argument('--n_classes', default=101, type=int, help='number of classes')
    parser.add_argument('--model', default='', type=str, help='test Model file path')
    parser.add_argument('--resume', default='', type=str, help='Model file path')
    parser.add_argument('--finetune', type=str, default='')
    parser.add_argument('--result_path', default='output.json', type=str, help='Output file path')
    parser.add_argument('--ft_begin_index', default=4, type=int)
    parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    parser.add_argument('--sample_size', type=int, default=224)
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--n_scales', default=5, type=int)
    parser.add_argument('--scale_step', default=0.84089641525, type=float)
    parser.add_argument('--initial_scale',default=1.0,type=float)

    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--model_name', default='resnet', type=str, help='Currently only support resnet')
    parser.add_argument('--lr',default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, metavar='W')

    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
    parser.add_argument('--epochs', default=80, type=int, metavar='N')
    parser.add_argument('--lr_steps', default=[30, 60], type=float, nargs="+",metavar='LRSteps')

    parser.add_argument('--print-freq', '-p', default=20, type=int, metavar='N')
    parser.add_argument('--eval-freq', '-ef', default=1, type=int, metavar='N')
    parser.add_argument('--snapshot_pref', type=str, default="")
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--no_cuda', action='store_true', default=False, help='If true, cuda is not used.')
    parser.set_defaults(verbose=False)
    parser.add_argument('--verbose', action='store_true', help='')
    parser.set_defaults(verbose=False)

    parser.add_argument('--lr_patience',default=10,type=int)
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    args = parser.parse_args()

    return args
