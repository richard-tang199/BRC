import os
import torch
import numpy as np
import random
import argparse
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='DADA')

parser.add_argument('--random_seed', type=int, default=2024, help='random seed')
parser.add_argument('--model', type=str, default='./DADA', help='model name')
parser.add_argument('--data', type=str, default='MSL', help='dataset type')
parser.add_argument('--root_path', type=str, default='/workspace/dataset/dataset', help='root path of the data file')
parser.add_argument('--batch_size', type=int, default=512, help='batch size of input data')
parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
parser.add_argument('--patch_len', type=int, default=5, help='patch length')
parser.add_argument('--stride', type=int, default=5, help='stride')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--hidden_dim', type=int, default=64, help='for DADA')
parser.add_argument('--depth', type=int, default=10, help='for DADA')
parser.add_argument('--bn_dims', type=int, nargs="+", default=[8, 16, 32, 64, 128, 256], help='for DADA')
parser.add_argument('--k', type=int, default=3, help='for DADA')
parser.add_argument("--mask_mode", type=str, default='c', help="for DADA")
parser.add_argument('--copies', type=int, default=10, help='')
parser.add_argument('--norm', type=int, default=0, help='True 1 False 0')
parser.add_argument('--L', type=float, default=1, help='anoamly score')
parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence')
parser.add_argument('--metric', type=str, nargs="+", default="affiliation", help="metric")
parser.add_argument('--q', type=float, nargs="+", default=[0.03], help="for SPOT")
parser.add_argument('--t', type=float, nargs="+", default=[0.06], help="threshold found by SPOT")
parser.add_argument('--max_iters', type=int, default=100000, help='for DADA')
parser.add_argument("--percentage", type=float, default=1, help="the percentage(*100) of train data")
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--des', type=str, default='zero_shot', help='exp description')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
parser.add_argument("--group", type=str, default="001")
parser.add_argument("--dataset_name", type=str, default="UCR")
parser.add_argument("--model_name", type=str, default="dada")
parser.add_argument("--plot", type=bool, default=False)


class DADA(object):
    def __init__(self):
        args = parser.parse_args()
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        from transformers import AutoModel
        model = AutoModel.from_pretrained("baseline//DADA", trust_remote_code=True)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def zero_shot(self, test_loader):
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        test_labels = []
        test_scores = None
        self.model.eval()
        # cal anomaly_socres
        with torch.no_grad():
            for i, (batch_x,) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                score = self.model.infer(batch_x, norm=self.args.norm)
                score = score.detach().cpu().numpy()
                if test_scores is None:
                    test_scores = score
                else:
                    test_scores = np.concatenate((test_scores, score), axis=0)
        return test_scores

if __name__ == '__main__':
    dada_config = parser.parse_args()
    dada = DADA(args)