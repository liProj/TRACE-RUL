# Driver around the authors' Exp class (vendored, unmodified model/data code).
# Mirrors CMAPSS_test.py / FD001_3_train.py argument handling, but lets us pick
# subset, model, seed, train/test and output dir from the command line.
import os, sys, warnings, argparse
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from configs.data_model_configs import update_namespace
from configs.hparams import get_configs


def str2bool(v):
    return v if isinstance(v, bool) else v == 'True'


def build_args(cli):
    # defaults copied from CMAPSS_test.py
    a = argparse.Namespace(
        task='normal', model_name=cli.model, info='rerun', train=cli.train, resume=not cli.train,
        save_test=True, save_path=cli.save_path, resume_path=cli.save_path, train_epochs=cli.epochs,
        seed=cli.seed, loss_type='MSE', data_root=None, dataset_name=cli.dataset,
        Data_id_CMAPSS=cli.subset, Data_id_CMAPSS_test=cli.subset, Data_id_N_CMAPSS=cli.subset,
        input_length=50, validation=0.1, batch_size=128, add_noise=False, noise_level=0.05,
        noise_level_train=0.05, noise_level_test=0.1, ablation_mode=cli.ablation_mode, AATE_dim=10,
        nlayer=1, hop=1, MAXLIFE_CMAPSS=125, normalization_CMAPSS='minmax', s=1, sampling=10,
        change_len=cli.change_len, rate=0.8, d_model=64, d_ff=256, dropout=0.1, patch_size=5, lamda=0.1,
        kernel_size=3, stride=1, is_diff=False, is_minmax=True, n_seg=8, n_stacks=2, n_heads=1,
        factor=3, e_layers=1, d_layers=1, activation='relu', patch_len=8, patch_stride=4,
        moving_avg=[24], embed='timeF', freq='h', DA=False, source_domain='FD001',
        target_domain='FD001', type=2, Classify=False, D1_lr=0.0001)
    a.data_path = '/' + a.dataset_name
    if a.dataset_name == 'CMAPSS':
        a.data_path_CMAPSS = './CMAPSS'
        a.difference_CMAPSS = False
    a.use_gpu = True
    a.gpu = 0
    a.optimizer = 'Adam'
    a.learning_rate_patience = 10
    a.learning_rate_factor = 0.3
    a.early_stop_patience = 10
    hp = get_configs(a.dataset_name, cli.subset)
    a.learning_rate = hp.train_params[a.model_name]['learning_rate']
    update_namespace(a, hp.alg_hparams[a.model_name])
    return a


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='CMAPSS')
    p.add_argument('--subset', default='FD001')
    p.add_argument('--model', default='AIDGN')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--train', type=str2bool, default=False)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--save_path', default='exp0')
    p.add_argument('--ablation_mode', default='linear')
    p.add_argument('--change_len', type=str2bool, default=True)
    cli = p.parse_args()
    args = build_args(cli)
    from Experiment.Experiment import Exp
    exp = Exp(args)
    exp.start()
