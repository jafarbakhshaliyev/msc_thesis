import argparse
import os
import torch
from exp.exp_main import Exp_Main
import ast


import random
import numpy as np

#def setup_seed(seed):
   # torch.manual_seed(seed)
   # torch.cuda.manual_seed_all(seed)
   # torch.cuda.manual_seed(seed)
   # np.random.seed(seed)
   # random.seed(seed)
   # torch.backends.cudnn.deterministic = True
#setup_seed(2021)

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"


### Important parser features:

#'--data_size' - percentage as 0.1

###

parser = argparse.ArgumentParser(description='Data Augmentations for Time Series Forecasting')

# random seed
#parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# MICN

parser.add_argument('--conv_kernel', type=int, nargs='+', default=[17,49], help='downsampling and upsampling convolution kernel_size')
parser.add_argument('--decomp_kernel', type=int, nargs='+', default=[17,49], help='decomposition kernel_size')
parser.add_argument('--isometric_kernel', type=int, nargs='+', default=[17,49], help='isometric convolution kernel_size')
parser.add_argument('--mode', type=str, default='regre', help='different mode of trend prediction block: [regre or mean]')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)


# LightTS
parser.add_argument('--chunk_size', type=int, default=40, help='LightTS')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')


# SCINet
parser.add_argument('--hidden_size', default=1, type=float, help='hidden channel of module')
parser.add_argument('--kernel', default=5, type=int, help='kernel size, 3, 5, 7')
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--levels', type=int, default=3)
parser.add_argument('--stacks', type=int, default=1, help='1 stack or 2 stacks')

parser.add_argument('--top_k', type=int, default=5)

# Formers 
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# CycleNet
parser.add_argument('--cycle', type=int, default=24, help='cycle length')
parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs') #default = 10
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus') #default='0,1,2,3'
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# Augmentation
parser.add_argument('--aug_method', type=str, default='f_mask', help='f_mask: Frequency Masking, f_mix: Frequency Mixing')
parser.add_argument('--aug_rate', type=float, default=0.5, help='mask/mix rate')
parser.add_argument('--in_batch_augmentation', action='store_true', help='Augmentation in Batch (save memory cost)', default=False)
parser.add_argument('--in_dataset_augmentation', action='store_true', help='Augmentation in Dataset', default=False)
parser.add_argument('--closer_data_aug_more', action='store_true', help='Augment times increase for data closer to test set', default=False)
parser.add_argument('--data_size', type=float, default=1, help='size of dataset, i.e, 0.01 represents uses 1 persent samples in the dataset')
parser.add_argument('--aug_data_size', type=int, default=1, help='size of augmented data, i.e, 1 means double the size of dataset')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--wo_original_set', action='store_true', help='without original train set')
parser.add_argument('--test_time_train', type=bool, default=False, help='Affect data division')
parser.add_argument('--wavelet', type=str, default='db2', help='wavelet form for DWT')
parser.add_argument('--level', type=int, default=2, help='level for DWT')
parser.add_argument('--rates', type=str, default="[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]",
                        help='List of float rates as a string, e.g., "[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]"')
parser.add_argument('--sampling_rate', type=float, default=0.5, help='sampling rate for WaveMask and WaveMix')
parser.add_argument('--uniform', action='store_true', help='Uniform rates for wavemask/mix', default=False)
parser.add_argument('--n_patch', type=int, default=4, help='# of patches')
parser.add_argument('--aug_stride', type=int, default=5, help='# of patches stride')
parser.add_argument('--aug_patch_len', type=int, default=5, help='# of patches')
parser.add_argument('--warp_scale', type=float, default=0.2, help='# of pkjatches')
parser.add_argument('--shuffle', action='store_true', help='Uniform rates for wavemask/mix', default=False)
parser.add_argument('--block_size', type=int, default=8, help='block size of MBB')
parser.add_argument('--K_num', type=int, default=1, help='K number of RobustTAD')
parser.add_argument('--seg_ratio', type=float, default=0.2, help='segment ratio of RobustTAD')



parser.add_argument('--use_PEMSmetric', action='store_true', help='use PEMS metric', default=False)
parser.add_argument('--use_former', action='store_true', help='use fomer', default=False)

args = parser.parse_args()
args.rates = ast.literal_eval(args.rates)

#fix_seed = args.seed
#random.seed(fix_seed)
#torch.manual_seed(fix_seed)
#np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False


if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]


print('Args in experiment:')
print(args)

if args.task_name == 'long_term_forecast':
    Exp = Exp_Main
    
#Exp = Exp_Main

if args.is_training:
    mse_avg, mae_avg, rse_avg = np.zeros(args.itr), np.zeros(args.itr), np.zeros(args.itr)
    val_mse_avg = np.zeros(args.itr)
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        _, val_loss = exp.train(setting)
        val_mse_avg[ii] = val_loss

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mse, mae, rse = exp.test(setting)
        mse_avg[ii] = mse
        mae_avg[ii] = mae
        rse_avg[ii] = rse

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()

    
    f = open("result-" + args.des + args.data + ".txt", 'a')
    f.write('\n')
    f.write('\n')
    f.write("-------START FROM HERE-----")
    f.write(args.aug_method +  " " + str(args.pred_len) +"  \n")
    f.write('avg val mse:{}, std val mse:{}, avg mse:{}, avg mae:{} avg rse:{}  std mse:{}, std mae:{} std rse:{}'.format(val_mse_avg.mean(), val_mse_avg.std(), mse_avg.mean(), mae_avg.mean(), rse_avg.mean(), mse_avg.std(), mae_avg.std(), rse_avg.std()))
    f.write('\n')
    f.write('\n')
    f.close()

    
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
