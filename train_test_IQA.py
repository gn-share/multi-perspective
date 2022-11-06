import os
import torch
import argparse
import random
import numpy as np
from HyerIQASolver import HyperIQASolver

main_path = os.environ['HOME']


def main(config):

    folder_path = {
        'live': main_path + '/image_data/LIVE/',  #
        'csiq': main_path + '/image_data/CSIQ/',  #
        'tid2013': main_path + '/image_data/tid2013',
        'livec': main_path + '/image_data/ChallengeDB_release/',  #
        'koniq': main_path + '/image_data/koniq/',  #
        'bid': main_path + '/image_data/BID/',  #
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
    }
    sel_num = img_num[config.dataset]

    srcc_all = []
    plcc_all = []

    for i in range(config.num):
        srcc_all.append(np.zeros(config.train_test_num, dtype=np.float64))
        plcc_all.append(np.zeros(config.train_test_num, dtype=np.float64))

    print('Training and testing on %s dataset for %d rounds...' %
          (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i+1))
        random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

        solver = HyperIQASolver(
            config, folder_path[config.dataset], train_index, test_index)
        result = solver.train()
        for j in range(config.num):
            srcc_all[j][i] = result[0][j]
            plcc_all[j][i] = result[1][j]

    srcc_med = []
    plcc_med = []
    for i in range(config.num):
        srcc_med.append(np.median(srcc_all[i]))
        plcc_med.append(np.median(plcc_all[i]))
    for i in range(len(srcc_med)):
        print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' %
              (srcc_med[i], plcc_med[i]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='live',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int,
                        default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int,
                        default=25, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float,
                        default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay',
                        type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int,
                        default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=48, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=16, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int,
                        default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num',
                        type=int, default=10, help='Train-test times')
    parser.add_argument('--beta', dest='beta',
                        type=float, default=0.3, help='beta control')

    config = parser.parse_args()
    config.device = []
    config.num_workers = 3
    cuda_number = [0,]
    config.num = len(cuda_number)
    for i in cuda_number:
        config.device.append(torch.device(f"cuda:{i:d}"))

    main(config)
