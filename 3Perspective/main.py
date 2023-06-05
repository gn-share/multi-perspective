import os
import torch
import argparse
import random
import numpy as np
from BaseIQASolver import BaseIQASolver

main_path = os.environ['HOME']
# main_path = ".."


def main(config):
    print = config.printf

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
        'koniq': list(range(0, 10073)),
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
        # Randomly select 80% images for training and the rest for testing
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

        solver = BaseIQASolver(
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


def try_gpu(i=0):  # @save
    """如果存在,则返回gpu(i),否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='live',
                        help='Support datasets: livec|koniq|bid|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int,
                        default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int,
                        default=25, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-5, help='Learning rate')
    parser.add_argument('--lrratio', dest='lrratio', type=int,
                        default=10, help='Learning rate ratio')
    parser.add_argument('--weight_decay', dest='weight_decay',
                        type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=48, help='Batch size')
    parser.add_argument('--test_batch_size', dest='test_batch_size',
                        type=int, default=48, help='The test batch size')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=16, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int,
                        default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num',
                        type=int, default=10, help='Train-test times')
    parser.add_argument('--test', dest='test',
                        type=bool, default=False, help='Is not test')
    parser.add_argument('--cuda1', dest='cuda1',
                        type=int, default=0, help='Choose the cuda number')
    parser.add_argument('--name1', dest='name1', type=str)
    parser.add_argument('--cuda2', dest='cuda2',
                        type=int, default=0, help='Choose the cuda number')
    parser.add_argument('--name2', dest='name2', type=str)
    parser.add_argument('--cuda3', dest='cuda3',
                        type=int, default=0, help='Choose the cuda number')
    parser.add_argument('--name3', dest='name3', type=str)
    parser.add_argument('--beta', dest='beta', type=float, default=0.7)
    # resnet18 resnet50 densenet121 densenet169 vgg11 vgg16 googlenet
    num_workers = {
        'live': 3,
        'csiq': 5,
        'tid2013': 5,
        'livec': 3,
        'koniq': 3,
        'bid': 8,
    }

    config = parser.parse_args()
    config.num = 3
    # Choose whether to test model
    if config.test:
        random.seed(1)
        print("Start to test model")
        config.printf = print
        config.train_test_num = 1
        config.epochs = 20

    else:
        result_file = f"./results/{config.dataset}-{config.name1}-{config.name2}-{config.name3}.txt"
        if not os.path.exists("results"):
            os.mkdir("results")

        def printf(s, file=result_file, end="\n"):
            with open(file, "a+") as f:
                print(s, file=f, end=end)
        config.printf = printf

    config.device = [try_gpu(i=config.cuda1), try_gpu(
        i=config.cuda2), try_gpu(i=config.cuda3)]
    config.num_workers = num_workers[config.dataset]
    main(config)
