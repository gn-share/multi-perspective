from re import I
import torch
from scipy import stats
import numpy as np
import model_res50
import model_res18
import model_vgg
import model_vgg_livec
import model_go
import model_target

import data_loader
import time


class HyperIQASolver(object):

    def __init__(self, config, path, train_idx, test_idx):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.device = config.device
        self.beta = config.beta

        # 不同模型最后采用的学习率
        self.lr = [config.lr,]

        # model_res50.HyperNet(16, 112, 224, 112, 56, 28,14, 7).to(self.device[0]),         *1
        # model_res18.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(self.device[0]),        *1
        # model_vgg.HyperNet(16, 112, 224, 112, 56, 28,14, 14).to(self.device[0]),          *20 live
        # model_vgg_livec.HyperNet(16, 112, 224, 112, 56, 28,14, 7).to(self.device[0]),     *10 livec
        # model_go.HyperNet(16, 112, 224, 112, 56, 28,14, 7).to(self.device[0]), *12 live   *6 livec
        
        self.model_hyper = [
            model_res50.HyperNet(16, 112, 224, 112, 56, 28,14, 7).to(self.device[0]), 
        ]

        self.model_num = len(self.model_hyper)

        self.l1_loss = []
        for i in range(self.model_num):
            self.l1_loss.append(torch.nn.L1Loss().to(self.device[i]))

        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        self.solver = []
        self.hypernet_params = []

        for i in range(self.model_num):
            backbone_params = list(
                map(id, self.model_hyper[i].res.parameters()))
            self.hypernet_params.append(filter(lambda p: id(
                p) not in backbone_params, self.model_hyper[i].parameters()))
            paras = [{'params': self.hypernet_params[i], 'lr': self.lr[i] * self.lrratio},
                     {'params': self.model_hyper[i].res.parameters(
                     ), 'lr': self.lr[i]}
                     ]
            self.solver.append(torch.optim.Adam(
                paras, weight_decay=self.weight_decay))

        train_loader = data_loader.DataLoader(
            config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True, num_workers=config.num_workers)
        test_loader = data_loader.DataLoader(
            config.dataset, path, test_idx, config.patch_size, config.test_patch_num, istrain=False, num_workers=config.num_workers)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc = [0.0 for _ in range(self.model_num)]
        best_plcc = [0.0 for _ in range(self.model_num)]
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            start_time=None
            if t == 0:
                start_time = time.time()    
            epoch_loss = []
            pred_scores = []
            gt_scores = []
            train_srcc = []
            for i in range(self.model_num):
                epoch_loss.append([])
                pred_scores.append([])
                train_srcc.append(None)
                self.model_hyper[i].train()

            for img, label in self.train_data:
                gt_scores = gt_scores + label.tolist()
                img_gpu = []
                label_gpu = []
                pred = []
                loss = []
                target_net = []
                for i in range(self.model_num):
                    pred.append([])
                    img_gpu.append(img.to(self.device[i]))
                    label_gpu.append(label.to(self.device[i]))
                    self.solver[i].zero_grad()
                paras = []
                # 为了让多个模型可以同时运行
                for i in range(self.model_num):
                    paras.append(self.model_hyper[i](img_gpu[i]))

                for i in range(self.model_num):
                    target_net.append(
                        model_target.TargetNet(paras[i]).to(self.device[i]))
                for i in range(self.model_num):
                    for param in target_net[i].parameters():
                        param.requires_grad = False
                for i in range(self.model_num):
                    pred[i] = target_net[i](paras[i]['target_in_vec'])
                for i in range(self.model_num):
                    try:
                        pred_scores[i] = pred_scores[i] + \
                            pred[i].cpu().tolist()
                    except Exception:
                        pred_scores[i] = pred_scores[i] + \
                            [pred[i].cpu().item()]
                # 通过循环来形成不同模型之间损失的计算，并将他们累加
                diff_score = 0.0

                for d_a in range(self.model_num):
                    for d_b in range(d_a+1, self.model_num):
                        diff_score += np.mean(
                            torch.abs(pred[d_a].cpu() - pred[d_b].cpu()).tolist())

                for i in range(self.model_num):
                    loss.append(self.l1_loss[i](
                        pred[i].squeeze(), label_gpu[i].float().detach()))
                    if self.model_num > 1:
                        loss[i] = (1-self.beta)*loss[i]+self.beta*diff_score
                    epoch_loss[i].append(loss[i].item())
                for i in range(self.model_num):
                    loss[i].backward()
                for i in range(self.model_num):
                    self.solver[i].step()
            # print(f"epoch {t} 训练结束用时为：{time.time()-start_time}")
            for i in range(self.model_num):
                train_srcc[i], _ = stats.spearmanr(pred_scores[i], gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)
            # test_srcc, test_plcc = [[0.0],[0.0]]
            if t == 0:
                print(f"epoch 测试结束用时为：{(time.time()-start_time)/60} minute")
            for i in range(self.model_num):
                if test_srcc[i] > best_srcc[i]:
                    best_srcc[i] = test_srcc[i]
                    best_plcc[i] = test_plcc[i]
            for i in range(self.model_num):
                print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                      (t + 1, sum(epoch_loss[i]) / len(epoch_loss[i]), train_srcc[i], test_srcc[i], test_plcc[i]), end="|")
            print("")
            # Update optimizer
            for i in range(self.model_num):
                lr = self.lr[i] / pow(10, (t // 6))
                if t > 8:
                    self.lrratio = 1
                paras = [{'params': self.hypernet_params[i], 'lr': lr * self.lrratio},
                         {'params': self.model_hyper[i].res.parameters(),
                          'lr': self.lr[i]}
                         ]
                self.solver[i] = torch.optim.Adam(
                    paras, weight_decay=self.weight_decay)

        for i in range(self.model_num):
            print('Best test SRCC %f, PLCC %f' % (best_srcc[i], best_plcc[i]))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        pred_scores = []
        test_srcc = []
        test_plcc = []
        gt_scores = []
        for i in range(self.model_num):
            pred_scores.append([])
            test_srcc.append([])
            test_plcc.append([])
            self.model_hyper[i].eval()
        with torch.no_grad():
            for img, label in data:
                gt_scores = gt_scores + label.tolist()

                img_gpu = []
                label_gpu = []
                paras = []
                target_net = []
                pred = []
                for i in range(self.model_num):
                    pred.append(None)
                    img_gpu.append(img.to(self.device[i]))
                    label_gpu.append(label.to(self.device[i]))

                for i in range(self.model_num):
                    paras.append(self.model_hyper[i](img_gpu[i]))

                for i in range(self.model_num):
                    target_net.append(model_target.TargetNet(
                        paras[i]).to(self.device[i]))
                    target_net[i].eval()
                for i in range(self.model_num):
                    pred[i] = target_net[i](paras[i]['target_in_vec'])
                for i in range(self.model_num):
                    pred_scores[i].append(float(pred[i].item()))

        gt_scores = np.mean(np.reshape(np.array(gt_scores),
                            (-1, self.test_patch_num)), axis=1)

        for i in range(self.model_num):
            pred_scores[i] = np.mean(np.reshape(
                np.array(pred_scores[i]), (-1, self.test_patch_num)), axis=1)

            test_srcc[i], _ = stats.spearmanr(pred_scores[i], gt_scores)
            test_plcc[i], _ = stats.pearsonr(pred_scores[i], gt_scores)

        return test_srcc, test_plcc
