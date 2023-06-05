import torch
from scipy import stats
import numpy as np
import model
import data_loader
import time


class BaseIQASolver(object):

    def __init__(self, config, path, train_idx, test_idx):
        self.print = config.printf
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.device = config.device
        self.lr = config.lr
        self.lrratio = config.lrratio

        self.beta = config.beta
        # resnet18 resnet50 densenet121 densenet169 vgg11 vgg16 googlenet
        self.model1 = model.BaseModel(config.name1).to(self.device[0])
        self.model2 = model.BaseModel(config.name2).to(self.device[1])
        self.model3 = model.BaseModel(config.name3).to(self.device[2])
        self.l1_loss1 = torch.nn.L1Loss().to(self.device[0])
        self.l1_loss2 = torch.nn.L1Loss().to(self.device[1])
        self.l1_loss3 = torch.nn.L1Loss().to(self.device[2])

        self.weight_decay = config.weight_decay
        paras1 = [{'params': self.model1.parameters(), "lr": self.lr}]
        self.solver1 = torch.optim.Adam(paras1, weight_decay=self.weight_decay)
        paras2 = [{'params': self.model2.parameters(), "lr": self.lr}]
        self.solver2 = torch.optim.Adam(paras2, weight_decay=self.weight_decay)
        paras3 = [{'params': self.model3.parameters(), "lr": self.lr}]
        self.solver3 = torch.optim.Adam(paras3, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(
            config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True, num_workers=config.num_workers)
        test_loader = data_loader.DataLoader(
            config.dataset, path, test_idx, config.patch_size, config.test_patch_num, batch_size=config.test_batch_size,  istrain=False, num_workers=config.num_workers)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc = [0.0 for _ in range(3)]
        best_plcc = [0.0 for _ in range(3)]
        self.print('Epoch\tTrain_Loss Train_SRCC Test_SRCC Test_PLCC | Epoch\tTrain_Loss Train_SRCC Test_SRCC Test_PLCC | Epoch\tTrain_Loss Train_SRCC Test_SRCC Test_PLCC |')
        for t in range(self.epochs):
            if t == 0:
                start_time = time.time()
            epoch_loss = [[], [], []]
            pred_scores = [[], [], []]
            gt_scores = []
            train_srcc = [None, None,None]

            self.model1.train()
            self.model2.train()
            self.model3.train()

            for img, label in self.train_data:
                gt_scores = gt_scores + label.tolist()
                self.solver1.zero_grad()
                self.solver2.zero_grad()
                self.solver3.zero_grad()

                pred1 = self.model1(img.to(self.device[0])).reshape(-1)
                pred2 = self.model2(img.to(self.device[1])).reshape(-1)
                pred3 = self.model3(img.to(self.device[2])).reshape(-1)

                loss1 = self.l1_loss1(pred1, label.to(self.device[0]).detach())
                loss2 = self.l1_loss2(pred2, label.to(self.device[1]).detach())
                loss3 = self.l1_loss3(pred3, label.to(self.device[2]).detach())

                pred_scores[0] = pred_scores[0] + pred1.cpu().tolist()
                pred_scores[1] = pred_scores[1] + pred2.cpu().tolist()
                pred_scores[2] = pred_scores[2] + pred3.cpu().tolist()

                test_score1 = np.mean(
                    torch.abs(pred1.cpu() - pred2.cpu()).tolist())
                test_score2 = np.mean(
                    torch.abs(pred1.cpu() - pred3.cpu()).tolist())
                test_score3 = np.mean(
                    torch.abs(pred2.cpu() - pred3.cpu()).tolist())

                loss1 = loss1 + (test_score1+test_score2 +
                                 test_score3) * (1-self.beta)
                loss2 = loss2 + (test_score1+test_score2 +
                                 test_score3) * (1-self.beta)
                loss3 = loss3 + (test_score1+test_score2 +
                                 test_score3) * (1-self.beta)

                epoch_loss[0].append(loss1.item())
                epoch_loss[1].append(loss2.item())
                epoch_loss[2].append(loss3.item())

                loss1.backward()
                loss2.backward()
                loss3.backward()

                self.solver1.step()
                self.solver2.step()
                self.solver3.step()

            train_srcc[0], _ = stats.spearmanr(pred_scores[0], gt_scores)
            train_srcc[1], _ = stats.spearmanr(pred_scores[1], gt_scores)
            train_srcc[2], _ = stats.spearmanr(pred_scores[2], gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)

            if t == 0:
                self.print(
                    f"epoch 测试结束用时为：{(time.time()-start_time)/60} minute")

            for i in range(3):
                if test_srcc[i] > best_srcc[i]:
                    best_srcc[i] = test_srcc[i]
                    best_plcc[i] = test_plcc[i]
            for i in range(3):
                self.print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                           (t + 1, sum(epoch_loss[i]) / len(epoch_loss[i]), train_srcc[i], test_srcc[i], test_plcc[i]), end=" | ")
            self.print("")

            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1

            paras1 = [{'params': self.model1.parameters(), "lr": lr}]
            paras2 = [{'params': self.model2.parameters(), "lr": lr}]
            paras3 = [{'params': self.model3.parameters(), "lr": lr}]
            self.solver1 = torch.optim.Adam(
                paras1, weight_decay=self.weight_decay)
            self.solver2 = torch.optim.Adam(
                paras2, weight_decay=self.weight_decay)
            self.solver3 = torch.optim.Adam(
                paras3, weight_decay=self.weight_decay)

        for i in range(3):
            self.print('Best test SRCC %f, PLCC %f' %
                       (best_srcc[i], best_plcc[i]), end=" | ")
        self.print("")

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        pred_scores = [[], [], []]
        test_srcc = [[], [], []]
        test_plcc = [[], [], []]
        gt_scores = []

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()

        with torch.no_grad():
            for img, label in data:
                pred1 = self.model1(img.to(self.device[0]))
                pred2 = self.model2(img.to(self.device[1]))
                pred3 = self.model3(img.to(self.device[2]))

                gt_scores = gt_scores + label.tolist()

                pred_scores[0] += pred1.reshape(-1).cpu().tolist()
                pred_scores[1] += pred2.reshape(-1).cpu().tolist()
                pred_scores[2] += pred3.reshape(-1).cpu().tolist()

        gt_scores = np.mean(np.reshape(np.array(gt_scores),
                            (-1, self.test_patch_num)), axis=1)
        pred_scores[0] = np.mean(np.reshape(
            np.array(pred_scores[0]), (-1, self.test_patch_num)), axis=1)
        pred_scores[1] = np.mean(np.reshape(
            np.array(pred_scores[1]), (-1, self.test_patch_num)), axis=1)
        pred_scores[2] = np.mean(np.reshape(
            np.array(pred_scores[2]), (-1, self.test_patch_num)), axis=1)

        test_srcc[0], _ = stats.spearmanr(pred_scores[0], gt_scores)
        test_plcc[0], _ = stats.pearsonr(pred_scores[0], gt_scores)

        test_srcc[1], _ = stats.spearmanr(pred_scores[1], gt_scores)
        test_plcc[1], _ = stats.pearsonr(pred_scores[1], gt_scores)

        test_srcc[2], _ = stats.spearmanr(pred_scores[2], gt_scores)
        test_plcc[2], _ = stats.pearsonr(pred_scores[2], gt_scores)

        return test_srcc, test_plcc
