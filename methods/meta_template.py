import os
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from data.datamgr import TransformLoader


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_shot = n_support
        self.n_query = -1  # (change depends on input)
        self.feature = model_func()
        self.transform = TransformLoader(224)
        self.feat_dim = self.feature.final_feat_dim
        self.change_way = change_way  # some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self, x, x2, x3, x4, x5, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x, finetune=False):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def centerDatas(self, datas):
        d = torch.zeros_like(datas)
        d[:, :self.n_shot] = datas[:, :self.n_shot, :] - datas[:, :self.n_shot].mean(1, keepdim=True)
        d[:, :self.n_shot] = datas[:, :self.n_shot, :] / torch.norm(datas[:, :self.n_shot, :], 2, 2)[:, :,
                                                         None]
        d[:, self.n_shot:] = datas[:, self.n_shot:, :] - datas[:, self.n_shot:].mean(1, keepdim=True)
        d[:, self.n_shot:] = datas[:, self.n_shot:, :] / torch.norm(datas[:, self.n_shot:, :], 2, 2)[:, :,
                                                         None]
        return d

    def QRreduction(self, datas):
        ndatas = torch.linalg.qr(datas.permute(0, 2, 1)).R
        ndatas = ndatas.permute(0, 2, 1)
        return ndatas

    def scaleEachUnitaryDatas(self, datas):
        norms = datas.norm(dim=2, keepdim=True)
        l = datas / norms
        return l

    def parse_feature(self, x, is_feature):
        x = Variable(x)
        n_shot_prime = self.n_shot * 5
        if is_feature:
            z_all = x.cuda()
            if self.n_shot > 1:
                z_query = z_all[:, self.n_shot:]
                z_support = z_all[:, :self.n_shot]
            else:
                z_query = z_all[:, n_shot_prime:]
                z_query = z_query.view(self.n_way, self.n_query, 5, -1)
                z_query = z_query[:, torch.arange(self.n_query), 0, :]  # choose the original samples as the query sets
                z_support = z_all[:, :n_shot_prime]
        else:
            if self.n_shot == 1:
                x_query = x[self.n_way * n_shot_prime:]
                x_query = x_query.view(self.n_way, self.n_query, 5, 3, 84, 84)
                x_query = x_query[:, torch.arange(self.n_query), 0]  # choose the original samples as the query sets
                x_support = x[:n_shot_prime * self.n_way]
                z_query, _ = self.feature.forward(x_query.view(self.n_way * self.n_query, 3, 84, 84).cuda(0))
                z_support, _ = self.feature.forward(x_support.cuda(0))
            else:
                # x = x.contiguous().view(self.n_way * (self.n_shot + self.n_query), *x.size()[2:])
                z_all, _ = self.feature.forward(x.cuda(0))
                z_all = z_all.view(self.n_way, self.n_shot + self.n_query, -1)
                z_query = z_all[:, self.n_shot:]
                z_support = z_all[:, :self.n_shot]

        return z_support, z_query

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def standardize_image(self, images):
        num_dims = images.dim()
        if not (num_dims == 4 or num_dims == 3):
            raise ValueError("The input tensor must have 3 or 4 dimnsions.")
        if num_dims == 3:
            images = images.unsqueeze(dim=0)
        batch_size, channels, height, width = images.size()
        images_flat = images.view(batch_size, -1)
        mean_values = images_flat.mean(dim=1, keepdim=True)
        std_values = images_flat.std(dim=1, keepdim=True) + 1e-5
        images_flat = (images_flat - mean_values) / std_values
        images = images_flat.view(batch_size, channels, height, width)
        if num_dims == 3:
            assert images.size(0) == 1
            images = images.squeeze(dim=0)
        return images

    def train_loop(self, epoch, train_loader, optimizer, finetune=False):
        print_freq = 10

        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_shot
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            x = x.contiguous().view(self.n_way * (self.n_shot + self.n_query), *x.size()[2:])
            loss = self.set_forward_loss(x, finetune)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))

    def test_loop(self, test_loader, record=None):
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_shot
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean

    def set_forward_adaptation(self, x,
                               is_feature=True,
                               checkpoint=''):  # further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_shot, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_shot))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=1e-4, momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)
        delta_opt = torch.optim.SGD(self.feature.parameters(), lr=1e-4)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        batch_size = 4
        support_size = self.n_way * self.n_shot
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                delta_opt.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()
                delta_opt.step()

        scores = linear_clf(z_query)
        outfile = os.path.join(f'{checkpoint}', '{:d}.tar'.format(450))
        torch.save(
            {'epoch': 450, 'state': self.state_dict(), },
            outfile)
        return scores
