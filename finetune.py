import os

import torch.optim

import configs
from data.datamgr import SetDataManager
from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot
from io_utils import model_dict, parse_args
from methods.protonet import ProtoNet
from utils import *


def standardize_image(images):
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


def finetune(novel_loader, n_query=5, pretrained_dataset='miniImageNet', freeze_backbone=False, n_way=5, n_support=5,
             params=None, model=None):
    iter_num = 600
    save_dir = configs.save_dir

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    params.checkpoint_dir += '_%s' % params.experiment
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    start_epoch = 0
    stop_epoch = 100
    acc_all = []
    modelfile = params.last_checkpoint
    modelfile = torch.load(modelfile)
    state = modelfile['state']
    for i, (x, _) in enumerate(novel_loader):
        model.load_state_dict(state)
        if model.n_shot <= 5:
            model = model.cuda()
        model.n_query = x.size(1) - model.n_shot
        if model.change_way:
            model.n_way = x.size(0)

        x = x.contiguous().view(n_way * (n_support + n_query), *x.size()[2:])
        x = standardize_image(x)

        model.feature.train()
        model.prototype_net.eval()
        for param in model.prototype_net.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam([
            {"params": model.feature.parameters(), "lr": 1e-2}])
        model.p_prime_init = False
        model.alpha = 0.5
        for epoch in range(start_epoch, stop_epoch):
            optimizer.zero_grad()
            loss = model.set_forward_loss(x, finetune=True, shuffle=False)
            loss.backward()
            optimizer.step()
            model.alpha = max(0.1, model.alpha * 0.99)
        model.eval()
        scores, _, _, _ = model.set_forward(x, is_feature=False, shuffle=False)
        pred = scores.squeeze(0).data.cpu().numpy().argmax(axis=1)
        y = np.repeat(range(n_way), n_query)
        acc = np.mean(pred == y) * 100
        acc_all.append(acc)
        print(f'episode{i}: accuracy:{acc}')
        del scores, _
        del x
        acc_all1 = np.asarray(acc_all)
        acc_mean = np.mean(acc_all1)
        acc_std = np.std(acc_all1)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (i + 1, acc_mean, 1.96 * acc_std / np.sqrt(i + 1)))
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))


def main():
    iter_num = 600
    np.random.seed(10)
    params = parse_args('train')
    ##################################################################
    image_size = 84
    n_query = max(1,
                  int(5 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    ##################################################################
    pretrained_dataset = "miniImageNet"
    novel_loaders = []
    if params.dataset == 'ISIC':
        print("Loading ISIC")
        datamgr = ISIC_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=n_query, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=False if params.n_shot > 1 else True)
        novel_loaders.append(novel_loader)
    if params.dataset == 'EuroSAT':
        print("Loading EuroSAT")
        datamgr = EuroSAT_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=n_query,
                                                  **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=False if params.n_shot > 1 else True)
        novel_loaders.append(novel_loader)
    if params.dataset == 'CropDisease':
        print("Loading CropDisease")
        datamgr = CropDisease_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=n_query,
                                                      **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=False if params.n_shot > 1 else True)
        novel_loaders.append(novel_loader)
    if params.dataset == 'ChestX':
        print("Loading ChestX")
        datamgr = Chest_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=n_query, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=False if params.n_shot > 1 else True)
        novel_loaders.append(novel_loader)
    if params.dataset in ['cub', 'cars', 'places', 'plantae']:
        novel_file = os.path.join('filelists/', params.dataset, 'novel.json')
        datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.test_n_way, n_support=params.n_shot,
                                 n_eposide=iter_num)
        novel_loader = datamgr.get_data_loader(novel_file, aug=False if params.n_shot > 1 else True)
        novel_loaders.append(novel_loader)
    #########################################################################
    for idx, novel_loader in enumerate(novel_loaders):
        model = ProtoNet(model_dict[params.model], **few_shot_params)
        # replace finetine() with your own method
        finetune(novel_loader, n_query=n_query, pretrained_dataset=pretrained_dataset,
                 params=params, model=model,
                 **few_shot_params)


if __name__ == '__main__':
    main()
