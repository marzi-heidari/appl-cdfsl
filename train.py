import os

import numpy as np
import torch
import torch.optim

import configs
from data.datamgr import SimpleDataManager
from datasets import Omniglot_train_fewshot, ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot
from datasets import miniImageNet_few_shot, DTD_few_shot, miniImageNet_train_few_shot
from io_utils import model_dict, parse_args
from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet


def train(base_loader, model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        if params.train_proto_calculator:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)
            print('setting optimizer for prototype network')
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            print('setting optimizer for feature extractor')
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    for epoch in range(start_epoch, stop_epoch):
        if params.train_proto_calculator:
            model.feature.eval()
            for param in model.feature.parameters():
                param.requires_grad = False
            model.prototype_net.train()
        else:
            model.feature.train()
            model.prototype_net.eval()
            for param in model.prototype_net.parameters():
                param.requires_grad = False
        model.train_loop(epoch, base_loader, optimizer, params.finetune)

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save(
                {'epoch': epoch, 'state': model.state_dict(), },
                outfile)

    return model


def main():
    iter_num = 600
    np.random.seed(10)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("working on gpu")
    else:
        device = torch.device("cpu")
        print("working on cpu")
    params = parse_args('train')

    image_size = 48
    optimization = 'Adam'
    if params.method in ['baseline']:
        if params.dataset == "miniImageNet":
            datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size=8)
            base_loader = datamgr.get_data_loader(aug=params.train_aug)
        else:
            raise ValueError('Unknown dataset')
        model = BaselineTrain(model_dict[params.model], params.num_classes)
    elif params.method in ['protonet']:
        n_query = max(1,
                      int(15 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        if params.dataset == "miniImageNet":
            datamgr = miniImageNet_few_shot.SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
            base_loader = datamgr.get_data_loader(aug=params.train_aug)
        else:
            raise ValueError('Unknown dataset')
    if params.method == 'protonet':
        model = ProtoNet(model_dict[params.model], **train_few_shot_params)

    else:
        raise ValueError('Unknown method')

    model = model.cuda()
    save_dir = configs.save_dir
    if params.last_checkpoint is not None:
        if params.from_scratch:
            modelfile = torch.load('./logs/checkpoints/miniImageNet/WideResNet28_10_S2M2_R/470.tar')
            state = modelfile['state']
            model_dict_load = model.feature.state_dict()
            model_dict_load.update(state)
            model.feature.load_state_dict(model_dict_load)
        else:
            modelfile = params.last_checkpoint
            modelfile = torch.load(modelfile)
            state = modelfile['state']
            model.load_state_dict(state)

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    params.checkpoint_dir += '_%s' % params.experiment
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    train(base_loader, model, optimization, start_epoch, stop_epoch, params)


if __name__ == '__main__':
    main()
