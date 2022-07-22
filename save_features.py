import os

import h5py
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from torch.autograd import Variable

import configs
from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file


def save_features(model, data_loader, outfile, n_shots):
    f = h5py.File(outfile, 'w')
    print(outfile)
    # if n_shots > 1:
    max_count = len(data_loader) * data_loader.batch_size
    # else:
    #     max_count = len(data_loader) * data_loader.batch_size * 5

    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0

    for i, (x, y) in enumerate(data_loader):
        if i % 10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats, _ = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
        all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        if n_shots > 1:
            all_labels[count:count + feats.size(0)] = y.cpu().numpy()
        else:
            all_labels[count:count + feats.size(0)] = y.repeat(5).cpu().numpy()
        count = count + feats.size(0)
        del x, x_var

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()


if __name__ == '__main__':
    params = parse_args('save_features')

    image_size = 84

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.train_dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'

    if not params.method in ['baseline']:
        checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    checkpoint_dir += f'_{params.experiment}'

    print('params.save_iter: ' + str(params.save_iter))

    # params.save_iter = 399
    if params.save_iter != -1:
        modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
    elif params.method in ['baseline']:
        modelfile = get_resume_file(checkpoint_dir)
    else:
        modelfile = get_best_file(checkpoint_dir)

    if params.save_iter != -1:
        outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"),
                               params.dataset + "_" + str(params.save_iter) + ".hdf5")
    else:
        outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"), params.dataset + ".hdf5")
    batch_size = 16
    if params.dataset in ["ISIC"]:
        datamgr = ISIC_few_shot.SimpleDataManager(image_size, batch_size=batch_size)
        data_loader = datamgr.get_data_loader(aug=False)
    elif params.dataset in ["EuroSAT"]:
        datamgr = EuroSAT_few_shot.SimpleDataManager(image_size, batch_size=batch_size)
        data_loader = datamgr.get_data_loader(aug=False if params.n_shot > 1 else True)
    elif params.dataset in ["CropDisease"]:
        datamgr = CropDisease_few_shot.SimpleDataManager(image_size, batch_size=batch_size)
        data_loader = datamgr.get_data_loader(aug=False)
    elif params.dataset in ["ChestX"]:
        datamgr = Chest_few_shot.SimpleDataManager(image_size, batch_size=batch_size)
        data_loader = datamgr.get_data_loader(aug=False)

    model = model_dict[params.model]()
    print(checkpoint_dir)
    model = model.cuda()
    print(modelfile)
    tmp = torch.load(modelfile)
    state = tmp['state']
    print(tmp.keys())
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.",
                                 "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
    model.load_state_dict(state)
    model.eval()
    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(model, data_loader, outfile, params.n_shot)


def tsne(prototypes_neural, prototypes_avg, features, n_ways, n_shot, feature_size):
    cmap_bold = ListedColormap(["darkorange", "magenta", "darkblue", "darkred", "darkgreen", "darkgray"])
    # alg = TSNE()
    alg = PCA(n_components=2)
    x_embedded = alg.fit_transform(features.reshape([n_ways * n_shot, feature_size]))
    p_avg = alg.fit_transform(prototypes_avg)
    p_neural = alg.fit_transform(prototypes_neural)
    plt.figure()
    plt.scatter(p_avg[:, 0], p_avg[:, 1], c=list(range(n_ways)), cmap=cmap_bold, edgecolor="k", s=100)
    plt.scatter(p_neural[:, 0], p_neural[:, 1], c=list(range(n_ways)), cmap=cmap_bold, edgecolor="k", s=150)
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=[i for i in range(n_ways) for _ in range(n_shot)], cmap=cmap_bold,
                edgecolor="k", s=20)
    plt.axis("tight")
    plt.show()
