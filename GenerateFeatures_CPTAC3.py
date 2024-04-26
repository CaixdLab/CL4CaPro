import torch
import os
import pandas as pd
import torchvision
from networks.resnet_big import activation21mlpl2CLCPnet as CLCPnet
import argparse

class model_config:
    def __init__(self):
        model_config.model_in_dim = 0
        model_config.model_n_hidden_1 = 0
        model_config.model_out_dim = 0
        model_config.feat_dim = 0

def parse_option():
    parser = argparse.ArgumentParser('argument for feature extraction from trained models')

    parser.add_argument('--layer_name', type=str, default='layer1',
                        help='name of the layer to extract features')
    parser.add_argument('--model_in_dim', type=str, default='16008',
                        help='dim input')
    parser.add_argument('--dim_1_list', type=str, default='5196',
                        help='dim 1 test list')
    parser.add_argument('--dim_2_list', type=str, default='4096',
                        help='dim 1 test list')
    parser.add_argument('--dim_3_list', type=str, default='4096',
                        help='dim 1 test list')
    parser.add_argument('--model_save_path', type=str, default='SupCLCP',
                        help='model save path')
    parser.add_argument('--batch_size', type=int, default=110,
                        help='batch size')
    parser.add_argument('--seed', type=int, default=1036,
                        help='random seed')
    parser.add_argument('--learning_rate_list', type=str, default='0.00005',
                        help='test learning rate list')
    parser.add_argument('--l2_rate', type=float, default=0.01,
                        help='l2 normalization rate')
    parser.add_argument('--round', type=int, default=1,
                        help='number of validation round')
    parser.add_argument('--split_class_num', type=int, default=8,
                        help='number of split classes')
    parser.add_argument('--cancer_group', type=str, default='KIRC',
                        help='cancer name')
    parser.add_argument('--cancer_pooled', type=str, default='CCPRC',
                        help='cancer group (SM/NGT/MSE/CCPRC/HC/GG/DS/LC)')
    parser.add_argument('--task', type=str, default='WholeTimeSeq', choices=['WholeTimeSeq', 'Risk'],
                        help='choose task to train the model')
    parser.add_argument('--gpu_device', type=int, default=0,
                        help='gpu device is used to generate the features')

    opt = parser.parse_args()

    iterations = opt.dim_1_list.split(',')
    opt.dim_1_list = list([])
    for it in iterations:
        opt.dim_1_list.append(int(it))

    iterations = opt.dim_2_list.split(',')
    opt.dim_2_list = list([])
    for it in iterations:
        opt.dim_2_list.append(int(it))

    iterations = opt.dim_3_list.split(',')
    opt.dim_3_list = list([])
    for it in iterations:
        opt.dim_3_list.append(int(it))

    iterations = opt.learning_rate_list.split(',')
    opt.learning_rate_list = list([])
    for it in iterations:
        opt.learning_rate_list.append(float(it))

    return opt

# convert feature
def gen_feat(X_get, model):
    X_tensor = torch.FloatTensor(X_get).cuda(non_blocking=True)
    features = model(X_tensor)
    feat_list = features.tolist()
    return  feat_list

def gen_feat_withinfo(X_get, model, info):
    X_tensor = torch.FloatTensor(X_get).cuda(non_blocking=True)
    features = model(X_tensor)
    feat_list = features.tolist()
    if len(info) != len(feat_list):
        print('Wrong info size!')
    merge_info = []
    for i in range(len(info)):
        merge_info.append(info[i] + feat_list[i])
    return  merge_info

def gen_layer_feat(X_get, model):
    X_tensor = torch.FloatTensor(X_get).cuda(non_blocking=True)
    features = model(X_tensor)
    for k, v in features.items():
        list_get = v
    feat_list = list_get.tolist()
    return  feat_list

def gen_layer_feat_withinfo(X_get, model, info):
    X_tensor = torch.FloatTensor(X_get).cuda(non_blocking=True)
    features = model(X_tensor)
    for k, v in features.items():
        list_get = v
    feat_list = list_get.tolist()
    if len(info) != len(feat_list):
        print('Wrong info size!')
    merge_info = []
    for i in range(len(info)):
        merge_info.append(info[i] + feat_list[i])
    return  merge_info

def main():

    opt_get = parse_option()

    torch.cuda.set_device(opt_get.gpu_device)

    if opt_get.task == 'WholeTimeSeq':
        TotalCancerDataPath = './CPTAC3&DKFZ/CancerRNA_CPTAC3_' + opt_get.cancer_group + '_' + opt_get.task + '_3.txt'
    else:
        TotalCancerDataPath = './CPTAC3&DKFZ/CancerRNA_CPTAC3_' + opt_get.cancer_group + '_' + opt_get.task + '_2.txt'
    data_get = pd.read_csv(TotalCancerDataPath)
    X_get = data_get.iloc[:, 6:].values.tolist()
    #Y_get = data_get.iloc[:, 4].values.tolist()
    Info = data_get.iloc[:, :6].values.tolist()

    dim_1_list = opt_get.dim_1_list
    dim_2_list = opt_get.dim_2_list
    dim_3_list = opt_get.dim_3_list
    batch_size = opt_get.batch_size
    seed = opt_get.seed
    opt = model_config()
    opt.model_in_dim = int(opt_get.model_in_dim)
    learning_rate_list = opt_get.learning_rate_list

    # 2 + 1 model
    for i in range(len(dim_1_list)):
        for j in range(len(dim_2_list)):
            for k in range(len(dim_3_list)):
                for lr in learning_rate_list:
                    opt.model_n_hidden_1 = int(dim_1_list[i])
                    opt.model_out_dim = int(dim_2_list[j])
                    opt.feat_dim = int(dim_3_list[k])
                    get_model = CLCPnet(opt).cuda()
                    model_path = opt_get.model_save_path
                    all_files = os.listdir(model_path)
                    pth_files = list(filter(lambda f: f.endswith('.pth'), all_files))

                    model_name = model_path + pth_files[-1]
                    state = torch.load(model_name)

                    get_model.load_state_dict(state['model'])
                    new_model = torchvision.models._utils.IntermediateLayerGetter(get_model,{opt_get.layer_name: 'feat'})

                    merge_get = gen_layer_feat_withinfo(X_get, new_model, Info)

                    header = ['bar', 'PFI', 'PFItime', 'gen_id', 'predicted_label', 'type']
                    for item in range(len(merge_get[0]) - 6):
                        header.append('feature_' + str(item))
                    merge_info = pd.DataFrame(merge_get, columns = header)
                    save_path = 'Features/CPTAC3DKFZFeature_{}.txt'.format(opt_get.cancer_group)
                    print('Generated: CPTAC3DKFZFeature_{}.txt'.format(opt_get.cancer_group))
                    merge_info.to_csv(save_path, index=False)

if __name__ == '__main__':
    main()
