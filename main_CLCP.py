from __future__ import print_function

import os
import sys
import argparse
import time
import csv

import numpy as np
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from CLCPnet import activation21mlpl2CLCPnet as CLCPnet
from losses import SupCLCPLoss
from sklearn.model_selection import train_test_split

# Optional Apex support for mixed precision training
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    """
        Parse input arguments for configuration such as training epochs, batch size, learning rate, etc.
        Also, it sets up model save paths and model names based on the input arguments.
    """
    # basic training setting
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=2000,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of training epochs')
    parser.add_argument('--round', type=int, default=1,
                        help='number of validation round')
    parser.add_argument('--os', type=str, default='W',
                        help='experimental OS system [Windows(W) or Linux(L)]')
    parser.add_argument('--gpu_device', type=int, default=0,
                        help='gpu device is used to train the model')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='500,1000,1500,5000',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--l2_rate', type=float, default=0.01,
                        help='l2 normalization rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--lr_early_stop', type=float, default=0,
                        help='learning rate early stop point')
    parser.add_argument('--epoch_early_stop', type=float, default=0,
                        help='epoch early stop point')
    parser.add_argument('--split_class_num', type=int, default=8,
                        help='number of split classes')

    # model dataset
    parser.add_argument('--model', type=str, default='CLCP')
    parser.add_argument('--model_in_dim', type=int, default=16008)
    parser.add_argument('--model_n_hidden_1', type=int, default=8196)
    parser.add_argument('--model_n_hidden_2', type=int, default=4096)
    parser.add_argument('--model_out_dim', type=int, default=2048)
    parser.add_argument('--feat_dim', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='path', choices=['CancerRNA', 'path'],
                        help='dataset')
    parser.add_argument('--data_folder', type=str, default=None,
                        help='path to custom dataset')
    parser.add_argument('--cancer_group', type=str, default='SM',
                        help='cancer group (SM/NGT/MSE/CCPRC/HC/GG/DS/LC)')
    parser.add_argument('--validation', type=str, default='TCGA',
                        help='validation dataset using')

    # method
    parser.add_argument('--method', type=str, default='CLCP', choices=['SupCon', 'CLCP', 'Test', 'CLCP'],
                        help='choose method')
    parser.add_argument('--task', type=str, default='WholeTimeSeq', choices=['WholeTimeSeq', 'Risk'],
                        help='choose task to train the model')
    parser.add_argument('--train_test', type=int, default=100,
                        help='train test split percentage')
    parser.add_argument('--split_seed', type=int, default=0,
                        help='train test split random seed')
    parser.add_argument('--pooled', type=bool, default=False,
                        help='whether pooled the cancer group')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'

    opt.model_path = './save/SupCLCP_{}_{}/{}_models'.format(opt.cancer_group, opt.split_seed, opt.dataset)
    opt.tb_path = './save/SupCLCP_{}_{}/{}_tensorboard'.format(opt.cancer_group, opt.split_seed, opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # choose saving setting for different model structures
    '''
    # 3+1model
    opt.model_name = '{}_{}_{}_{}_{}_feat_{}_lr_{}_round_{}_decay_{}_bsz_{}_temp_{}'.\
        format(opt.method, opt.model_in_dim,
               opt.model_n_hidden_1, opt.model_n_hidden_2, opt.model_out_dim,
               opt.feat_dim, opt.learning_rate, opt.round,
               opt.weight_decay, opt.batch_size, opt.temp)
    '''
    # 2+1model with seed fine-tune
    opt.model_name = '{}_{}_{}_{}_feat_{}_lr_{}_l2_{}_cl_{}_seed_{}_round_{}_decay_{}_bsz_{}_temp_{}'. \
        format(opt.method, opt.model_in_dim,
               opt.model_n_hidden_1, opt.model_out_dim, opt.feat_dim,
               opt.learning_rate, opt.l2_rate, opt.split_class_num, opt.split_seed, opt.round,
               opt.weight_decay, opt.batch_size, opt.temp)
    '''
    # 1+1model
    opt.model_name = '{}_{}_{}_feat_{}_lr_{}_round_{}_decay_{}_bsz_{}_temp_{}'. \
        format(opt.method, opt.model_in_dim,
               opt.model_out_dim,
               opt.feat_dim, opt.learning_rate, opt.round,
               opt.weight_decay, opt.batch_size, opt.temp)
    '''
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    opt.warmup_to = opt.learning_rate

    if opt.os == 'W':
        opt.tb_folder = opt.tb_path + '/' + opt.model_name + '/'
    else:
        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    """
        Construct and return the data loader for the dataset specified in the options.
    """
    # construct data loader
    if opt.dataset == 'CancerRNA':
        if opt.cancer_group == 'BRCA' and opt.validation == 'Affy':
            """
                For Affy BRCA independent dataset validation
                Choose Full genes or 16 genes (Oncotype) datasets
            """
            #TotalCancerDataPath = './DataSet/CancerRNA_AffyBRCA_' + opt.task + '_' + str(
            #    opt.split_class_num) + '_DMFS.txt'
            TotalCancerDataPath = './DataSet/CancerRNA_AffyBRCAOncotypeDX_' + opt.task + '_' + str(
                opt.split_class_num) + '_DMFS.txt'
        else:
            TotalCancerDataPath = './DataSet/CancerRNA_' + opt.cancer_group + '_' + opt.task + '_' + str(opt.split_class_num) +'.txt'  # Normal TCGA cancer datasets
        data_get = pd.read_csv(TotalCancerDataPath)

        x_train_get, y_train_get, x_val_get, y_val_get = random_split(opt.cancer_group, data_get, opt.train_test, opt.split_seed, opt.task, val = True)
        print('Split dataset with seed ', opt.split_seed, x_train_get.shape, y_train_get.shape)
        x_train = torch.from_numpy(x_train_get)
        x_train = x_train.float()
        y_train = torch.from_numpy(y_train_get)
        y_train = y_train.long()
        x_val = torch.from_numpy(x_val_get)
        x_val = x_val.float()
        y_val = torch.from_numpy(y_val_get)
        y_val = y_val.long()
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=True)

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)

    return train_loader, val_loader

def divide_number(n, bing=10):
    """
        Divide the sample into as many bins as needed and extract the sample evenly
    """
    if n < 20:
        return [n]
    else:
        if bing > 10:
            remainder = n % bing
            if n // bing > remainder:
                result = [bing] * (n // bing)
                for j in range(remainder):
                    result[-(j + 1)] += 1
                return result
            else:
                try:
                    min_remainder = 15
                    result = []
                    for k in range(bing, 16):
                        remainder = n % k
                        if remainder < min_remainder:
                            min_remainder = remainder
                            result = [k] * (n // k)
                    for j in range(min_remainder):
                        result[-(j + 1)] += 1
                    return result
                except:
                    return divide_number(n, bing - 1)
        else:
            min_remainder = 15
            result = []
            for k in range(10, 16):
                remainder = n % k
                if remainder < min_remainder:
                    min_remainder = remainder
                    result = [k] * (n // k)
            for j in range(min_remainder):
                result[-(j + 1)] += 1
            return result

def random_split(cancer_group, data, percent, seed, task, val = True):
    """
        Embedded pooled TCGA cancer datasets setting
    """
    if cancer_group == 'SM':
        CancerTypeList = ['BLCA', 'CESC', 'ESCA', 'HNSC', 'LUSC']
    elif cancer_group == 'NGT':
        CancerTypeList = ['GBM', 'LGG', 'PCPG']
    elif cancer_group == 'MSE':
        CancerTypeList = ['SKCM', 'UVM']
    elif cancer_group == 'CCPRC':
        CancerTypeList = ['KIRC', 'KIRP']
    elif cancer_group == 'HC':
        CancerTypeList = ['CHOL', 'LIHC']
    elif cancer_group == 'GG':
        CancerTypeList = ['COAD', 'READ', 'ESCA', 'STAD']
    elif cancer_group == 'DS':
        CancerTypeList = ['PAAD', 'STAD']
    elif cancer_group == 'LC':
        CancerTypeList = ['LUAD', 'LUSC']
    else:
        CancerTypeList = [cancer_group]

    first_flag = 0
    for eachCancer in CancerTypeList:
        if first_flag == 0:
            first_flag = 1
        get_data = data.loc[data['gen_id'] == eachCancer]
        get_data = get_data.sort_values(by=['PFItime'])

        if task == 'WholeTimeSeq':
            data_len = get_data.shape[0]
            start = 0
            if len(get_data) < 80:
                piece_num = 5
            else:
                piece_num = 20

            end = start + int(data_len / piece_num)
            x_train_get = []
            y_train_get = []
            x_val_get = []
            y_val_get = []

            for i in range(piece_num):
                if i == piece_num - 1:
                    end = data_len
                data_sample = get_data.iloc[start:end]
                start = end
                end = end + int(data_len / piece_num)
                labely = data_sample.iloc[:, 4]
                labelx = data_sample.iloc[:, 6:]
                x_train, x_val, y_train, y_val = train_test_split(labelx, labely, test_size=(1 - percent / 100),
                                                                  random_state=seed)
                if val:
                    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.125)
                x_train_get.append(x_train)
                y_train_get.append(y_train)
                x_val_get.append(x_val)
                y_val_get.append(y_val)

        elif task == 'Risk':
            get_data_0 = get_data[get_data['PFItime'] < (3 * 365)]
            get_data_1 = get_data[get_data['PFItime'] >= (3 * 365)]
            data_len_0 = get_data_0.shape[0]
            data_len_1 = get_data_1.shape[0]
            if data_len_0 < data_len_1:
                get_data_0_split = divide_number(data_len_0)
                get_data_1_split = divide_number(data_len_1, get_data_0_split[0])
            else:
                get_data_1_split = divide_number(data_len_1)
                get_data_0_split = divide_number(data_len_0, get_data_1_split[0])
            print(data_len_0, get_data_0_split, data_len_1, get_data_1_split)

            x_train_get = []
            y_train_get = []
            x_val_get = []
            y_val_get = []

            start = 0
            for split_get in get_data_0_split:
                end = start + split_get
                # print(start, end)
                data_sample = get_data.iloc[start:end]
                start = end
                labely = data_sample['predicted_label']
                labelx = data_sample.iloc[:, 6:]
                x_train, x_test, y_train, y_test = train_test_split(labelx, labely, test_size=(1 - percent / 100),
                                                                    random_state=seed)
                if val:
                    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.125)
                x_train_get.append(x_train)
                y_train_get.append(y_train)
                x_val_get.append(x_test)
                y_val_get.append(y_test)

            for split_get in get_data_1_split:
                end = start + split_get
                # print(start, end)
                data_sample = get_data.iloc[start:end]
                start = end
                labely = data_sample['predicted_label']
                labelx = data_sample.iloc[:, 6:]
                x_train, x_test, y_train, y_test = train_test_split(labelx, labely, test_size=(1 - percent / 100),
                                                                    random_state=seed)
                if val:
                    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.125)
                x_train_get.append(x_train)
                y_train_get.append(y_train)
                x_val_get.append(x_test)
                y_val_get.append(y_test)
        else:
            raise TypeError("No corresponded task given")

        x_train_combine = pd.concat(x_train_get)
        y_train_combine = pd.concat(y_train_get)

        x_val_combine = pd.concat(x_val_get)
        y_val_combine = pd.concat(y_val_get)

        if first_flag == 1:
            total_x = x_train_combine.values
            total_y = y_train_combine.values
            total_val_x = x_val_combine.values
            total_val_y = y_val_combine.values
            first_flag = 2
        else:
            total_x = np.concatenate((total_x, x_train_combine.values), axis=0)
            total_y = np.concatenate((total_y, y_train_combine.values), axis=0)
            total_val_x = np.concatenate((total_val_x, x_val_combine.values), axis=0)
            total_val_y = np.concatenate((total_val_y, y_val_combine.values), axis=0)
    return total_x, total_y, total_val_x, total_val_y

def set_model(opt):
    """
        Initialize and return the model and criterion based on the configuration.
    """
    model = CLCPnet(opt)
    criterion = SupCLCPLoss(temperature=opt.temp)

    # Enable synchronized Batch Normalization if specified
    if opt.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Move model to GPU if available
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            torch.cuda.set_device(opt.gpu_device)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt, val_loader):
    """
        Conducts a single epoch of training and validation.
    """
    model.train()

    # Initialize meters to track batch time, data loading time, and losses
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (RNASeq, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # Move data to GPU if available
        if torch.cuda.is_available():
            RNASeqT = RNASeq.cuda(non_blocking=True)
            labelsT = labels.cuda(non_blocking=True)
        bsz = labelsT.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(RNASeqT)

        if opt.method == 'CLCP':
            loss = criterion(features, labelsT)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        '''
        # Compute L1 loss component
        l1_weight = 1.0
        l1_parameters = []
        for parameter in model.parameters():
            l1_parameters.append(parameter.view(-1))
        l1 = l1_weight * model.compute_l1_loss(torch.cat(l1_parameters))

        # Add L1 loss component
        loss += l1
        '''
        # Compute l2 loss component
        l2_weight = opt.l2_rate
        l2_parameters = []
        for parameter in model.parameters():
            l2_parameters.append(parameter.view(-1))
        l2 = l2_weight * model.compute_l2_loss(torch.cat(l2_parameters))

        # Add l2 loss component
        loss += l2

        # update metric
        if not pd.isna(loss.item()):
            losses.update(loss.item(), bsz)

            # SGD to CLCP
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    val_losses = AverageMeter()
    for idx, (RNASeq_val, labels_val) in enumerate(val_loader):
        if torch.cuda.is_available():
            RNASeq_valT = RNASeq_val.cuda(non_blocking=True)
            labels_valT = labels_val.cuda(non_blocking=True)
        bsz = labels_valT.shape[0]
        val_features = model(RNASeq_valT)
        val_loss = criterion(val_features, labels_valT)
        if not pd.isna(val_loss.item()):
            val_losses.update(val_loss.item(), bsz)

    # print info
    if epoch % opt.print_freq == 0:
        print('Train: [{0}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'val_loss {val_loss.avg:.3f}\t'.format(
                epoch, batch_time=batch_time,
                data_time=data_time, loss=losses, val_loss=val_losses))
        sys.stdout.flush()

    return losses, val_losses


def main():
    """
        Main function to set up the environment, prepare data loaders, models, optimizers, and run the training routine.
    """
    opt = parse_option()

    # clean memory
    torch.cuda.empty_cache()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = SummaryWriter(log_dir=opt.tb_path, flush_secs=2) # lunix opt.tb_folder

    # print model.name
    print('Start:', opt.model_name)

    # training routine
    start_loss_val = 0
    start_loss_avg = 0
    end_loss_val = 0
    end_loss_avg = 0
    pre_loss_val = 0
    pre_loss_avg = 0
    tol = 1500
    tol_count = 0
    min_val_loss = 100
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, val_loss = train(train_loader, model, criterion, optimizer, epoch, opt, val_loader)
        time2 = time.time()
        if epoch == 1:
            start_loss_val = loss.val
            start_loss_avg = loss.avg
        print('epoch {}, total time {:.2f}, min val {:.3f}'.format(epoch, time2 - time1, min_val_loss))

        # logger
        logger.add_scalar('loss', loss.avg, epoch)
        logger.add_scalar('val_loss', val_loss.avg, epoch)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # early stop
        if val_loss.avg < min_val_loss and val_loss.avg > 0.01:
            min_val_loss = val_loss.avg
            if epoch > 50:
                save_file = os.path.join(
                    opt.save_folder, 'min_val.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, save_file)
            tol_count = 0
        else:
            tol_count += 1

        if tol_count == tol:
            # If you want to save model based on epoch number
            #save_file = os.path.join(
            #    opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            #save_model(model, optimizer, opt, epoch, save_file)
            end_loss_val = loss.val
            end_loss_avg = loss.avg
            break

        # nan stop
        if pd.isna(loss.avg):
            if pre_loss_avg != 0:
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, save_file)
                end_loss_avg = pre_loss_avg
                end_loss_val = pre_loss_val
                break
            else:
                break
        else:
            pre_loss_avg = loss.avg
            pre_loss_val = loss.val

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
            end_loss_val = loss.val
            end_loss_avg = loss.avg

        # Check early stop
        if (opt.epoch_early_stop > 0 and epoch > opt.epoch_early_stop and loss.avg > opt.lr_early_stop) or loss.avg < 0.01:
            end_loss_val = loss.val
            end_loss_avg = loss.avg
            break

    # save log to txt file
    with open('validation_records.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        # choose training log saving configuration based on the model structure you picked
        # 3+1 model
        # writer.writerow([opt.model, opt.model_n_hidden_1, opt.model_n_hidden_2, opt.model_out_dim, opt.feat_dim, opt.batch_size, opt.round, start_loss_val, start_loss_avg, end_loss_val, end_loss_avg])
        # 2+1 model with seed
        writer.writerow([opt.model, opt.model_n_hidden_1, opt.model_out_dim, opt.feat_dim, opt.batch_size, opt.split_seed, start_loss_val, start_loss_avg, end_loss_val, end_loss_avg, opt.learning_rate, opt.l2_rate, opt.split_class_num])
        # 1+1 model
        # writer.writerow([opt.model, opt.model_out_dim, opt.feat_dim, opt.batch_size, opt.round,
        #                 start_loss_val, start_loss_avg, end_loss_val, end_loss_avg])
    # save the last model
    # save_file = os.path.join(
    #    opt.save_folder, 'last.pth')
    # save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
