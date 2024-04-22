import os
import concurrent.futures

seed_list = [10, 74, 341, 925, 1036, 1555, 1777, 2030, 2060, 2090, 2200,
             2222, 2268, 2289, 2341, 2741, 2765, 2782, 2857, 2864, 2918,
             2937, 2948, 2960, 2968, 3005, 3008, 3154, 3199, 3212, 3388,
             3455, 3466, 3611, 3679, 3927, 4000, 4013, 4416, 4520]
def run_training(get_seed, device):
    # Fine-Tune 2 + 1 layers
    dim_1_list = [5196, 4096]
    dim_2_list = [2048, 1024]
    dim_3_list = [512, 256]
    batch_size = 120
    seed = get_seed
    train_epoch = 20000
    round_num = 1
    cancer_group = 'BRCA'
    validation = 'Affy'

    if cancer_group == 'SM' or cancer_group == 'BLCA' or cancer_group == 'CESC' or cancer_group == 'HNSC' or cancer_group == 'LUSC':
        input_dim = 16008
    elif cancer_group == 'NGT' or cancer_group == 'LGG' or cancer_group == 'GBM':
        input_dim = 20531
    elif cancer_group == 'MSE':
        input_dim = 20531
    elif cancer_group == 'CCPRC' or cancer_group == 'KIRP':
        input_dim = 20531
    elif cancer_group == 'HC':
        input_dim = 20531
    elif cancer_group == 'GG' or cancer_group == 'COAD' or cancer_group == 'READ' or cancer_group == 'STAD':
        input_dim = 12503
    elif cancer_group == 'DS':
        input_dim = 14193
    elif cancer_group == 'LC':
        input_dim = 20531
    elif cancer_group == 'SARC' or cancer_group == 'BRCA' or cancer_group == 'THCA':
        input_dim = 20531
    elif cancer_group == 'PRAD':
        input_dim = 16135
    elif cancer_group == 'UCEC':
        input_dim = 13984
    elif cancer_group == 'OV':
        input_dim = 15962

    if validation == 'Affy':
        input_dim = 13235

    split_class_num = 9
    l2_list = [0.001, 0.0007, 0.0003]
    learning_rate_list = [0.00002]

    for i in range(len(dim_1_list)):
        for j in range(len(dim_2_list)):
            for k in range(len(dim_3_list)):
                for l2_rate in l2_list:
                    for lr in learning_rate_list: #for round in range(round_num):
                        for round in range(round_num):
                            model_n_hidden_1 = str(dim_1_list[i])
                            model_out_dim = str(dim_2_list[j])
                            feat_dim = str(dim_3_list[k])

                            order_line = 'python main_CLCP.py --dataset CancerRNA --model_in_dim ' + str(input_dim) + ' --model_n_hidden_1 ' \
                                             + model_n_hidden_1 + ' --model_out_dim ' \
                                             + model_out_dim + ' --feat_dim ' + feat_dim + ' --batch_size ' \
                                             + str(batch_size) + ' --train_test 80' \
                                             + ' --split_seed ' + str(seed) \
                                             + ' --save_freq ' + str(train_epoch) + ' --epochs ' + str(train_epoch) \
                                             + ' --learning_rate ' + str(lr) \
                                             + ' --round ' + str(round + 1) + ' --lr_decay_epochs 0' \
                                             + ' --l2_rate ' + str(l2_rate) + ' --split_class_num ' + str(split_class_num) \
                                             + ' --os W --task WholeTimeSeq --cancer_group ' + cancer_group \
                                             + ' --gpu_device ' + str(device) + ' --validation ' + validation

                            print('Execute following:', order_line)
                            os.system(order_line)

                            order_line = 'python GenerateFeatures.py --layer_name layer1 --model_in_dim ' + str(input_dim) + ' --dim_1_list ' \
                                         + model_n_hidden_1 + ' --dim_2_list ' \
                                         + model_out_dim + ' --dim_3_list ' + feat_dim + ' --batch_size ' \
                                         + str(batch_size) + ' --l2_rate ' + str(l2_rate) + ' --validation ' + validation \
                                         + ' --seed ' + str(seed) + ' --round ' + str(round + 1) + ' --gpu_device ' + str(device) \
                                         + ' --learning_rate_list ' + str(lr) + ' --split_class_num ' + str(split_class_num) + ' --task WholeTimeSeq --cancer_group ' + cancer_group
                            print('Execute following:', order_line)
                            os.system(order_line)

                            order_line = 'python GenerateFeatures.py --layer_name layer2 --model_in_dim ' + str(input_dim) + ' --dim_1_list ' \
                                         + model_n_hidden_1 + ' --dim_2_list ' \
                                         + model_out_dim + ' --dim_3_list ' + feat_dim + ' --batch_size ' \
                                         + str(batch_size) + ' --l2_rate ' + str(l2_rate) + ' --validation ' + validation \
                                         + ' --seed ' + str(seed) + ' --round ' + str(round + 1) + ' --gpu_device ' + str(device) \
                                         + ' --learning_rate_list ' + str(lr) + ' --split_class_num ' + str(split_class_num) + ' --task WholeTimeSeq --cancer_group ' + cancer_group
                            print('Execute following:', order_line)
                            os.system(order_line)

                            order_line = 'python GenerateFeatures.py --layer_name head --model_in_dim ' + str(input_dim) + ' --dim_1_list ' \
                                         + model_n_hidden_1 + ' --dim_2_list ' \
                                         + model_out_dim + ' --dim_3_list ' + feat_dim + ' --batch_size ' \
                                         + str(batch_size) + ' --l2_rate ' + str(l2_rate) + ' --validation ' + validation \
                                         + ' --seed ' + str(seed) + ' --round ' + str(round + 1) + ' --gpu_device ' + str(device) \
                                         + ' --learning_rate_list ' + str(lr) + ' --split_class_num ' + str(split_class_num) + ' --task WholeTimeSeq --cancer_group ' + cancer_group
                            print('Execute following:', order_line)
                            os.system(order_line)
    return 'Finish seed ' + str(get_seed) + ' on device ' + str(device)
# specify the number of threads to use
num_threads = 4

# divide the seeds into two equal parts for each device
seeds_per_device = 20

# create a list of thread objects
threads = []

# create threads for device 0
with concurrent.futures.ProcessPoolExecutor(num_threads) as executor:
    for i in range(5):
        start = i * num_threads
        end = (i + 1) * num_threads

        # create a new thread and add it to the list
        for result in executor.map(run_training, seed_list[start:end], [0] * num_threads):
            print(result)

# create threads for device 1 (rerun after run for device 0)
'''
with concurrent.futures.ProcessPoolExecutor(num_threads) as executor:
    for i in range(5):
        start = i * num_threads + seeds_per_device
        end = (i + 1) * num_threads + seeds_per_device

        # create a new thread and add it to the list
        for result in executor.map(run_training, seed_list[start:end], [1] * num_threads):
            print(result)
'''