import os
import concurrent.futures
import argparse

seed_list = [10, 74, 341, 925, 1036, 1555, 1777, 2030, 2060, 2090, 2200,
             2222, 2268, 2289, 2341, 2741, 2765, 2782, 2857, 2864, 2918,
             2937, 2948, 2960, 2968, 3005, 3008, 3154, 3199, 3212, 3388,
             3455, 3466, 3611, 3679, 3927, 4000, 4013, 4416, 4520]

def parse_option():
    parser = argparse.ArgumentParser('argument for generate auto validation results')

    parser.add_argument('--num_threads', type=int, default=10,
                        help='number of threads')
    parser.add_argument('--task', type=str, default='Classifier', choices=['Classifier', 'Cox'],
                        help='task (Classifier/Cox)')
    parser.add_argument('--cancer_group', type=str, default='SM',
                        help='cancer group (SM/NGT/MSE/CCPRC/HC/GG/DS/LC)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    opt = parser.parse_args()

    return opt

def main():
    opt_get = parse_option()
    # specify the number of threads to use
    # cancer_group_list = ["BLCA", "BRCA", "CESC", "COAD", "HNSC", "KIRC", "KIRP", "LGG", "LIHC", "LUAD", "LUSC", "OV", "PRAD", "SARC", "SKCM", "STAD", "THCA", "UCEC"]
    cancer_group_get = opt_get.cancer_group
    num_threads = opt_get.seed
    task = opt_get.task
    seed_get = opt_get.seed
    if seed_get != 0:
        for seed in seed_list:
            if task == 'Classifier':
                command = 'python Classifier_method.py --seed {} --cancer_group {} --core {} > ClassifierLog/{}.log &'.format(
                    seed, cancer_group_get, num_threads)
            elif task == 'Cox':
                command = 'python Cox_method.py --seed {} --cancer_group {} --core {} > CoxLog/{}.log &'.format(
                    seed, cancer_group_get, num_threads)
            os.system(command)
            print('Finish ' + cancer_group_get)
    else:
        if task == 'Classifier':
            command = 'python Classifier_method.py --seed {} --cancer_group {} --core {} > ClassifierLog/{}.log &'.format(
                seed_get, cancer_group_get, num_threads)
        elif task == 'Cox':
            command = 'python Cox_method.py --seed {} --cancer_group {} --core {} > CoxLog/{}.log &'.format(
                seed_get, cancer_group_get, num_threads)
        os.system(command)
        print('Finish ' + cancer_group_get)