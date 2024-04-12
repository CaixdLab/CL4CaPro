import os
import concurrent.futures

seed_list = [10, 74, 341, 925, 1036, 1555, 1777, 2030, 2060, 2090, 2200,
             2222, 2268, 2289, 2341, 2741, 2765, 2782, 2857, 2864, 2918,
             2937, 2948, 2960, 2968, 3005, 3008, 3154, 3199, 3212, 3388,
             3455, 3466, 3611, 3679, 3927, 4000, 4013, 4416, 4520]
# specify the number of threads to use
# cancer_group_list = ["BLCA", "BRCA", "CESC", "COAD", "HNSC", "KIRC", "KIRP", "LGG", "LIHC", "LUAD", "LUSC", "OV", "PRAD", "SARC", "SKCM", "STAD", "THCA", "UCEC"]
cancer_group_get = 'BLCA'
num_threads = 10
task = 'Classifier' # 'Cox'
for seed in seed_list:
    if task == 'Classifier':
        command = 'python Classifier_method.py --seed {} --cancer_group {} --core {} > ClassifierLog/{}.log &'.format(
            seed, cancer_group_get, num_threads)
    elif task == 'Cox':
        command = 'python Cox_method.py --seed {} --cancer_group {} --core {} > CoxLog/{}.log &'.format(
            seed, cancer_group_get, num_threads)
    os.system(command)
    print('Finish ' + cancer_group_get)