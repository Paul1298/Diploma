import os
import numpy as np

def calc(data_dir):
    algo_map = {}
    for dirpath, dirnames, files in os.walk(data_dir):
        if dirnames and dirnames[0].startswith('start'):
            cur_algo = dirpath.split('/')[-1]
        for f in files:
            if f == 'evaluations.log':
                log_file = os.path.join(dirpath, f)
                with open(log_file) as file:
                    next(file)

                    for iter_num, line in enumerate(file):
                        parts = line.strip().split('\t')

                        iter_time = float(parts[-1])
                        algo_map[cur_algo] = algo_map.get(cur_algo, []) + [iter_time]

    with open(os.path.join(data_dir, data_dir.lstrip('data_') + '_iter_time.tsv'), 'w') as log_file:
        print('Algorithm',
              'min',
              'mean',
              'max',
              'count',
              file=log_file, sep='\t')

        for a, v in algo_map.items():
            print(a,
                  np.min(v),
                  np.mean(v),
                  np.max(v),
                  len(v),
                  file=log_file, sep='\t')

if __name__ == '__main__':
    # data_dirs = list(filter(lambda x: x.startswith('data'), next(os.walk('.'))[1]))
    data_dirs = ['4_YRI_CEU_CHB_JPT_17_Jou']

    [calc(d_d) for d_d in data_dirs]