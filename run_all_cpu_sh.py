import os 
import sys
import numpy as np
from IPython import embed


task_type = ["binary"]

cnt = 1

for task in task_type:
    for i in range(13):
        name = f"sh_files/{task}_wav2vec2B_{str(i)}.sh"
        run_cmd = 'sbatch' + ' ' + '-a' + ' ' + str(cnt) + ' ' + name
        os.system(run_cmd)
        cnt += 1

for task in task_type:
    for i in range(25):
        name = f"sh_files/{task}_wav2vec2L_{str(i)}.sh"
        run_cmd = 'sbatch' + ' ' + '-a' + ' ' + str(cnt) + ' ' + name
        os.system(run_cmd)
        cnt += 1

for task in task_type:
    for i in range(25):
        name = f"sh_files/{task}_hubert_{str(i)}.sh"
        run_cmd = 'sbatch' + ' ' + '-a' + ' ' + str(cnt) + ' ' + name
        os.system(run_cmd)
        cnt += 1
