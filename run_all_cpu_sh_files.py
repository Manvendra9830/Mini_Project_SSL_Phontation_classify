import os 
import sys
import numpy as np
from IPython import embed


line1 = '#!/bin/bash'

line2 = '#SBATCH --time=20:00'
line3 = '#SBATCH -o /scratch/elec/t412-speechcom/Farhad/code/Kiran/res/res_%a.log'
line4 = '#SBATCH --constraint="csl"'
line5 = '#SBATCH --mem-per-cpu=16G'
line6 = '#SBATCH --mail-type=END'

line8 = 'module load mamba'
line9 = 'source /scratch/elec/t412-speechcom/Farhad/r2b/bin/activate'

line10 = 'cd /scratch/elec/t412-speechcom/Farhad/code/Kiran'

def main():

    task_type = ["binary"]

    for task in task_type:
        for i in range(13):
            name = f"sh_files/{task}_wav2vec2B_{str(i)}.sh"

            infile = open(name, 'w')
            infile.write(line1)
            infile.write('\n')
            infile.write('\n')
            infile.write(line2)
            infile.write('\n')
            infile.write(line3)
            infile.write('\n')
            infile.write(line4)
            infile.write('\n')
            infile.write(line5)
            infile.write('\n')
            infile.write(line6)
            infile.write('\n')
            infile.write('\n')
            infile.write(line8)
            infile.write('\n')
            infile.write(line9)
            infile.write('\n')
            infile.write('\n')
            infile.write(line10)
            infile.write('\n')
            infile.write(f"python -u classification.py wav2vec2B {str(i)} {task}")
            infile.write('\n')
            infile.close()


    for task in task_type:
        for i in range(25):
            name = f"sh_files/{task}_wav2vec2L_{str(i)}.sh"

            infile = open(name, 'w')
            infile.write(line1)
            infile.write('\n')
            infile.write('\n')
            infile.write(line2)
            infile.write('\n')
            infile.write(line3)
            infile.write('\n')
            infile.write(line4)
            infile.write('\n')
            infile.write(line5)
            infile.write('\n')
            infile.write(line6)
            infile.write('\n')
            infile.write('\n')
            infile.write(line8)
            infile.write('\n')
            infile.write(line9)
            infile.write('\n')
            infile.write('\n')
            infile.write(line10)
            infile.write('\n')
            infile.write(f"python -u classification.py wav2vec2L {str(i)} {task}")
            infile.write('\n')
            infile.close()

    for task in task_type:
        for i in range(25):
            name = f"sh_files/{task}_hubert_{str(i)}.sh"

            infile = open(name, 'w')
            infile.write(line1)
            infile.write('\n')
            infile.write('\n')
            infile.write(line2)
            infile.write('\n')
            infile.write(line3)
            infile.write('\n')
            infile.write(line4)
            infile.write('\n')
            infile.write(line5)
            infile.write('\n')
            infile.write(line6)
            infile.write('\n')
            infile.write('\n')
            infile.write(line8)
            infile.write('\n')
            infile.write(line9)
            infile.write('\n')
            infile.write('\n')
            infile.write(line10)
            infile.write('\n')
            infile.write(f"python -u classification.py hubert {str(i)} {task}")
            infile.write('\n')
            infile.close()

if __name__ == '__main__':
    main()

