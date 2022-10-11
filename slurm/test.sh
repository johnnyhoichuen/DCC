#!/bin/bash

#SBATCH --job-name dcc-test
#SBATCH --time=12:00:00
#SBATCH --mem=8G # memory per node

#######################################
## cpu only
##SBATCH -p cpu-share
##SBATCH -N 1 -n 1
##SBATCH --output=./testing/multi-cpu/test-%j.out # working
#######################################
# option 2, failed
#####SBATCH -p gpu-share
#####SBATCH -N 1 -n 1 --gres=gpu:2
#######################################
# gpu only
#SBATCH -p gpu-share
#SBATCH --nodes=1
#######SBATCH --tasks-per-node=1
#######SBATCH --ntasks=1
#######SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --output=./testing/multi-gpu/computation_time/test-%j.out # working
#######################################


##SBATCH --error=test-%j.err

srun which python # confirm python version. This should be executed if we used 'conda activate big2rl' before

# train with both cpu and gpu
echo -e "\n\n\nTesting started"
cd .. # go back to the main folder

#######################################
# Reminder: do not use "", use ''

# # r=5, normal
# model_train_time='22-08-02_at_01.13.25'
# start_range_1=90000 # 30-40mins per test, rmb to adjust time limit
# end_range_1=90000
# interval=15000

## r=4, normal
#model_train_time='22-07-26_at_18.43.26'
#start_range_1=120000 # 30-40mins per test, rmb to adjust time limit
#end_range_1=120000
#interval=15000

# r=3, normal
#model_train_time='22-08-03_at_20.55.01-r3'
#start_range_1=90000 # 30-40mins per test, rmb to adjust time limit
#end_range_1=90000
#interval=15000

## r=2, normal
# model_train_time='22-08-08_at_23.37.56'
# start_range_1=120000 # 30-40mins per test, rmb to adjust time limit
# end_range_1=120000
# interval=15000

# r=2 with max curriculum map length = 80
# model_train_time='22-08-06_at_21.24.46-r2-curri-update'
# start_range_1=45000 # 30-40mins per test, rmb to adjust time limit
# end_range_1=45000
# interval=15000

## r=1, normal
#model_train_time='22-08-11_at_22.45.38'
#start_range_1=105000
#end_range_1=105000
#interval=15000

# add test comment here
#####SBATCH --comment="testing 45000 & 60000"

obs_radius=$1
test_note=$@

echo -e "ntasks: $SLURM_NTASKS"
echo -e "GPU: $SLURM_GRES"
echo -e "Memory: $SLURM_MEM_PER_NODE"
echo
echo -e "Test note: $test_note"
echo

# double check the num_cpu & num_gpu defined above
# srun python test.py "$start_range_1" "$end_range_1" "$interval"

# using gpu & cpu with option 1/2
# srun -n 4 --gres=gpu:1 python test.py "$start_range_1" "$end_range_1" &
# srun -n 4 --gres=gpu:1 python test.py "$start_range_2" "$end_range_2"

# option 3

python test.py "$obs_radius"
# python test.py "$start_range_2" "$end_range_2" "$interval" "$model_train_time"


echo -e "Testing done"


# #######################################

# # working version with gpu

# #SBATCH --job-name dcc-test
# #SBATCH --time=2:00:00
# #SBATCH --mem=8G # memory per node

# #SBATCH -p gpu-share
# #SBATCH --nodes=1
# #SBATCH --gres=gpu:2
# #SBATCH --mem=16G

# #SBATCH --output=./testing/multi-cpu/test-%j.out # working

# srun which python # confirm python version. This should be executed if we used 'conda activate big2rl' before

# # train with both cpu and gpu
# echo -e "\n\n\nTesting started"
# cd .. # go back to the main folder

# model_train_time='22-08-02_at_01.13.25' # do not use "", use ''
# interval=15000

# start_range_1=45000
# end_range_1=45000

# echo -e "ntasks: $SLURM_NTASKS"
# echo -e "GPU: $SLURM_GRES"
# echo -e "Memory: $SLURM_MEM_PER_NODE"
# echo
# echo -e "Test note: $@"
# echo

# python test.py "$start_range_1" "$end_range_1" "$interval" "$model_train_time"

# echo -e "Testing done"