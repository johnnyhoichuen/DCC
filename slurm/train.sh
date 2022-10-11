#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=dcc-train
#SBATCH --time=00:01:00

#SBATCH -p gpu-share
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1

######SBATCH --mem-per-cpu=500M
######SBATCH --gpus=1
#SBATCH --gres=gpu:4
#SBATCH --mem=16G

#SBATCH --output=./training/train-%j.out # working
##SBATCH --error=test-%j.err

# email notificaitons when the job ends
#SBATCH --mail-user=hcchengaa@ust.hk
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

obs_radius=$1
notes=$2

# adding comments to the job
echo
echo -e "Training spec"
echo "ntasks: $SLURM_NTASKS"
echo "CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES"
echo "Memory: $SLURM_MEM_PER_NODE"
echo
echo -e "Training notes: $notes"
echo

# set -x

# # __doc_head_address_start__

# # Getting the node names
# nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
# nodes_array=($nodes)

# # start head node first
# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# # *new reader can ignore this part
# # if we detect a space character in the head node IP, we'll
# # convert it to an ipv4 address. This step is optional.
# if [[ "$head_node_ip" == *" "* ]]; then
# IFS=' ' read -ra ADDR <<<"$head_node_ip"
# if [[ ${#ADDR[0]} -gt 16 ]]; then
#   head_node_ip=${ADDR[1]}
# else
#   head_node_ip=${ADDR[0]}
# fi
# echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
# fi
# # __doc_head_address_end__



# # __doc_head_ray_start__
# port=6379
# ip_head=$head_node_ip:$port
# export ip_head
# echo "IP Head: $ip_head"

# echo "Starting HEAD at $head_node"
# srun --nodes=1 --ntasks=1 -w "$head_node" \
#     # ray start --address="$ip_head" \
#     ray start --head --node-ip-address="$head_node_ip" --port=$port \
#     --num-cpus "${SLURM_CPUS_PER_TASK}" \
#     # --num-gpus "${SLURM_GPUS}" \
#     --block &
# # __doc_head_ray_end__



# # __doc_worker_ray_start__
# # optional, though may be useful in certain versions of Ray < 1.0.
# sleep 10

# # number of nodes other than the head node
# worker_num=$((SLURM_JOB_NUM_NODES - 1))

# for ((i = 1; i <= worker_num; i++)); do
#     node_i=${nodes_array[$i]}
#     echo "Starting WORKER $i at $node_i"
#     srun --nodes=1 --ntasks=1 -w "$node_i" \
#         ray start --address "$ip_head" \
#         # ray start --node-ip-address="$head_node_ip" --port=$port \
#         --num-cpus "${SLURM_CPUS_PER_TASK}" \
#         --num-gpus "${SLURM_GPUS}" \
#         --block &
#     sleep 5
# done
# # __doc_worker_ray_end__


# __doc_script_start__
# train with both cpu and gpu
echo -e "\n\n\n Training"
cd .. # go back to the main folder

# 1. training from scratch
python -u train.py "$obs_radius"
# python -u train.py #"$SLURM_CPUS_PER_TASK" #"$head_node_ip"
# 2. to continue training
# ckpt_path='22-08-08_at_23.37.56/135000.pt' # r2 normal
# ckpt_path="22-08-11_at_23.13.17/82500.pt" # r5 1st retrain
# python -u train.py "$ckpt_path"

# ##############################################
# ## working version
# ##############################################
# # shellcheck disable=SC2206
# #SBATCH --job-name=dcc-train
# #SBATCH --time=24:00:00

# #SBATCH -p gpu-share
# #SBATCH --cpus-per-task=16
# #SBATCH --tasks-per-node=1
# #SBATCH --nodes=1

# #SBATCH --gres=gpu:4
# #SBATCH --mem=16G

# #SBATCH --output=./training/train-%j.out # working

# # email notificaitons when the job begins
# #SBATCH --mail-user=hcchengaa@ust.hk
# #SBATCH --mail-type=end

# # adding comments to the job
# echo
# echo -e "Training spec"
# echo "ntasks: $SLURM_NTASKS"
# echo "CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES"
# echo "Memory: $SLURM_MEM_PER_NODE"
# echo
# echo -e "Training notes: $@"
# echo

# # train with both cpu and gpu
# echo -e "\n\n\n Training"
# cd .. # go back to the main folder
# python -u train.py

##############################################
## old version
##############################################
# #!/bin/bash

# #SBATCH --job-name dcc-train
# #SBATCH --time=24:00:00

# # **** available partition ****
# # 1. cpu-share (max: 6)
# # 2. gpu-share (max: 2)
# # 3. himem-share (max: 2)
# # *****************

# #SBATCH -p gpu-share
# #SBATCH --mem=16G # memory per node
# #SBATCH -N 1 -n 16 --gres=gpu:1

# ##SBATCH --cpus-per-task=6
# ##SBATCH --nodes=1
# ##SBATCH --tasks-per-node=1


# # email notificaitons when the job begins
# #SBATCH --mail-user=hcchengaa@ust.hk
# #SBATCH --mail-type=end

# srun which python # confirm python version. This should be executed if we used 'conda activate big2rl' before

# # train with both cpu and gpu
# echo -e "\n\n\n Training"
# cd .. # go back to the main folder

# srun python train.py

# echo -e "Training done"