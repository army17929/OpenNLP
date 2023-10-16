#!/bin/bash
#SBATCH --time=0-01:00
#SBATCH -A edu_res
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:4

echo job nodelist: $SLURM_JOB_NODELIST # It will give you the allocated resources

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
echo "Nodes Array: ${nodes[@]}"

nodes=${nodes[@]}
nodes_array=($nodes)
echo $nodes_array

head_node=${nodes_array[0]}
second_node=${nodes_array[1]}
echo HEAD: $head_node
echo SECOND: $second_node

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo NODE1_IP: $head_node_ip
second_node_ip=$(srun --nodes=1 --ntasks=1 -w "$second_node" hostname --ip-address)
echo NODE2_IP: $second_node_ip



#torchrun --nnodes 2 --nproc_per_node 4 --node_rank 1 --rdzv_id 456 --rdzv_backend c10d --rdzv_endpoint hoodoo-compute-10:29500 multinode_toymodel.py 5 3
#salloc --time 0-01:00 -A edu_res --nodes 2 --ntasks-per-node 4 --gpus-per-node a100:4

torchrun --nnodes 2 --nproc_per_node 4 --node_rank 0 --master_addr $head_node --master_port 29500 test_multinode.py 