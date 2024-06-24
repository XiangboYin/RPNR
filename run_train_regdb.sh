# 1 Trial
CUDA_VISIBLE_DEVICES=0,1 \
 python train_regdb.py -mb CMhcl -b 128 -a agw -d  regdb_rgb \
 --iters 100 --momentum 0.1 --eps 0.6 --num-instances 16 --trial 1 \
 --data-dir "/data/yxb/datasets/ReIDData/RegDB/"

# 10 Trials
# for trial in 1 2 3 4 5 6 7 8 9 10
# do
# CUDA_VISIBLE_DEVICES=0,1 \
#  python train_regdb.py -mb CMhcl -b 128 -a agw -d  regdb_rgb \
#  --iters 100 --momentum 0.1 --eps 0.6 --num-instances 16 --trial $trial \
#  --data-dir "/data/yxb/datasets/ReIDData/RegDB/"
# done
# echo 'Don