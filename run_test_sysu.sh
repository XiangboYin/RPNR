CUDA_VISIBLE_DEVICES=0,1 \
 python test_sysu_vis.py -b 256 -a agw -d  sysu_all \
 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
 --data-dir "/data/yxb/datasets/ReIDData/SYSU-MM01/" 