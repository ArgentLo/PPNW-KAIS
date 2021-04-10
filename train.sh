python3 train.py --gpu=0 --dataset=data/pinterest.npz --pretrain=pretrain/pinterest_e50.npz --neg=4 --hops=2 --iters=1 --batch_size=128 --use_unpop_weight=True --logdir=result/CMN_citeulike

# Options for diff datasets:
#--dataset=data/citeulike-a.npz --pretrain=pretrain/citeulike-a_e50.npz --neg=6
#--dataset=data/pinterest.npz --pretrain=pretrain/pinterest_e50.npz --neg=4
#--dataset=data/ml-1m.npz --pretrain=pretrain/ml-1m_e50.npz --neg=4
