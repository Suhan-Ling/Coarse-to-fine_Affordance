CUDA_VISIBLE_DEVICES=0 \
  python train_actor.py \
  --model_version model_all \
  --exp_suffix 0515_pull_StorageFurniture_actor_40 \
  --batch_size 64 \
  --epochs 41 \
  --real \
  --load_dir exp-model_all_0515_pull_StorageFurniture \
  --load_epoch 40
