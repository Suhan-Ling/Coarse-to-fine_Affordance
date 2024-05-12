CUDA_VISIBLE_DEVICES=0 \
  python _train_all.py \
  --exp_suffix 0515_pull_StorageFurniture \
  --real \
  --batch_size 64 \
  --model_version model_all \
  --lbd 1 \
  --epoch 41
