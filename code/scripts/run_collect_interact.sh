xvfb-run -a \
  python collect_interact.py \
  --no_gui \
  --category StorageFurniture \
  --out_dir ../data/test_0515_SturageFurniture \
  --sapien_dir ../data \
  --load_dir exp-model_all_0515_pull_StorageFurniture \
  --load_epoch 40 \
  --actor_dir exp-model_all_0515_pull_StorageFurniture_actor \
  --actor_epoch 40 \
  --data_per_shape 10 \
  --model_version model_all \
  --test
