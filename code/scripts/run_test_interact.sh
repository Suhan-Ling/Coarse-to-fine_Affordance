CUDA_VISIBLE_DEVICES=0 \
  xvfb-run -a \
  python test_interact.py \
  --no_gui \
  --sapien_dir ../data \
  --data_dir ../data/test_0515_SturageFurniture \
  --state closed
