CUDA_VISIBLE_DEVICES=0 \
  xvfb-run -a \
  python collect.py \
  --date 0515 \
  --category StorageFurniture \
  --no_gui \
  --num_processes 3 \
  --ww_range 20000 \
  --collect_num 1
