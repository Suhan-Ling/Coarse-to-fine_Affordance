# Articulated Object Manipulation with Coarse-to-fine Affordance for Mitigating the Effect of Point Cloud Noise

![Door](./images/Door.gif)

We propose a novel affordance framework that utilizes an eye-on-hand camera to obtain closer views tailored to the requirements of manipulation tasks. This approach effectively addresses the challenges posed by the noise present in the point cloud data.

## About this paper

Website: https://sites.google.com/view/coarse-to-fine/

Paper: https://arxiv.org/abs/2402.18699

## About this repository

```
code/         # contains code and scripts
stats/        # contains helper statistics
```

## Dependencies

Please follow the instructions in `./data/README.md`.

## Running

Run the scripts in `./code/scripts` in the following order:

```
run_collect.sh --> run_collect_real.sh --> run_train.sh -->
run_train_actor.sh --> run_collect_interact.sh --> run_test_interact.sh
```

You may need to modify few parameters to suit your settings. For example, `--category`, `--data_dir` and `--out_dir`.
