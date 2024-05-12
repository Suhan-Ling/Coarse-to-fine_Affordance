import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
from data import SAPIENVisionDataset
import utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'blender_utils'))
from tensorboardX import SummaryWriter
from pointnet2_ops.pointnet2_utils import furthest_point_sample


def train(conf, train_data_list, val_data_list):
    # create training and validation datasets and data loaders
    data_features = ['up', 'forward', 'pc_near', 'pc_far']

    # load network model
    model_def = utils.get_model_module(conf.model_version)

    # create models
    critic = model_def.Critic(feat_dim=conf.feat_dim)

    data_to_restore = torch.load(os.path.join("../logs", conf.load_dir,
                                              'ckpts', f"{conf.load_epoch}-network.pth"))
    critic.load_state_dict(data_to_restore)
    critic.to(conf.device)

    network = model_def.Actor(feat_dim=conf.feat_dim, rv_dim=conf.rv_dim, rv_cnt=conf.rv_cnt)

    network_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=conf.lr,
                                   weight_decay=conf.weight_decay)
    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every,
                                                           gamma=conf.lr_decay_by)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)      TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        train_writer = SummaryWriter(os.path.join(conf.tb_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(conf.tb_dir, 'val'))

    # send parameters to device
    network.to(conf.device)
    utils.optimizer_to_device(network_opt, conf.device)
    # load dataset
    train_dataset = SAPIENVisionDataset([conf.primact_type], data_features)
    val_dataset = SAPIENVisionDataset([conf.primact_type], data_features)

    ### load data for the current epoch
    val_dataset.load_data('../data/0515_pull_StorageFurniture_1', real=conf.real, succ_only=True)
    train_dataset.load_data('../data/0515_pull_StorageFurniture_2', real=conf.real, succ_only=True)
    train_dataset.load_data('../data/0515_pull_StorageFurniture_3', real=conf.real, succ_only=True)
    train_dataset.load_data('../data/0515_pull_StorageFurniture_4', real=conf.real, succ_only=True)
    train_dataset.load_data('../data/0515_pull_StorageFurniture_5', real=conf.real, succ_only=True)
    train_dataset.load_data('../data/0515_pull_StorageFurniture_6', real=conf.real, succ_only=True)
    train_dataset.load_data('../data/0515_pull_StorageFurniture_7', real=conf.real, succ_only=True)
    train_dataset.load_data('../data/0515_pull_StorageFurniture_8', real=conf.real, succ_only=True)

    print(f'val len: {len(val_dataset)}')
    print(f'train len: {len(train_dataset)}')

    utils.printout(conf.flog, str(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=5,
                                                   drop_last=True, collate_fn=utils.collate_feats,
                                                   worker_init_fn=utils.worker_init_fn)
    train_num_batch = len(train_dataloader)
    utils.printout(conf.flog, str(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size,
                                                 shuffle=True, pin_memory=True, num_workers=5,
                                                 drop_last=True, collate_fn=utils.collate_feats,
                                                 worker_init_fn=utils.worker_init_fn)
    val_num_batch = len(val_dataloader)
    print('train_num_batch: %d, val_num_batch: %d' % (train_num_batch, val_num_batch))

    # start training
    start_time = time.time()
    last_train_console_log_step, last_val_console_log_step = None, None
    start_epoch = 0
    network_opt.zero_grad()

    # train for every epoch
    save_step = 0
    for epoch in range(start_epoch, conf.epochs):
        ### collect data for the current epoch
        if epoch > start_epoch:
            utils.printout(conf.flog,
                           f'  [{strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} Waiting epoch-{epoch} data ]')

        ### print log
        if not conf.no_console_log:
            utils.printout(conf.flog, f'training run {conf.exp_name}')
            utils.printout(conf.flog, header)

        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)

        train_fraction_done = 0.0
        val_fraction_done = 0.0
        val_batch_ind = -1

        ### train for every batch

        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind
            log_console = not conf.no_console_log and \
                          (last_train_console_log_step is None or
                           train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # save checkpoint
            if epoch % 5 == 0 and train_batch_ind == 0:
                with torch.no_grad():
                    utils.printout(conf.flog, 'Saving checkpoint ...... ')
                    torch.save(network.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-actor.pth' % (epoch)))
                    utils.printout(conf.flog, 'DONE')

            # set models to training mode
            network.train()

            # forward pass (including logging)
            losses = actor_forward(batch=batch, data_features=data_features, network=network,
                                   critic=critic, conf=conf, is_val=False, step=train_step,
                                   epoch=epoch, batch_ind=train_batch_ind,
                                   num_batch=train_num_batch, start_time=start_time,
                                   log_console=log_console, log_tb=not conf.no_tb_log,
                                   tb_writer=train_writer, lr=network_opt.param_groups[0]['lr'])
            total_loss = losses
            total_loss.backward()
            # optimize one step
            network_opt.step()
            network_opt.zero_grad()
            network_lr_scheduler.step()

            # validate one batch
            val_cnt = 0
            total_loss, total_precision, total_recall, total_Fscore, total_accu = 0, 0, 0, 0, 0
            while val_fraction_done <= train_fraction_done and val_batch_ind + 1 < val_num_batch:
                val_cnt += 1
                val_batch_ind, val_batch = next(val_batches)

                val_fraction_done = (val_batch_ind + 1) / val_num_batch
                val_step = (epoch + val_fraction_done) * train_num_batch - 1

                log_console = not conf.no_console_log and \
                              (last_val_console_log_step is None or
                               val_step - last_val_console_log_step >= conf.console_log_interval)
                if log_console:
                    last_val_console_log_step = val_step

                # set models to evaluation mode
                network.eval()

                with torch.no_grad():
                    # forward pass (including logging)
                    total_loss = actor_forward(batch=val_batch, data_features=data_features,
                                               network=network, critic=critic, conf=conf,
                                               is_val=True,  step=val_step, epoch=epoch,
                                               batch_ind=val_batch_ind, num_batch=val_num_batch,
                                               start_time=start_time, log_console=log_console,
                                               log_tb=not conf.no_tb_log, tb_writer=val_writer,
                                               lr=network_opt.param_groups[0]['lr'])


def actor_forward(batch, data_features, network, critic, conf,
                  is_val=False, step=None, epoch=None, batch_ind=0,
                  num_batch=1, start_time=0.0, log_console=False,
                  log_tb=False, tb_writer=None, lr=None):

    dir = batch[data_features.index('up')]
    pc_near = batch[data_features.index('pc_near')]
    pc_far = batch[data_features.index('pc_far')]
    f_dir = batch[data_features.index('forward')]

    dir = torch.FloatTensor(np.array(dir)).to(conf.device)
    batch_size = dir.shape[0]
    dir = dir.view(batch_size, -1)

    f_dir = torch.FloatTensor(np.array(f_dir)).view(batch_size, -1).to(conf.device)

    pc_far = torch.cat(pc_far, dim=0).to(conf.device)  # B x N x 3   # point cloud
    pc_near = torch.cat(pc_near, dim=0).to(conf.device)
    with torch.no_grad():
        point_features = critic.get_near_feat(pc_near, pc_far)
    loss = network.get_loss(point_features, dir, f_dir).mean()  # B x 2, B x F x N

    # display information
    data_split = 'train'
    if is_val:
        data_split = 'val'

    with torch.no_grad():
        # log to console
        if log_console:
            utils.printout(conf.flog, \
                           f'''{strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} '''
                           f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                           f'''{data_split:^10s} '''
                           f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                           f'''{100. * (1 + batch_ind + num_batch * epoch) / (num_batch * conf.epochs):>9.1f}%      '''
                           f'''{lr:>5.2E} '''
                           f'''{loss.item():>10.5f}'''
                           )
            conf.flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('ce_loss', loss.item(), step)
            tb_writer.add_scalar('lr', lr, step)

    return loss


if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()

    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--model_version', type=str, default='model_all', help='model def file')
    parser.add_argument('--primact_type', type=str, default='pushing', help='the primact type')
    parser.add_argument('--data_dir_prefix', type=str, help='data directory')
    parser.add_argument('--offline_data_dir', type=str, help='data directory')
    parser.add_argument('--val_data_dir', type=str, help='data directory')
    parser.add_argument('--train_shape_fn', type=str, help='training shape file that indexs all shape-ids')
    parser.add_argument('--ins_cnt_fn', type=str, help='a file listing all category instance count')
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--load_epoch', type=int, default=20)
    parser.add_argument('--data_dir1', type=str)
    parser.add_argument('--data_dir2', type=str, default=None)
    parser.add_argument('--data_dir3', type=str, default=None)
    parser.add_argument('--data_dir4', type=str, default=None)
    parser.add_argument('--data_dir5', type=str, default=None)
    parser.add_argument('--data_dir6', type=str, default=None)
    parser.add_argument('--data_dir7', type=str, default=None)
    parser.add_argument('--data_dir8', type=str, default=None)

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='../logs', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--num_point_per_shape', type=int, default=10000)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--real', action='store_true', default=False)
    parser.add_argument('--rv_dim', type=int, default=10)
    parser.add_argument('--rv_cnt', type=int, default=100)

    # training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)
    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10,
                        help='number of optimization steps beween console log prints')

    # visu
    parser.add_argument('--num_batch_every_visu', type=int, default=1, help='num batch every visu')
    parser.add_argument('--num_epoch_every_visu', type=int, default=10, help='num epoch every visu')
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')

    # pc
    parser.add_argument('--sample_type', type=str, default='fps')
    # parse args
    conf = parser.parse_args()
    conf.ignore_joint_info = True
    ### prepare before training
    # make exp_name
    conf.exp_name = f'exp-{conf.model_version}_{conf.exp_suffix}'

    if conf.overwrite and conf.resume:
        raise ValueError('ERROR: cannot specify both --overwrite and --resume!')

    # mkdir exp_dir; ask for overwrite if necessary; or resume
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    print('exp_dir: ', conf.exp_dir)
    conf.tb_dir = os.path.join(conf.exp_dir, 'tb')
    if os.path.exists(conf.exp_dir):
        if not conf.resume:
            if not conf.overwrite:
                response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
                if response != 'y':
                    exit(1)
            shutil.rmtree(conf.exp_dir)
    else:
        if conf.resume:
            raise ValueError('ERROR: no training run named %s to resume!' % conf.exp_name)
    if not conf.resume:
        os.mkdir(conf.exp_dir)
        os.mkdir(conf.tb_dir)
        os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))
        if not conf.no_visu:
            os.mkdir(os.path.join(conf.exp_dir, 'val_visu'))

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    if not conf.resume:
        torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))

    # file log
    if conf.resume:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'a+')
    else:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    conf.flog = flog

    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')

    # set training device
    device = torch.device(conf.device)
    utils.printout(flog, f'Using device: {conf.device}\n')
    conf.device = device

    train_data_list, val_data_list = [], []
    train(conf, train_data_list, val_data_list)

    ### before quit
    # close file log
    flog.close()
