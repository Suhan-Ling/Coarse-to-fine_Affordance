import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )
        self.SA_modules_near = nn.ModuleList()
        self.SA_modules_near.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules_near.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.FP_modules_near = nn.ModuleList()
        self.FP_modules_near.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules_near.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )
        self.fc_layer_near = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pc_near, pc_far, near_only=False):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        ############################# encode far ########################
        xyz_far, features_far = self._break_up_pc(pc_far)

        l_xyz_far, l_features_far = [xyz_far], [features_far]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz_far[i], l_features_far[i])
            l_xyz_far.append(li_xyz)
            l_features_far.append(li_features)

        if near_only:
            for i in range(-1, -3, -1):
                l_features_far[i - 1] = self.FP_modules[i](
                    l_xyz_far[i - 1], l_xyz_far[i], l_features_far[i - 1], l_features_far[i]
                )
        else:
            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                l_features_far[i - 1] = self.FP_modules[i](
                    l_xyz_far[i - 1], l_xyz_far[i], l_features_far[i - 1], l_features_far[i]
                )
        ##################################################################

        ############################# encode near ########################
        xyz_near, features_near = self._break_up_pc(pc_near)

        l_xyz_near, l_features_near = [xyz_near], [features_near]
        for i in range(len(self.SA_modules_near)):
            li_xyz, li_features = self.SA_modules_near[i](l_xyz_near[i], l_features_near[i])
            l_xyz_near.append(li_xyz)
            l_features_near.append(li_features)

        l_features_near[-1] = self.FP_modules[-2](l_xyz_near[-1], l_xyz_far[-2], l_features_near[-1], l_features_far[-2])

        for i in range(-1, -(len(self.FP_modules_near) + 1), -1):
            l_features_near[i - 1] = self.FP_modules_near[i](
                l_xyz_near[i - 1], l_xyz_near[i], l_features_near[i - 1], l_features_near[i]
            )
        ##################################################################

        if near_only:
            return self.fc_layer_near(l_features_near[0])
        else:
            return self.fc_layer_near(l_features_near[0]), self.fc_layer(l_features_far[0])

    def forward_far_only(self, pc_far):

        xyz_far, features_far = self._break_up_pc(pc_far)

        l_xyz_far, l_features_far = [xyz_far], [features_far]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz_far[i], l_features_far[i])
            l_xyz_far.append(li_xyz)
            l_features_far.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features_far[i - 1] = self.FP_modules[i](
                l_xyz_far[i - 1], l_xyz_far[i], l_features_far[i - 1], l_features_far[i]
            )

        return self.fc_layer(l_features_far[0])


class Critic(nn.Module):
    def __init__(self, feat_dim):
        super(Critic, self).__init__()
        self.mlp3 = nn.Linear(3 + 3, feat_dim)
        self.mlp1 = nn.Linear(feat_dim + feat_dim, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, 1)

        self.mlp_far = nn.Linear(feat_dim, 1)

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, dir, f_dir, pc_near, pc_far, near_only=False):
        batch_size = dir.shape[0]

        pc_near = pc_near.repeat(1, 1, 2)
        pc_far = pc_far.repeat(1, 1, 2)

        if near_only:
            near_feats = self.pointnet2(pc_near, pc_far, near_only=near_only)
        else :
            near_feats, far_feats = self.pointnet2(pc_near, pc_far, near_only=near_only)
        near_feat = near_feats[:, :, 0]

        x = torch.cat([dir, f_dir], dim=-1)
        x = self.mlp3(x)
        x = torch.cat([near_feat, x], dim=-1)
        x = self.mlp1(x)
        x = F.leaky_relu(x)
        x = self.mlp2(x).squeeze(-1)
        if not near_only:
            far_feat = far_feats[:, :, 0]
            far_x = self.mlp_far(far_feat).squeeze(-1)
            return x, far_x

        return x

    # cross entropy loss
    def get_ce_loss(self, dir, f_dir, pc_near, pc_far, gt_labels, near_only=False, far_valid=None, lbd=1):
        if near_only:
            pred_logits = self.forward(dir, f_dir, pc_near, pc_far, near_only=near_only)
            loss = self.BCELoss(pred_logits, gt_labels)
            return loss.mean()
        else :
            pred_logits_1, pred_logits_2 = self.forward(dir, f_dir, pc_near, pc_far, near_only=near_only)
            loss = self.BCELoss(pred_logits_1, gt_labels)
            loss1 = self.BCELoss(pred_logits_2, gt_labels)
            if not (far_valid is None):
                loss1 = loss1[far_valid]
            else:
                return loss.mean()
            return loss.mean() + loss1.mean() * lbd

    def get_aff_all(self, pc_near, pc_far, dir, f_dir):
        batch_size = pc_near.shape[0]
        pt_size = pc_near.shape[1]
        dir = dir.view(batch_size, -1)
        dir = dir.unsqueeze(1).repeat(1, pt_size, 1).view(batch_size * pt_size, -1)
        f_dir = f_dir.view(batch_size, -1)
        f_dir = f_dir.unsqueeze(1).repeat(1, pt_size, 1).view(batch_size * pt_size, -1)
        pc_near = pc_near.repeat(1, 1, 2)
        pc_far = pc_far.repeat(1, 1, 2)
        near_feats, far_feats = self.pointnet2(pc_near, pc_far, near_only=False)
        near_feats = near_feats.permute(0, 2, 1).reshape(batch_size * pt_size, -1)
        far_feats = far_feats.permute(0, 2, 1).reshape(batch_size * pt_size, -1)

        x = torch.cat([dir, f_dir], dim=-1)
        x = self.mlp3(x)
        x = torch.cat([near_feats, x], dim=-1)
        x = self.mlp1(x)
        x = F.leaky_relu(x)
        near_result_logits = self.mlp2(x).squeeze(-1)
        near_result_logits = torch.sigmoid(near_result_logits)

        far_result_logits = self.mlp_far(far_feats).squeeze(-1)
        far_result_logits = torch.sigmoid(far_result_logits)

        return near_result_logits, far_result_logits

    def get_aff_far(self, pc_far):
        batch_size = pc_far.shape[0]
        pt_size = pc_far.shape[1]
        pc_far = pc_far.repeat(1, 1, 2)
        far_feats = self.pointnet2.forward_far_only(pc_far)
        far_feats = far_feats.permute(0, 2, 1).reshape(batch_size * pt_size, -1)
        far_result_logits = self.mlp_far(far_feats).squeeze(-1)
        far_result_logits = torch.sigmoid(far_result_logits)
        return far_result_logits

    def get_near_feat(self, pc_near, pc_far):
        pc_near = pc_near.repeat(1, 1, 2)
        pc_far = pc_far.repeat(1, 1, 2)
        near_feats = self.pointnet2(pc_near, pc_far, near_only=True)
        return near_feats[:, :, 0]

    def get_near_feats(self, pc_near, pc_far):
        pc_near = pc_near.repeat(1, 1, 2)
        pc_far = pc_far.repeat(1, 1, 2)
        near_feats = self.pointnet2(pc_near, pc_far, near_only=True)
        return near_feats.permute(0, 2, 1)

    def inference_critic_score_diff_naction(self, dir, f_dir, feats, n):
        batch_size = 1
        pt_size = 10000

        dir = dir.view(batch_size * pt_size * n, -1)
        f_dir = f_dir.view(batch_size * pt_size * n, -1)
        feats = feats.unsqueeze(dim=1).repeat(1, n, 1).reshape(batch_size * pt_size * n, -1)

        x = torch.cat([dir, f_dir], dim=-1)
        x = self.mlp3(x)
        x = torch.cat([feats, x], dim=-1)
        x = self.mlp1(x)
        x = F.leaky_relu(x)
        pred_result_logits = self.mlp2(x).squeeze(-1)
        pred_result_logits = torch.sigmoid(pred_result_logits)
        return pred_result_logits

    def inference_critic_score_naction(self, dir, f_dir, feat):

        n = dir.shape[0]
        feats = feat.repeat(n).reshape(n, -1)

        x = torch.cat([dir, f_dir], dim=-1)
        x = self.mlp3(x)
        x = torch.cat([feats, x], dim=-1)
        x = self.mlp1(x)
        x = F.leaky_relu(x)
        pred_result_logits = self.mlp2(x).squeeze(-1)
        pred_result_logits = torch.sigmoid(pred_result_logits)
        return pred_result_logits


class Actor(nn.Module):
    def __init__(self, feat_dim, rv_dim, rv_cnt):
        super(Actor, self).__init__()

        self.mlp1 = nn.Linear(feat_dim + rv_dim, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, 3 + 3)

        self.rv_dim = rv_dim
        self.rv_cnt = rv_cnt
        self.feat_dim =feat_dim

    # pixel_feats B x F, rvs B x RV_DIM
    # output: B x 6
    def forward(self, pixel_feats, rvs):
        net = torch.cat([pixel_feats, rvs], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net).reshape(-1, 3, 2)
        net = self.bgs(net)[:, :, :2].reshape(-1, 6)
        return net

    # input sz bszx3x2
    def bgs(self, d6s):
        bsz = d6s.shape[0]
        b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
        a2 = d6s[:, :, 1]
        b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

    # batch geodesic loss for rotation matrices
    def bgdR(self, Rgts, Rps):
        Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
        Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1)  # batch trace
        # necessary or it might lead to nans and the likes
        theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
        return torch.acos(theta)

    # 6D-Rot loss
    # input sz bszx6
    def get_6d_rot_loss(self, pred_6d, gt_6d):
        pred_Rs = self.bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        gt_Rs = self.bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        theta = self.bgdR(gt_Rs, pred_Rs)
        return theta

    def get_loss(self, point_features, dirs1, dirs2):

        batch_size = point_features.shape[0]
        input_s6d = torch.cat([dirs1, dirs2], dim=1)

        rvs = torch.randn(batch_size, self.rv_cnt, self.rv_dim).float().to(point_features.device)
        expanded_net = point_features.unsqueeze(dim=1).repeat(1, self.rv_cnt, 1).reshape(batch_size * self.rv_cnt, -1)
        expanded_rvs = rvs.reshape(batch_size * self.rv_cnt, -1)
        expanded_pred_s6d = self.forward(expanded_net, expanded_rvs)

        expanded_input_s6d = input_s6d.unsqueeze(dim=1).repeat(1, self.rv_cnt, 1).reshape(batch_size * self.rv_cnt, -1)
        expanded_actor_coverage_loss_per_rv = self.get_6d_rot_loss(expanded_pred_s6d, expanded_input_s6d)
        actor_coverage_loss_per_rv = expanded_actor_coverage_loss_per_rv.reshape(batch_size, self.rv_cnt)
        actor_coverage_loss_per_data = actor_coverage_loss_per_rv.min(dim=1)[0]

        return actor_coverage_loss_per_data

    def inference_actor(self, feat):
        feat = feat.reshape(-1, self.feat_dim)
        batch_size = feat.shape[0]

        rvs = torch.randn(batch_size, self.rv_cnt, self.rv_dim).float().to(feat.device)
        expanded_net = feat.unsqueeze(dim=1).repeat(1, self.rv_cnt, 1).reshape(batch_size*self.rv_cnt, -1)
        expanded_rvs = rvs.reshape(batch_size*self.rv_cnt, -1)
        expanded_pred_s6d = self.forward(expanded_net, expanded_rvs)
        pred_s6d = expanded_pred_s6d.reshape(batch_size, self.rv_cnt, 6)
        return pred_s6d

    def inference_nactor_whole_pc(self, whole_feats, n):
        batch_size = 1
        pt_size = 10000
        net = whole_feats.reshape(-1, self.feat_dim)

        rvs = torch.randn(batch_size * pt_size, n, self.rv_dim).float().to(net.device)
        expanded_net = net.unsqueeze(dim=1).repeat(1, n, 1).reshape(batch_size * pt_size * n, -1)
        expanded_rvs = rvs.reshape(batch_size * pt_size * n, -1)
        expanded_pred_s6d = self.forward(expanded_net, expanded_rvs)
        pred_s6d = expanded_pred_s6d.reshape(batch_size * pt_size * n, 6)
        return pred_s6d
