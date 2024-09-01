import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from utils import getters, setters
from utils.functions import AverageMeter, registerSTModel, normalize_common_torch, AffineKeypointAligner, find_matches, SpatialTransformer, get_downsampled_images, VecInt
from utils.loss import NccLoss, Grad2d
from models.backbones.layers import encoder, affineOptimization2d
from models.xfeat import XFeat
from ff_mi import mi_loss
from utils.functions import image2cat_kmeans, to_tensor
import os
import nibabel as nib

def get_flow(xfeat_wrap, img_x, img_y):

    dev = img_x.device
    h,w = img_x.shape[2:]

    kps_x, kps_y = xfeat_wrap.match_xfeat(img_x, img_y, top_k = 16000)
    kps_x, kps_y  = find_matches(kps_x, kps_y)

    kps_x_ = torch.from_numpy(kps_x).unsqueeze(0).float().to(dev)
    kps_y_ = torch.from_numpy(kps_y).unsqueeze(0).float().to(dev)

    kps_x_ = kps_x_ / (h-1) * 2 - 1
    kps_y_ = kps_y_ / (w-1) * 2 - 1

    affine_model = AffineKeypointAligner(2)
    pos_flow = affine_model.flow_from_points(kps_x_, kps_y_, (1,2,h,w))

    return pos_flow

class dispWarp(nn.Module):

    def __init__(self, in_cs=1, N_s=32, out_cs=2, lk_size=3):

        super(dispWarp, self).__init__()

        self.disp_field = nn.Sequential(
            encoder(2*in_cs, N_s, lk_size, 1, lk_size//2),
            encoder(N_s, 2*N_s, lk_size, 1, lk_size//2),
        )
        self.flow = nn.Conv2d(2*N_s, out_cs, 3, 1, 1)
        self.init_zero_flow()

    def init_zero_flow(self):
        self.flow.weight.data.fill_(0)
        self.flow.bias.data.fill_(0)

    def forward(self, x, y):

        flow = self.disp_field(torch.cat((y+x,y-x), dim=1))
        flow = self.flow(flow)

        return flow

class smallDispModel(nn.Module):

    def __init__(self, N_s=32, lk_size=3, ss=(832,832)):

        super(smallDispModel, self).__init__()

        self.disp_field = dispWarp(in_cs=N_s, N_s=N_s*4, out_cs=2, lk_size=lk_size)

        self.transformer_4 = SpatialTransformer([s // 16 for s in ss])
        self.transformer_3 = SpatialTransformer([s // 8 for s in ss])
        self.transformer_2 = SpatialTransformer([s // 4 for s in ss])
        self.transformer_1 = SpatialTransformer([s // 2 for s in ss])
        self.transformer_0 = SpatialTransformer([s // 1 for s in ss])

        self.aff_opt = affineOptimization2d(alpha=1.)

        self.pre_conv = nn.Sequential(
            encoder(1, N_s, 3, 1, 1),
            encoder(N_s, 2*N_s, 3, 1, 1),
            encoder(2*N_s, N_s, 3, 1, 1),
        )

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.kaiming_normal_(m.bias)

    def forward(self, x, y, y_msk, alpha=1):

        x = self.pre_conv(x)
        y = self.pre_conv(y)

        x0, y0, y_msk0 = x, y, y_msk
        x1 = F.interpolate(x0, scale_factor=0.5, mode='bilinear', align_corners=True)
        y1 = F.interpolate(y0, scale_factor=0.5, mode='bilinear', align_corners=True)
        y_msk1 = F.interpolate(y_msk0, scale_factor=0.5, mode='nearest')
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=True)
        y2 = F.interpolate(y1, scale_factor=0.5, mode='bilinear', align_corners=True)
        y_msk2 = F.interpolate(y_msk1, scale_factor=0.5, mode='nearest')
        x3 = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=True)
        y3 = F.interpolate(y2, scale_factor=0.5, mode='bilinear', align_corners=True)
        y_msk3 = F.interpolate(y_msk2, scale_factor=0.5, mode='nearest')
        x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=True)
        y4 = F.interpolate(y3, scale_factor=0.5, mode='bilinear', align_corners=True)
        y_msk4 = F.interpolate(y_msk3, scale_factor=0.5, mode='nearest')

        pflow_4 = self.disp_field(x4, y4)
        flow_4 = self.aff_opt(pflow_4, y_msk4)
        up_flow_4 = F.interpolate(flow_4, scale_factor=2, mode='bilinear', align_corners=True) * 2
        warped_x3 = self.transformer_3(x3, up_flow_4)

        pflow_3 = self.disp_field(warped_x3, y3)
        # flow_3 = pflow_3 + self.transformer_3(up_flow_4, pflow_3)
        flow_3 = pflow_3 + up_flow_4
        flow_3 = self.aff_opt(flow_3, y_msk3)
        up_flow_3 = F.interpolate(flow_3, scale_factor=2, mode='bilinear', align_corners=True) * 2
        warped_x2 = self.transformer_2(x2, up_flow_3)

        pflow_2 = self.disp_field(warped_x2, y2)
        flow_2 = pflow_2 + up_flow_3
        flow_2 = self.aff_opt(flow_2, y_msk2)
        up_flow_2 = F.interpolate(flow_2, scale_factor=2, mode='bilinear', align_corners=True) * 2
        warped_x1 = self.transformer_1(x1, up_flow_2)

        pflow_1 = self.disp_field(warped_x1, y1)
        flow_1 = pflow_1 + up_flow_2
        flow_1 = self.aff_opt(flow_1, y_msk1)
        up_flow_1 = F.interpolate(flow_1, scale_factor=2, mode='bilinear', align_corners=True) * 2
        warped_x0 = self.transformer_0(x0, up_flow_1)

        pflow_0 = self.disp_field(warped_x0, y0)
        flow_0 = pflow_0 + up_flow_1
        flow_0 = self.aff_opt(flow_0, y_msk0)

        pflow = [pflow_0, pflow_1, pflow_2, pflow_3, pflow_4]
        flows = [flow_0, flow_1, flow_2, flow_3, flow_4]

        xs = [x0, x1, x2, x3, x4]
        ys = [y0, y1, y2, y3, y4]
        y_msks = [y_msk0, y_msk1, y_msk2, y_msk3, y_msk4]

        return pflow, flows, xs, ys



def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    # Getting model-related components
    test_loader = getters.getDataLoader(opt, split='test')
    model, _ = getters.getTestModelWithCheckpoints(opt)
    reg_model_bi = registerSTModel(opt['img_size'], 'bilinear').cuda()
    creterion_ncc_0 = NccLoss(win=[19,19])
    creterion_ncc_1 = NccLoss(win=[13,13])
    creterion_ncc_2 = NccLoss(win=[9,9])
    creterion_ncc_3 = NccLoss(win=[5,5])
    creterion_ncc_4 = NccLoss(win=[3,3])


    xfeat_wrap = XFeat(device = opt['device'])
    xfeat_wrap.init_net(model.xfeat)
    model.eval()
    for param in model.parameters():
        param.requires_grad=False
    xfeat_wrap.eval()
    print('xfeat initialized')

    k_means_cluster = 8
    # kmeans = KMeans(n_clusters=8)
    '''
    Validation
    '''
    eval_ncc = AverageMeter()
    init_ncc = AverageMeter()
    for data in test_loader:
        round_best_ncc = 0
        round_best_flow = None

        for round_ in range(3):
            #### re-initialization
            flow_model = smallDispModel(N_s=32, lk_size=15).cuda()
            optimizer = torch.optim.Adam(flow_model.parameters(), lr=opt['lr'], weight_decay=0, amsgrad=True)
            flow_model.train()
            print('flow model initialized')
            # flow_model.random_init()

            model.eval()
            idx = data[-1].item()
            if opt['is_first_half']==1 and idx not in [33, 41, 134, 11, 155, 80, 174, 73, 91, 156, 61, 75, 171, 205, 159, 199, 84, 120, 194, 91]:
                continue
            elif opt['is_first_half']==0 and idx in [33, 41, 134, 11, 155, 80, 174, 73, 91, 156, 61, 75, 171, 205, 159, 199, 84, 120, 194, 91]:
                continue

            data = [Variable(t.cuda()) for t in data[:6]]
            x_imgs, y_imgs = data[0].float(), data[1].float()
            x_msk, y_msk = data[2].float(), data[3].float()
            x_imgs, y_imgs = x_imgs[...,0:832,0:832], y_imgs[...,0:832,0:832] # crop from 834 to 832
            x_msk, y_msk = x_msk[...,0:832,0:832], y_msk[...,0:832,0:832]

            y_imgs = torch.log(y_imgs + 1)
            x_feas, y_feas = model.tiramisu(x_imgs, y_imgs)
            x_feas, y_feas = normalize_common_torch(x_feas, y_feas)

            y_lbl = image2cat_kmeans(y_imgs.cpu().numpy()[0].reshape(832, 832, 1), k=k_means_cluster)

            y_lbl = to_tensor(y_lbl, True)

            pred_flow_ = get_flow(xfeat_wrap, x_feas, y_feas)

            ### wapred flow
            x_org = reg_model_bi(x_imgs, pred_flow_)
            xx = x_org[0].permute(1, 2, 0).cpu().numpy() #@.reshape(832, 832, 3)
            x_lbl = image2cat_kmeans(xx, k=k_means_cluster)
            x_lbl = to_tensor(x_lbl, True)

            x, y = x_feas.detach(), y_feas.detach()
            x = reg_model_bi(x, pred_flow_)

            x_lbl, y_lbl = x_lbl.detach(), y_lbl.detach().float()
            x_mask = torch.ones_like(x_lbl)
            x_mask = reg_model_bi(x_mask, pred_flow_)

            best_flow = pred_flow_
            best_ncc = 0

            print('Processing %d' % idx)
            for iter_ in range(opt['n_iters']):
                optimizer.zero_grad()
                pflows, flows, xs, ys = flow_model(x, y, y_msk)
                xs = get_downsampled_images(x, 4, mode='bilinear')
                ys = get_downsampled_images(y, 4, mode='bilinear')

                x_lbl_s = get_downsampled_images(x_lbl, 4, mode='nearest')
                y_lbl_s = get_downsampled_images(y_lbl, 4, mode='nearest')
                x_mask_s = get_downsampled_images(x_mask, 4, mode='nearest')

                ###################
                if iter_ < int(opt['n_iters']) / 2:
                    sim_loss_0 = creterion_ncc_0(flow_model.transformer_0(xs[0], flows[0]), ys[0])
                    sim_loss_1 = creterion_ncc_1(flow_model.transformer_1(xs[1], flows[1]), ys[1]) / 2
                    sim_loss_2 = creterion_ncc_2(flow_model.transformer_2(xs[2], flows[2]), ys[2]) / 4
                    sim_loss_3 = creterion_ncc_3(flow_model.transformer_3(xs[3], flows[3]), ys[3]) / 8
                    sim_loss_4 = creterion_ncc_4(flow_model.transformer_4(xs[4], flows[4]), ys[4]) / 16

                else:
                    warped_0 = flow_model.transformer_0(x_lbl_s[0], flows[0])
                    warped_1 = flow_model.transformer_1(x_lbl_s[1], flows[1])
                    warped_2 = flow_model.transformer_2(x_lbl_s[2], flows[2])
                    warped_3 = flow_model.transformer_3(x_lbl_s[3], flows[3])
                    warped_4 = flow_model.transformer_4(x_lbl_s[4], flows[4])

                    sim_loss_0 = (1. - mi_loss(y_lbl_s[0], warped_0, x_mask_s[0], Q_A=k_means_cluster, Q_B=k_means_cluster))
                    sim_loss_1 = (1. - mi_loss(y_lbl_s[1], warped_1, x_mask_s[1], Q_A=k_means_cluster, Q_B=k_means_cluster)) / 2
                    sim_loss_2 = (1. - mi_loss(y_lbl_s[2], warped_2, x_mask_s[2], Q_A=k_means_cluster, Q_B=k_means_cluster)) / 4
                    sim_loss_3 = (1. - mi_loss(y_lbl_s[3], warped_3, x_mask_s[3], Q_A=k_means_cluster, Q_B=k_means_cluster)) / 8
                    sim_loss_4 = (1. - mi_loss(y_lbl_s[4], warped_4, x_mask_s[4], Q_A=k_means_cluster, Q_B=k_means_cluster)) / 16


                sim_loss = sim_loss_0 + sim_loss_1 + sim_loss_2 + sim_loss_3 + sim_loss_4

                reg_loss = sim_loss * 0

                # pred_flow = flows[0] + reg_model_bi(pred_flow_, flows[0])
                pred_flow = flows[0] + pred_flow_

                loss = opt['sim_w']*sim_loss + opt['reg_w']*reg_loss

                loss.backward()
                optimizer.step()
                #
                warped_x = reg_model_bi(x_lbl, flows[0])


                init_ncc_ = mi_loss(x_lbl, y_lbl, x_mask, Q_A=k_means_cluster, Q_B=k_means_cluster)
                eval_ncc_ = mi_loss(warped_x, y_lbl, x_mask, Q_A=k_means_cluster, Q_B=k_means_cluster)

                if eval_ncc_.item() > best_ncc:
                    best_ncc = eval_ncc_.item()
                    best_flow = pred_flow

                print(
                    'round%d [%d/%d], loss: %.4f, sim: %.4f, reg: %.4f, init_ncc: %.4f, eval_ncc: %.7f, best_ncc: %.7f' %
                    (round_, iter_, opt['n_iters'], loss.item(), sim_loss.item(), reg_loss.item(), init_ncc_.item(),
                     eval_ncc_.item(), best_ncc))

            print('current round %d, round ncc %.4f, round best ncc %.4f' %(round_, best_ncc, round_best_ncc))
                # Save the best results across all rounds
            if best_ncc > round_best_ncc:
                round_best_ncc = best_ncc
                round_best_flow = best_flow
                init_ncc.update(init_ncc_.item(), x.numel())
                eval_ncc.update(round_best_ncc, x.numel())

                warped_x = torch.clamp(reg_model_bi(x_imgs, round_best_flow), 0, 1).detach()
                os.makedirs('save_test_data', exist_ok=True)
                warped_x = warped_x.permute(0, 2, 3, 1).detach().cpu().numpy() * 255
                nib.save(nib.Nifti1Image(warped_x, np.eye(4)), 'save_test_data/%s_warped.nii.gz' % str(idx).zfill(4))
                y_imgs = y_imgs.permute(0, 2, 3, 1).detach().cpu().numpy() * 255
                nib.save(nib.Nifti1Image(y_imgs, np.eye(4)), 'save_test_data/%s_gt.nii.gz' % str(idx).zfill(4))

                os.makedirs('fields', exist_ok=True)
                save_name = 'disp_%s_%s.nii.gz' % (str(idx).zfill(4), str(idx).zfill(4))
                save_fp = os.path.join('fields', save_name)
                round_best_flow = F.pad(round_best_flow, (0, 2, 0, 2), mode='replicate')
                round_best_flow = round_best_flow.permute(2, 3, 0, 1).detach().cpu().numpy()
                nib.save(nib.Nifti1Image(round_best_flow, np.eye(4)), save_fp)
                print('new best!! Saved %s' % save_fp)


            xfeat_wrap = XFeat(device=opt['device'])
            xfeat_wrap.init_net(model.xfeat)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            xfeat_wrap.eval()
            print('xfeat initialized')



    print('Avg init ncc {:.4f}, eval ncc {:.4f}, best ncc {:.4f}'.format(init_ncc.avg, eval_ncc.avg, best_ncc))

if __name__ == '__main__':

    opt = {
        'logs_path': './logs',       # path to save logs
        'save_freq': 2,              # save model every save_freq epochs
        'n_checkpoints': 6,          # number of checkpoints to keep
        'power': 0.9,                # decay power
        'num_workers': 0,            # number of workers for data loading
    }

    parser = argparse.ArgumentParser(description = "cardiac")
    parser.add_argument("-m", "--model", type = str, default = 'tiramisuAndXfeatComplex')
    parser.add_argument("-bs", "--batch_size", type = int, default = 60)
    parser.add_argument("-d", "--dataset", type = str, default = 'ShgBfReg')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("-dp", "--datasets_path", type = str, default = "/home/jackywang/Documents/Datasets")
    parser.add_argument("--img_size", type = str, default = '(832,832)')
    parser.add_argument("--load_ckpt", type = str, default = "none") # best, last or epoch
    parser.add_argument("--n_iters", type = int, default = 60) # best, last or epoch
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--sim_w", type = float, default = 1.)
    parser.add_argument("--reg_w", type = float, default = 1.)
    parser.add_argument("--is_first_half", type = int, default = 1)

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}
    opt['img_size'] = eval(opt['img_size'])

    run(opt)

'''
python test_ShgBfReg_testset_io.py -m tiramisuAndXfeatComplex -bs 1 --gpu_id 0 --load_ckpt none --is_first_half 1 -dp PATH ti_pretrained=0 xf_pretrained=0
python test_ShgBfReg_testset_io.py -m tiramisuAndXfeatComplex -bs 1 --gpu_id 0 --load_ckpt none --is_first_half 0 -dp PATH ti_pretrained=0 xf_pretrained=0

'''
