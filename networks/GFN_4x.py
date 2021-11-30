import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F
import os

################################################################################
# Commons

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inpu, adj):
        support = torch.matmul(inpu, self.weight)
        #print(adj.size())
       #print(support.size())
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def get_Adj(adj_file):
    import scipy.io as spio
    data = spio.loadmat(adj_file)
    data = data['FULL'].astype(np.float32)
    return data

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class ResGCN(nn.Module):
    def __init__(self, features, adj):
        super(ResGCN, self).__init__()
        self.adj = adj
        self.A = nn.Parameter(torch.from_numpy(self.adj).float())
        self.relu = nn.LeakyReLU(0.2)
        self.graph_conv1 = GraphConvolution(features, features)
        self.graph_conv2 = GraphConvolution(features, features)

    def forward(self, inpu):
        adj = gen_adj(self.A).detach()
        res_g = self.graph_conv1(inpu, adj)
        res_g = self.relu(res_g)
        res_g = self.graph_conv1(res_g, adj)
        return inpu + res_g





################################################################################

class _ResBLockDB(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBLockDB, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=True)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out

class _ResBlockSR(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBlockSR, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=True)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out

class _DeblurringMoudle(nn.Module):
    def __init__(self):
        super(_DeblurringMoudle, self).__init__()
        self.conv1     = nn.Conv2d(3, 64, (7, 7), 1, padding=3)
        self.relu      = nn.LeakyReLU(0.2, inplace=True)
        self.resBlock1 = self._makelayers(64, 64, 6)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), 2, 1),
            nn.ReLU(inplace=True)
        )
        self.resBlock2 = self._makelayers(128, 128, 6)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), 2, 1),
            nn.ReLU(inplace=True)
        )
        self.resBlock3 = self._makelayers(256, 256, 6)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (4, 4), 2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4, 4), 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (7, 7), 1, padding=3)
        )
        self.convout = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, (3, 3), 1, 1)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        con1   = self.relu(self.conv1(x))
        res1   = self.resBlock1(con1)
        res1   = torch.add(res1, con1)
        con2   = self.conv2(res1)
        res2   = self.resBlock2(con2)
        res2   = torch.add(res2, con2)
        con3   = self.conv3(res2)
        res3   = self.resBlock3(con3)
        res3   = torch.add(res3, con3)
        decon1 = self.deconv1(res3)
        deblur_feature = self.deconv2(decon1)
        deblur_out = self.convout(torch.add(deblur_feature, con1))
        # print(deblur_feature.size(), deblur_out.size())
        return deblur_feature, deblur_out

class _SRMoudle(nn.Module):
    def __init__(self):
        super(_SRMoudle, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (7, 7), 1, padding=3)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resBlock = self._makelayers(64, 64, 8, 1)
        self.conv2 = nn.Conv2d(64, 64, (3, 3), 1, 1)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBlockSR(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        con1 = self.relu(self.conv1(x))
        res1 = self.resBlock(con1)
        con2 = self.conv2(res1)
        sr_feature = torch.add(con2, con1)
        return sr_feature

class _GateMoudle(nn.Module):
    def __init__(self):
        super(_GateMoudle, self).__init__()

        self.conv1 = nn.Conv2d(131,  64, (3, 3), 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 64, (1, 1), 1, padding=0)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        con1 = self.relu(self.conv1(x))
        scoremap = self.conv2(con1)
        return scoremap

class _ReconstructMoudle(nn.Module):
    def __init__(self):
        super(_ReconstructMoudle, self).__init__()
        self.resBlock = self._makelayers(64, 64, 8)
        self.conv1 = nn.Conv2d(64, 256, (3, 3), 1, 1)
        self.pixelShuffle1 = nn.PixelShuffle(2)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(64, 256, (3, 3), 1, 1)
        self.pixelShuffle2 = nn.PixelShuffle(2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, 1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(64, 3, (3, 3), 1, 1)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        res1 = self.resBlock(x)
        con1 = self.conv1(res1)
        pixelshuffle1 = self.relu1(self.pixelShuffle1(con1))
        con2 = self.conv2(pixelshuffle1)
        pixelshuffle2 = self.relu2(self.pixelShuffle2(con2))
        con3 = self.relu3(self.conv3(pixelshuffle2))
        sr_deblur = self.conv4(con3)
        return sr_deblur

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.deblurMoudle      = self._make_net(_DeblurringMoudle)
        self.srMoudle          = self._make_net(_SRMoudle)
        self.geteMoudle        = self._make_net(_GateMoudle)
        self.reconstructMoudle = self._make_net(_ReconstructMoudle)

    def forward(self, x, gated, isTest):
        if isTest == True:
            origin_size = x.size()
            input_size  = (math.ceil(origin_size[2]/4)*4, math.ceil(origin_size[3]/4)*4)
            out_size    = (origin_size[2]*4, origin_size[3]*4)
            x           = nn.functional.upsample(x, size=input_size, mode='bilinear')
        
        # print("input", x.size())
        deblur_feature, deblur_out = self.deblurMoudle(x)
        # print("deblur_feature", deblur_feature.size()," | " , deblur_out.size())
        
        sr_feature = self.srMoudle(x)
        if gated == True:
            scoremap = self.geteMoudle(torch.cat((deblur_feature, x, sr_feature), 1))
        else:
            scoremap = torch.cuda.FloatTensor().resize_(sr_feature.shape).zero_()+1
        repair_feature = torch.mul(scoremap, deblur_feature)
        fusion_feature = torch.add(sr_feature, repair_feature)
        recon_out = self.reconstructMoudle(fusion_feature)

        if isTest == True:
            recon_out = nn.functional.upsample(recon_out, size=out_size, mode='bilinear')

        return deblur_out, recon_out

    def _make_net(self, net):
        nets = []
        nets.append(net())
        return nn.Sequential(*nets)

class _Deblur_model(nn.Module):
    def __init__(self, conv=default_conv):
        super(_Deblur_model, self).__init__()
        #Typically, it needs 2 downsample layers for deblurring
        n_resblocks = 6  #number of resblocks is each stage
        n_filters = 128 #number of filters is each stage
        n_graph_features = 32 #number of graph features
        n_graph_layers = 5
        kernel_size = 3 
        act = nn.ReLU(True)
        adj_dir = "./Adj_matrix/deblur_2_full.mat"
        n_ResGCN = 5
        rgb_range = 255
        n_colors = 3

        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_filters, kernel_size)]

        # define body module 
        # encoder_1 -> encoder_2 -> encoder_3 -> GNN -> decoder_3 -> decoder_2 -> decoder_1
        encoder_1 = [ResBlock(conv, n_filters, 
                kernel_size, act=act, res_scale=1) for _ in range(n_resblocks)]
        self.downsample_1 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=2, padding=1)

        encoder_2 = [ResBlock(conv, n_filters, 
                kernel_size, act=act, res_scale=1) for _ in range(n_resblocks)]
        self.downsample_2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=2, padding=1)

        encoder_3 = [ResBlock(conv, n_filters, 
                kernel_size, act=act, res_scale=1) for _ in range(n_resblocks)]
        self.downsample_graph = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=10, padding=1)


        _adj = get_Adj(adj_file=adj_dir)
        self.A = nn.Parameter(torch.from_numpy(_adj).float())


        GCN_body = [
            ResGCN(
                n_graph_features, _adj
            ) for _ in range(n_ResGCN)
        ]
        self.graph_convhead = GraphConvolution(1, n_graph_features)
        self.graph_convtail = GraphConvolution(n_graph_features, 1)


        self.upsample_graph = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=kernel_size, stride=8, padding=1, output_padding=7)
        decoder_3 = [ResBlock(conv, n_filters, 
                kernel_size, act=act, res_scale=1) for _ in range(n_resblocks)]

        self.upsample_2 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        decoder_2 = [ResBlock(conv, n_filters, 
                kernel_size, act=act, res_scale=1) for _ in range(n_resblocks)]

        self.upsample_1 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        decoder_1 = [ResBlock(conv, n_filters, 
                kernel_size, act=act, res_scale=1) for _ in range(n_resblocks)]

        m1_tail = [conv(n_filters, 64, kernel_size)]

        # define tail module
        m_tail = [conv(n_filters, n_colors, kernel_size)]

        self.relu = nn.LeakyReLU(0.2)
        self.head = nn.Sequential(*m_head)
        self.encoder_1 = nn.Sequential(*encoder_1)
        self.encoder_2 = nn.Sequential(*encoder_2)
        self.encoder_3 = nn.Sequential(*encoder_3)
        self.GCN_body = nn.Sequential(*GCN_body)

        #self.graph_conv = nn.Sequential(*graph_conv)
        self.decoder_3 = nn.Sequential(*decoder_3)
        self.decoder_2 = nn.Sequential(*decoder_2)
        self.decoder_1 = nn.Sequential(*decoder_1)

        self.m1_tail = nn.Sequential(*m1_tail)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #res = self.sub_mean(x)
        res = self.head(x)

        res_enc1 = self.encoder_1(res)
        res_down1 = self.downsample_1(res_enc1)
        res_enc2 = self.encoder_2(res_down1)
        res_down2 = self.downsample_2(res_enc2)
        res_enc3 = self.encoder_3(res_down2)

        print("input", x.size())
        print("encoders", res_enc1.size(), res_enc2.size(), res_enc3.size())

        yin = self.downsample_graph(res_enc3)
        
        yin = yin.permute(0,2,3,1)
        yin = yin.unsqueeze(4)

        adj = gen_adj(self.A).detach()
        # print(adj.size())
        res_g = self.graph_convhead(yin, adj) #make(yin, adj) as dict to let it available in nn.Sequential
        res_g = self.GCN_body(res_g)
        res_g = self.graph_convtail(res_g, adj)
        res_g = self.relu(res_g)
    

        res_g = res_g.squeeze(4)
        yout = res_g.permute(0,3,1,2)
        
        print("Graph i/p,o/p", yin.size(), yout.size())

        res_up3 = self.upsample_graph(yout)
        print("up_3", res_up3.size())
        res_dec3 = self.decoder_3(res_up3 + res_enc3)
        print("decoder_3", res_dec3.size())
        res_up2 = self.upsample_2(res_dec3)
        print("up_2", res_up2.size())
        res_dec2 = self.decoder_2(res_up2 + res_enc2)
        print("decoder_2", res_dec2.size())
        res_up1 = self.upsample_1(res_dec2)
        print("up_1", res_up1.size())
        res_out = self.decoder_1(res_up1 + res_enc1)
        print("res_out", res_out.size())

        res_out1 = self.m1_tail(res_out)
        # print("res_out1", res_out1.size())
        res = self.tail(res_out)
        #res = self.add_mean(res)
        res += x

        # print(res_dec2.size(), res_up1.size(), res.size())
        return res_out1, res


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.deblurMoudle      = self._make_net(_Deblur_model)
        self.srMoudle          = self._make_net(_SRMoudle)
        self.geteMoudle        = self._make_net(_GateMoudle)
        self.reconstructMoudle = self._make_net(_ReconstructMoudle)

    def forward(self, x, gated, isTest):
        if isTest == True:
            origin_size = x.size()
            input_size  = (math.ceil(origin_size[2]/4)*4, math.ceil(origin_size[3]/4)*4)
            out_size    = (origin_size[2]*4, origin_size[3]*4)
            x           = nn.functional.upsample(x, size=input_size, mode='bilinear')

        # print("input", x.size())
        deblur_feature, deblur_out = self.deblurMoudle(x)
        # print("deblur_feature", deblur_feature.size()," | " , deblur_out.size())
        sr_feature = self.srMoudle(x)
        if gated == True:
            scoremap = self.geteMoudle(torch.cat((deblur_feature, x, sr_feature), 1))
        else:
            scoremap = torch.cuda.FloatTensor().resize_(sr_feature.shape).zero_()+1
        repair_feature = torch.mul(scoremap, deblur_feature)
        fusion_feature = torch.add(sr_feature, repair_feature)
        recon_out = self.reconstructMoudle(fusion_feature)

        if isTest == True:
            recon_out = nn.functional.upsample(recon_out, size=out_size, mode='bilinear')

        return deblur_out, recon_out

    def _make_net(self, net):
        nets = []
        nets.append(net())
        return nn.Sequential(*nets)




