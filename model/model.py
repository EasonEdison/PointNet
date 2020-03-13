import torch
import torch.nn as nn
# active function
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class TNet_3d(nn.Module):
    def __init__(self):
        super(TNet_3d,self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # 32 100 3
        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class TNet_kd(nn.Module):
    def __init__(self, k=64):
        super(TNet_kd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class Base_Feature(nn.Module):
    def __init__(self,seg=False,feature_trans=False):
        super(Base_Feature,self).__init__()
        # only one dimention
        # this eq MLP
        self.seg = seg
        self.feature_trans = feature_trans

        self.conv_1 = nn.Conv1d(3,64,1)
        self.bn_1 = nn.BatchNorm1d(64)

        self.conv_2 = nn.Conv1d(64,128,1)
        self.bn_2 = nn.BatchNorm1d(128)

        self.conv_3 = nn.Conv1d(128,1024,1)
        self.bn_3 = nn.BatchNorm1d(1024)

        self.TNet_3d = TNet_3d()
        self.TNet_kd = TNet_kd()

    def forward(self,x):
        # batch 3 num_point
        n_points = x.size()[2]
        trans_point = self.TNet_3d(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x,trans_point)
        x = x.transpose(2, 1)
        # 3->64
        x = F.relu(self.bn_1(self.conv_1(x)))
        trans_feature = None
        if self.feature_trans:
            trans_feature = self.TNet_kd(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feature)
            x = x.transpose(2, 1)

        to_seg_feature = x
        # 64->128
        x = F.relu(self.bn_2(self.conv_2(x)))
        # 128->1024
        x = F.relu(self.bn_3(self.conv_3(x)))
        # max pool
        global_feature = torch.max(x, dim=2)[0].view((-1, 1024))
        if self.seg:
            # 注意计算维度
            global_feature = global_feature[:, :, None].repeat((1, 1, n_points))

            out_put = torch.cat((to_seg_feature,global_feature), dim=1)
            return out_put, trans_point, trans_feature
        else:
            return global_feature, trans_point, trans_feature




class Classfier(nn.Module):
    def __init__(self, classes):
        super(Classfier,self).__init__()
        self.global_feature = Base_Feature()
        self.seg = False
        self.k = classes
        # end is fc
        self.fc1 = nn.Linear(1024,512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256,self.k)
        # forget dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x, trans_point, trans_feature = self.global_feature(x)
        x = F.relu(self.bn1(self.fc1(x)))
        # in fc2 dropout,but I do not know wht
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.log_softmax(self.fc3(x),dim=1)
        return x, trans_point, trans_feature


class Segment(nn.Module):
    def __init__(self, classes):
        super(Segment, self).__init__()
        self.m = classes
        self.seg = True
        self.base_feature = Base_Feature(seg=self.seg)

        self.conv1 = nn.Conv1d(64+1024, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, self.m, 1)

    def forward(self, x):
        x = x.transpose(2, 1)
        batch_size = x.size()[0]
        num_points = x.size()[2]
        x, trans_point, trans_feature = self.base_feature(x)
        # 卷积的通道要放在第几个位置？难道是根据维度自动寻找的？
        # 答：并不能自动寻找
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # batch m n
        x = self.conv4(x).contiguous()
        # 这块分数还不懂
        x = F.log_softmax(x,dim=1).transpose(2,1)

        return x, trans_point, trans_feature

def get_trans_loss(trans):
    trans_T = trans.transpose(2, 1)
    # use eye() not diag
    # not need repeat have gb
    I = torch.eye(trans.size()[1])[None, :, :].cuda()
    AA_T = torch.bmm(trans, trans_T)
    # f1
    # need mean
    loss = torch.mean(torch.norm((I - AA_T), dim=(1, 2)))

    return loss


#
# if __name__ == '__main__':
#     test_cls = Variable(torch.rand([2,3,100]))
#
#     base  = Base_Feature()
    # result_base = base(test_cls)
    # print('base feature size',result_base[0].size())
    #
    # clas = Classfier(5)
    # result_cls = clas(test_cls)
    # print('result cls size', result_cls[0].size())
    #
    # seg = Segment(5)
    # result_seg = seg(test_cls)
    # print('result seg size', result_seg[0].size())
    #
    #
    #
