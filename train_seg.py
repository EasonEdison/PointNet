import torch
import numpy as np
from dataset.dataset import ShapeNet_dataload
import argparse
import torch.utils.data as data
from tqdm import tqdm
import torch.optim as optim
from dataset.dataset import ShapeNet_dataload
from model.model import Segment, get_trans_loss
import torch.nn.functional as F
import os


parse = argparse.ArgumentParser()
parse.add_argument('--batch_size', type=int, default=32)
parse.add_argument('--numpoints', type=int, default=100)
parse.add_argument('--workers', type=int, default=2)
parse.add_argument('--nepoch', type=int, default=100)
parse.add_argument('--root_dir', type=str)
parse.add_argument('--test_interval', type=int, default=20)
parse.add_argument('--out_path', type=str, default='model_save')
parse.add_argument('--feature_trans', action='store_true', default=False)
parse.add_argument('--choice', type=str, default='Chair')

args = parse.parse_args('--root_dir /home/ql-b423/sda/TXH/dataset/PointNet/ShapeNet'.split())


if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

train_seg_dataset = ShapeNet_dataload(
    num_points=args.numpoints,
    seg=True,
    root_dir=args.root_dir,
    split='train',
    cls_choice=args.choice
)
# num_workers need int()
train_seg_dataload = data.DataLoader(
    dataset=train_seg_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=int(args.workers)
)

test_seg_dataset = ShapeNet_dataload(
    num_points=args.numpoints,
    seg=True,
    root_dir=args.root_dir,
    split='test',
    cls_choice=args.choice
)

test_seg_dataload = data.DataLoader(
    dataset=test_seg_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=int(args.workers)
)

# define to train
num_seg_cls = train_seg_dataset.num_seg_cls[args.choice]
model = Segment(num_seg_cls).cuda()
# parameters need ()
optimier = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimier, step_size=20, gamma=0.1)


for epoch in range(args.nepoch):
    scheduler.step()
    for index, data in enumerate(train_seg_dataload):
        model = model.train()
        # this target is cls
        pred, target = data
        # change [32, 1] to [32]
        pred, target = pred.cuda(), target[:, :, None].cuda()
        optimier.zero_grad()
        # batch softmax num_point
        predic, trans_point, trans_feature = model(pred)
        # 让网络自己来学习分类对应
        # [M,N] 对应 [M]
        predic = predic.contiguous().view(-1, num_seg_cls)
        # need - 1 , because file start from 1
        target = target.contiguous().view(-1, 1)[:, 0] - 1
        loss = F.nll_loss(predic, target)
        if args.feature_trans:
            loss += get_trans_loss(trans_point)
        loss.backward()
        optimier.step()
        # need to [1]
        pred_result = torch.max(predic, dim=1)[1]
        # use cpu
        curr = target.eq(pred_result.data).sum().float() / (target.size()[0] + 1e-6)
        print('Epoch:%d loss:%f acc:%f' % (epoch, loss.item(), curr))

    if epoch % args.test_interval == 0:
        index, data = next(enumerate(test_cls_dataload))
        model = model.eval()
        pred, target = data
        pred, target = pred.cuda(), target.cuda()
        predic, _, _ = model(pred)
        pred_result = torch.max(predic, dim=1)[1]
        curr = target.eq(pred_result.data).sum().float() / (target.size()[0] + 1e-6)

        print('Test:%d acc:%f' % (index, curr))

    torch.save(model.state_dict(), '%s/model_cls_%d.pth' % (args.out_path, args.nepoch))


print('Train over---------------------')
total_right = 0.0
total_sample = 0.0
for epoch in tqdm(enumerate(test_cls_dataload)):
    model = model.eval()
    pred, target = data
    predic, _, _ = model(pred)
    pred_result = torch.max(pred, dim=1)
    total_right += target.eq(pred_result).cpu.sum()
    total_sample += (args.batch_size * predic.size()[1])

print('Test all acc:%f' % (total_right / total_sample))
