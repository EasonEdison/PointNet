import torch.utils.data as data
import os
import torch
import numpy as np
import json

class ShapeNet_dataload(data.DataLoader):
    def __init__(self,
                 num_points,
                 seg,
                 root_dir,
                 split,
                 cls_choice=None,
                 augment=False
                 ):
        self.augment = augment
        self.root = root_dir
        self.split = split
        self.num_points = num_points
        self.seg = seg
        self.cat = {}
        self.choice = cls_choice
        # set id to class
        cat_file = os.path.join(self.root,'synsetoffset2category.txt')
        with open(cat_file,'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if not self.choice is None:
            # 注意，这里要用cat.items()
            # items会返回一个迭代器，迭代字典中每个内容
            a = self.cat.items()
            self.cat = {k:v for k,v in self.cat.items() if k in self.choice}

        self.id2cat = {k:v for v,k in self.cat.items()}
        # set all information
        self.meta = {}
        for item in self.cat:
            self.meta[item] = []

        all_file = os.path.join(self.root, 'train_test_split','shuffled_{}_file_list.json'.format(self.split))
        # json要用json.load+open
        file_list = json.load(open(all_file, 'r'))

        for line in file_list:
            _, id, file＿name = line.strip().split('/')
            if id in self.cat.values():
                # set item list pro
                self.meta[self.id2cat[id]].append((os.path.join(self.root,id,'points',file＿name+'.pts'),
                                                os.path.join(self.root,id,'points_label',file＿name+'.seg')))

        self.data = []
        for cls in self.cat:
            for list_2 in self.meta[cls]:
                # detail infor
                # append tuple
                self.data.append((cls,list_2[0],list_2[1]))

        # set cls id, to get loss
        self.cls_sorted = dict(zip(sorted(self.cat), range(len(self.cat))))
        # do seg
        self.num_seg_cls = {}
        seg_part_file = os.path.join('misc', 'num_seg_classes.txt')
        # only float can np.loadtxt
        with open(seg_part_file,'r') as f:
            for line in f:
                cls_name, cls_num = line.strip().split()
                self.num_seg_cls[cls_name] = int(cls_num)

    def __getitem__(self, index):
        cls_name, point_pre_file, point_target_file = self.data[index]
        cls_id = self.cls_sorted[cls_name]
        point_set = np.loadtxt(point_pre_file).astype(np.float32)
        seg = np.loadtxt(point_target_file).astype(np.float32)

        choice = np.random.choice(len(seg), self.num_points, replace=False)

        point_set = point_set[choice, :]
        seg_set = seg[choice]
        # 归一化处理忘掉了

        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            rotate_matrix = [[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotate_matrix)
            point_set = point_set + np.random.uniform(0, 0.01, size=point_set.shape())

        # trans to tensor
        point_set = torch.from_numpy(point_set)
        seg_set = torch.from_numpy(seg_set).long()
        # need change cls_id to array,add [cls_id] to be array
        cls_id = torch.from_numpy(np.array([cls_id]).astype(np.int64))


        if self.seg:
            return point_set, seg_set
        else:
            return point_set, cls_id
    # forget
    def __len__(self):
        return len(self.data)
