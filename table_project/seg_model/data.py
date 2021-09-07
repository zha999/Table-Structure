import random
import cv2
import numpy as np
import os
import codecs
from shapely.geometry import Point, Polygon
import torch
from torch.utils.data import Dataset,DataLoader
import torch_geometric.transforms as GT
import json


class Seg_Dataset(Dataset):
    def __init__(self, root_path, img_size):
        super(Seg_Dataset, self).__init__()
        self.root_path = root_path
        jsonfile = os.path.join(self.root_path, "imglist.json")
        self.img_size = img_size

        if os.path.exists(jsonfile):
            with open(jsonfile, "r") as read_file:
                self.imglist = json.load(read_file)
        else:  
            self.imglist = list(filter(lambda fn:fn.lower().endswith('.jpg') or fn.lower().endswith('.png') ,
                                       os.listdir(os.path.join(self.root_path,"img"))))
            with open(jsonfile, "w") as write_file:
                json.dump(self.imglist, write_file)

    def __len__(self):
        return len(self.imglist)

    def if_seg_valid(self, x_line, y_line):
        if  x_line > 0 and x_line < 1 and y_line > 0 and y_line < 1:
            return True
        return False

    def get_lable(self, base_name):
        seg_lines = os.path.join(self.root_path,"seg_label", base_name + ".json")
        if not os.path.exists(seg_lines):
            return 1,1
        
        with open(seg_lines, 'r') as f:
            segLines = json.load(f)
        seg_att = segLines["x_attribute"]
        seg_header = 0.8 # segLines["y_header"]
        
        is_valid = self.if_seg_valid(seg_att, seg_header)
        if is_valid == False:
            print("seg_att or seg_header is not valid")
            return 1, 1

        return seg_att, seg_header


    def __getitem__(self, idx):
        imgfn = self.imglist[idx]
        base_name = os.path.basename(imgfn)
        file_name, _ = os.path.splitext(base_name)
        imgfn = os.path.join(self.root_path,"img", file_name + ".png")
        img = cv2.imread(imgfn)
        if not img is None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = cv2.dilate(img,self.kernel, iterations = 1)
            img = cv2.resize(img, (self.img_size,self.img_size), interpolation = cv2.INTER_AREA) 
        
        x_att, y_header = self.get_lable(file_name)

        x_att = x_att * 10
        y_header = y_header * 10

        img = torch.FloatTensor(img/255.0).unsqueeze(0) #torch.IntTensor(img)
        # print("img data size {}".format(img.size()))

        x_att = torch.as_tensor(x_att)
        y_header = torch.as_tensor(y_header)

        data = {'image': img, 'att':x_att, 'header': y_header}
        return data



if __name__ == "__main__":  
    print("starting main")
    root_path = r'E:\cnn_segment\dataset_att\Test'
    
    ds = Seg_Dataset(root_path)
    # print(len(ds))

    test_loader = DataLoader(ds, batch_size=5)
    for data in test_loader:
        # print("data:{}".format(data))
        img = data['image']
        print("image:{}".format(img))
        print("image size:{}".format(img.size()))
        print("att:{}".format(data['att']))
        print("header:{}".format(data['header']))
