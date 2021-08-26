import os
import torch
from seg_model.model import Seg_CNN
from seg_model.data import Seg_Dataset
from torch.utils.data import DataLoader
import json
import codecs
import cv2

img_size = 64

def predict_seg(model, data):
    preds = model(data)
    preds = preds.squeeze(1)
    p_value = preds.item() / 10
    return p_value

def save_seg_files(att_line, header_line, save_path):
    if att_line>1 or att_line < 0 or header_line>1 or header_line<0:
        assert(False, "input not valid!")
        return
    seg_lable_dict = {'x_attribute': att_line, 'y_header':header_line}
    with open(save_path, 'w') as f:
        json.dump(seg_lable_dict, f)

def generate_seg_file(root_path):
    dataset = Seg_Dataset(root_path, img_size)
    data_loader = DataLoader(dataset, batch_size=1)
    data_iter = iter(data_loader)
    dataset_length = len(data_loader)

    Y_model = Seg_CNN(img_size)
    Y_model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "Y_net_model.pth")
    Y_model.load_state_dict(torch.load(Y_model_path, map_location=torch.device('cpu')))
    
    X_model = Seg_CNN(img_size)
    X_model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "X_net_model.pth")
    X_model.load_state_dict(torch.load(X_model_path, map_location=torch.device('cpu')))

    img_list_path = os.path.join(root_path, "imglist.json")
    img_list = json.load(codecs.open(img_list_path, 'r', 'utf-8-sig'))
    img_list_len = len(img_list)
    assert(dataset_length == img_list_len)

    for i in range(dataset_length):
        current_data = data_iter.next()
        
        line_y = predict_seg(Y_model, current_data)
        print(line_y)
        
        line_x = predict_seg(X_model, current_data)
        print(line_x)

        img_name = img_list[i]
        img_path = os.path.join(root_path, "img", img_name)
        if not os.path.exists(img_path):
            continue
        
        file_name = img_name.split(".")[0]
        seg_file_path = os.path.join(root_path, "seg_label", file_name+".json")
        save_seg_files(line_x, line_y, seg_file_path)
        

if __name__=="__main__":
    print("hhhhhh")
    root_path = r"E:\PyWindow\tmp"
    generate_seg_file(root_path)

