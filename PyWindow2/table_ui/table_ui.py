import sys
sys.path.append("/home/xxx/demo_test01/a")

from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import tkinter.simpledialog
import tkinter as tk  
import os
import table_ui.WinPyQt5
from PyQt5 import QtCore, QtGui, QtWidgets,QtWebEngineWidgets
import sys
import codecs
import cv2 as cv
import json
import shutil
from post_processor.post_proceed_v8 import post_proceed
from seg_model.generate_seg_file import generate_seg_file

class Table_UI(object):
    def __init__(self, path):
        if not os.path.exists(path):
            assert("img path not exists!")
            return
        # path in temp file
        self.input_img_Path  = path



        self.img = cv.imread(self.input_img_Path, cv.IMREAD_COLOR)
        self.img_height = self.img.shape[0]
        self.img_width = self.img.shape[1]

        self.status_index = 0
        self.seg_lines = {'X': "0", 'Y': "0"}   # id of the seg line

        self.init()
        self.show()

    def init(self):
        Table_UI.clear_tmp_dir()
        self.tmp_path = self.get_tmp_dir()
        self.tmp_seg_path = self.get_subfolder("seg_label")
        self.tmp_chunk_path = self.get_subfolder("chunk")
        self.tmp_img_path = self.get_subfolder("img")
        self.tmp_pred_structure_path = self.get_subfolder("pred_structure")
        self.tmp_table_html_path = self.get_subfolder("table_html")


        img_dir, filename = os.path.split(self.input_img_Path)
        name, extension = os.path.splitext(filename)
        base_dir = os.path.split(img_dir)[0]
        chunk_path = os.path.join(base_dir, "chunk", name+".chunk") # chunk seg_label img
        seg_path = os.path.join(base_dir, "seg_label", name+".json")
        
        self.img_file = os.path.join(self.tmp_img_path, name+".png")
        self.cell_chunk_file = os.path.join(self.tmp_chunk_path, name+".chunk")
        self.seg_label_file = os.path.join(self.tmp_seg_path, name + ".json")
        self.struct_res_file = os.path.join(self.tmp_table_html_path, name+".html")

        # Todo:
        # 1. structure file in tmp file shoule be generated from model
        shutil.copy(self.input_img_Path, self.img_file)
        shutil.copy(chunk_path, self.cell_chunk_file)
        # shutil.copy(seg_path, self.seg_label_file)
        generate_seg_file(self.tmp_path)


    def show(self):
        # UI
        self.root = Tk()
        self.canvas = Canvas(self.root, width=self.img_width, height=self.img_height)
        background_image=tk.PhotoImage(file=self.img_file)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=background_image)

        self.frame = Frame(self.root, background="blue", width=self.img_width, height=200)
        self.button01 = Button(self.frame,text = "Stop", command=self.stop_button).pack(side = LEFT)
        self.button02 = Button(self.frame,text = "Next", command=self.next_step).pack(side = LEFT)
        
        # hook the mouseclick event
        self.canvas.bind("<Button 1>", self.printcoords)
        self.canvas.pack(side = TOP)
        self.frame.pack(fill=BOTH, side = TOP)

        # mainloop
        self.root.mainloop()

    def draw_first_seg_line(self):
        proportion_x, proportion_y = self.get_seg_proportions()
        point_x = proportion_x*self.img_width
        point_y = proportion_y*self.img_height
        self.seg_lines['X'] = self.canvas.create_line(0, point_y, self.img_width, point_y, fill="red")
        self.seg_lines['Y']= self.canvas.create_line(point_x, 0, point_x, self.img_height, fill="red")

    def printcoords(self, event):
        point_x = event.x
        point_y = event.y
        print(event.x, event.y)
        self.canvas.delete(self.seg_lines['X'])
        self.canvas.delete(self.seg_lines['Y'])
        self.seg_lines['X'] = self.canvas.create_line(0, point_y, self.img_width, point_y, fill="red")
        self.seg_lines['Y'] = self.canvas.create_line(point_x, 0, point_x, self.img_height, fill="red")
        self.save_seg_proportions(point_x/self.img_width, point_y/self.img_height)

    def stop_button(self):
        exit(0)

    def next_step(self):
        self.update_window_title()

        if self.status_index==0:
            self.draw_rect()
        elif self.status_index==1:
            self.draw_first_seg_line()
        else:  # status_index==2
            self.show_structure_result()
        # else:
        #     return
        
        self.status_index += 1


    def draw_rect(self):
        chunks = json.load(codecs.open(self.cell_chunk_file, 'r', 'utf-8-sig'))['chunks']
        for i in range(len(chunks)):
            bbox = chunks[i]['pos']
            if len(bbox) != 4:
                continue
            p1_x = bbox[0]
            p1_y = self.img_height - bbox[2]
            p2_x = bbox[1]
            p2_y = self.img_height - bbox[3]
            self.canvas.create_rectangle(p1_x, p1_y, p2_x, p2_y, outline='green')

    def update_window_title(self):
        switcher = {
            0: "cell recognition",
            1: "segmentation line",
            2: "table structure",
        }
        win_title = switcher.get(self.status_index, "table structure recognition")
        self.root.title(win_title)
    
    @staticmethod
    def clear_tmp_dir():
        tmp_dir = Table_UI.get_tmp_dir()
        shutil.rmtree(tmp_dir)
        # os.removedirs(tmp_dir)

    @staticmethod
    def get_tmp_dir():
        cwd = os.getcwd()
        tmp_dir = os.path.join(cwd, "tmp")
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        return tmp_dir

    @staticmethod
    def get_subfolder(subfolder):
        if len(subfolder)<=0:
            return
        sub_Path = os.path.join(Table_UI.get_tmp_dir(), subfolder)
        if not os.path.exists(sub_Path):
            os.mkdir(sub_Path)
        return sub_Path

    def save_seg_proportions(self, x_prop, y_prop):
        seg_lable_dict = {'x_attribute': x_prop, 'y_header':y_prop}
        with open(self.seg_label_file, 'w') as f:
            json.dump(seg_lable_dict, f)

    def get_seg_proportions(self):
        with open(self.seg_label_file, 'r') as f:
            json_file = json.load(f)
            x_proportion = json_file['x_attribute']
            y_proportion = json_file['y_header']
        return x_proportion, y_proportion

    def show_structure_result(self):
        post_proceed(self.tmp_path)
        if not os.path.exists(self.struct_res_file):
            assert("struct_res_file path not exists!")
            return
        app = QtWidgets.QApplication(sys.argv)
        Dialog = QtWidgets.QDialog()
        ui = table_ui.WinPyQt5.Ui_Dialog()
        ui.setupUi(Dialog, self.struct_res_file)
        Dialog.show()
        app.exec_()




def batch_process(root_path):
    if not os.path.exists(root_path):
        assert(False, "root path not exists")
        return
    img_path = os.path.join(root_path, "img")
    chunk_path = os.path.join(root_path, "chunk")

    Table_UI.clear_tmp_dir()
    tmp_dir = Table_UI.get_tmp_dir()
    tmp_img_path = os.path.join(tmp_dir, "img")
    tmp_chunk_path = os.path.join(tmp_dir, "chunk")
    # tmp_img_path = Table_UI.get_subfolder("img")
    # tmp_chunk_path = Table_UI.get_subfolder("chunk")    
    tmp_seg_path = Table_UI.get_subfolder("seg_label")
    tmp_pred_struct_path = Table_UI.get_subfolder("pred_structure")
    tmp_table_html_path = Table_UI.get_subfolder("table_html")

    # copy img and chunk to tmp directory
    shutil.copytree(img_path, tmp_img_path)
    shutil.copytree(chunk_path, tmp_chunk_path) # need to be generated by model

    # generate seg_lines
    generate_seg_file(tmp_dir)

    # predict relationship and post-porcess
    post_proceed(tmp_dir)


