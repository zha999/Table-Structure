from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import tkinter.simpledialog
import tkinter as tk  
import os
import cv2 as cv
import json
import WinPyQt5
from PyQt5 import QtCore, QtGui, QtWidgets,QtWebEngineWidgets
import sys
import codecs

def get_tmp_dir():
    cwd = os.getcwd()
    tmp_dir = os.path.join(cwd, "tmp")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    return tmp_dir

def get_tmp_segfile():
    segfile = os.path.join(get_tmp_dir(), "seg_label.json")
    return segfile

def save_seg_proportions(x_prop, y_prop):
    tmp_segfile = get_tmp_segfile()
    seg_lable_dict = {}
    seg_lable_dict['x_attribute'] = x_prop
    seg_lable_dict['y_header'] = y_prop
    with open(tmp_segfile, 'w') as f:
        tmp_segfile.dump(seg_lable_dict, f)

def get_seg_proportions(seg_file):
    with open(seg_file, 'r') as f:
        json_file = json.load(f)
        x_proportion = json_file['x_attribute']
        y_proportion = json_file['y_header']
    return x_proportion, y_proportion


# get image size
sample_file = r"E:\cnn_segment\dataset_add\Test\img\page_1_tbl_0.png"
seg_file = r"E:\cnn_segment\dataset_add\Test\seg_label\page_1_tbl_0.json"
struct_res_file = r"E:\PyWindow\page_1_tbl_0.png.html"
chunck_file = r"E:\cnn_segment\dataset_add\Test\chunk\page_1_tbl_0.chunk"

if not os.path.exists(sample_file):
    assert("img_file_path not exists!")
img = cv.imread(sample_file, cv.IMREAD_COLOR)
if img is None:
    assert("fail to load image!")

img_height = img.shape[0]
img_width = img.shape[1]

root = Tk()
status_index = 0
# set window title
def update_window_title(status):
    switcher = {
        0: "cell recognition",
        1: "segmentation line",
        2: "table structure",
    }
    win_title = switcher.get(status, "table structure recognition")
    root.title(win_title)

update_window_title(status_index)

#setting up a tkinter canvas
canvas = Canvas(root, width=img_width, height=img_height)
background_image=tk.PhotoImage(file=sample_file)
canvas.pack(side = TOP)
image = canvas.create_image(0, 0, anchor=tk.NW, image=background_image)

#draw rects
def draw_rect():
    chunks = json.load(codecs.open(chunck_file, 'r', 'utf-8-sig'))['chunks']
    for i in range(len(chunks)):
        bbox = chunks[i]['pos']
        if len(bbox) is not 4:
            continue
        p1_x = bbox[0]
        p1_y = img_height - bbox[2]
        p2_x = bbox[1]
        p2_y = img_height - bbox[3]
        canvas.create_rectangle(p1_x, p1_y, p2_x, p2_y, outline='green')


# draw seg lines
seg_lines = {'X': "0", 'Y': "0"}
def draw_first_seg_line():
    proportion_x, proportion_y = get_seg_proportions(seg_file)
    point_x = proportion_x*img_width
    point_y = proportion_y*img_height
    global seg_lines
    seg_lines['X'] = canvas.create_line(0, point_y, img_width, point_y, fill="red")
    seg_lines['Y']= canvas.create_line(point_x, 0, point_x, img_height, fill="red")

def printcoords(event):
    point_x = event.x
    point_y = event.y
    print(event.x, event.y)
    global seg_lines
    canvas.delete(seg_lines['X'])
    canvas.delete(seg_lines['Y'])
    seg_lines['X'] = canvas.create_line(0, point_y, img_width, point_y, fill="red")
    seg_lines['Y'] = canvas.create_line(point_x, 0, point_x, img_height, fill="red")
    save_seg_proportions(point_x/img_width, point_y/img_height)

def show_structure_result():
    global struct_res_file
    if not os.path.exists(struct_res_file):
        return
    # top = Toplevel(root)
    # top.title("result")
    # top.mainloop()
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = WinPyQt5.Ui_Dialog()
    ui.setupUi(Dialog, struct_res_file)
    Dialog.show()
    app.exec_()
    # sys.exit(app.exec_())

# hook the mouseclick event
canvas.bind("<Button 1>", printcoords)

# add a frame
frame = Frame(root, background="blue", width=img_width, height=200)
# hook command to buttons
def stop_button():
    exit(0)

def next_step():
    global status_index
    update_window_title(status_index)

    if status_index==0:
        draw_rect()
    elif status_index==1:
        draw_first_seg_line()
    else:  # status_index==2
        show_structure_result()
    # else:
    #     return
    
    status_index += 1


# add buttons in frame
button01 = Button(frame,text = "Stop", command=stop_button).pack(side = LEFT)
button02 = Button(frame,text = "Next", command=next_step).pack(side = LEFT)
frame.pack(fill=BOTH, side = TOP)


# mainloop
root.mainloop()