
import os
import shutil
from table_ui.table_ui import Table_UI, batch_process
import argparse

<<<<<<< HEAD
sample_file = "/Users/hxr/PycharmProjects/GFTE/Test/img/page_1_tbl_1.png"
sample_dir = "/Users/hxr/PycharmProjects/GFTE/Test"

parser = argparse.ArgumentParser()
parser.add_argument('--ui', action='store_true', help='whether prase the table with UI')
parser.add_argument('--if_cuda', action='store_true', help='whether use cuda')
opt = parser.parse_args()
=======
sample_file = r"E:\cnn_segment\dataset_add\Test\img\page_1_tbl_0.png"
sample_dir = r"E:\PyWindow\Test"

parser = argparse.ArgumentParser()
parser.add_argument('--ui', action='store_true', help='whether prase the table with UI')


>>>>>>> b549f23fbcd5bdb0522feb3cd975c2c0650a091a


if __name__=="__main__":
    print("start table recognition.")
<<<<<<< HEAD
    if(opt is None):
        use_Ui = False # False  True
        if_cuda = False
    else:
        use_Ui = opt.ui
        if_cuda  = opt.if_cuda
=======

    use_Ui = True # False  True
>>>>>>> b549f23fbcd5bdb0522feb3cd975c2c0650a091a

    if use_Ui:
        ui = Table_UI(sample_file)
    else:
<<<<<<< HEAD
        batch_process(sample_dir, if_cuda)
=======
        batch_process(sample_dir)
>>>>>>> b549f23fbcd5bdb0522feb3cd975c2c0650a091a
    
    print("end.")




