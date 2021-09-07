
import os
import shutil
from table_ui.table_ui import Table_UI, batch_process
import argparse

sample_file = r"E:\cnn_segment\dataset_add\Test\img\page_1_tbl_0.png"
sample_dir = r"E:\PyWindow\Test"

parser = argparse.ArgumentParser()
parser.add_argument('--ui', action='store_true', help='whether prase the table with UI')




if __name__=="__main__":
    print("start table recognition.")

    use_Ui = True # False  True

    if use_Ui:
        ui = Table_UI(sample_file)
    else:
        batch_process(sample_dir)
    
    print("end.")




