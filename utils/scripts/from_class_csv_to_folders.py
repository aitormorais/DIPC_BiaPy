import sys
import os
import argparse
import shutil
import pandas as pd
from tqdm import tqdm 

parser = argparse.ArgumentParser(description="Organize classification data",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-image_dir", "--image_dir", help="Image directory")
parser.add_argument("-csv", "--csv", help="CSV file with the class of each image.")
parser.add_argument("-out_dir", "--out_dir",help="Output directory to organize the images into")
args = vars(parser.parse_args())

df = pd.read_csv(args['csv'])

if 'filename' not in df.columns or 'label' not in df.columns:
    raise ValueError("CSV file need to contain at least two columns: 'filename' and 'label'")

for index, row in tqdm(df.iterrows()):
    dest_folder = os.path.join(args['out_dir'], str(row['label']).replace(" ", "_").lower())
    os.makedirs(dest_folder, exist_ok=True)

    filename = os.path.join(args['image_dir'], row['filename'])
    print("Copying file {} to {}".format(filename, dest_folder))
    shutil.copy(filename, dest_folder)