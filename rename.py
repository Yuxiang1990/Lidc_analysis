import pandas as pd
import shutil
import os



df = pd.read_csv("D:\yeyuxiang_workdir\data\lidc_data\manifest-1600709154662\metadata.csv")
print(df.shape)
basepath = "D:\yeyuxiang_workdir\data\lidc_data\manifest-1600709154662"
for index, row in df.iterrows():
    nowf = row["File Location"]
    SeriesUID = row['Series UID']
    StudyUID = row['Study UID']
    srcf = os.path.join(basepath, nowf)
    if os.path.exists(srcf):
        components = (srcf.split("/"))
        components[-1] = SeriesUID
        components[-2] = StudyUID
        print(components)
        shutil.move(srcf, "/".join(components))




