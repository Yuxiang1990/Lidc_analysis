from glob import glob
import pandas as pd
from loguru import logger
import json
import os
import re

jsonfiles = sorted(glob("D://yeyuxiang_workdir//data//lidc_data//feature_engineering//new_nrrd_2//*//*.json"))
logger.info(len(jsonfiles))

maglinant_df = pd.read_csv("D://yeyuxiang_workdir//data//lidc_data//feature_engineering//characteristics_new.csv")


def task(f):
    feature_dict = {}
    logger.info(f)
    basename = os.path.basename(f).replace(".json", "")
    feature_dict['basename'] = basename
    with open(f, "r") as f:
        data = json.load(f)
        print(data)
    for k, v in data.items():
        if re.search(re.compile(r'(shape)|(firstorder)|(glcm)|(gldm)|(glrlm)|(glszm)|(ngtdm)'), k):
            print(k, v)
            feature_dict[k.replace("original_", "")] = float(v)
    pid, sid, Nodule_Str, *_ = basename.split("_")
    malignant_label = maglinant_df.loc[(maglinant_df.Patient_ID == pid) &
                     (maglinant_df.Session_ID == sid) &
                     (maglinant_df.Nodule_Str == Nodule_Str.lstrip('0')), "malignancy"].values[0]
    feature_dict['malignancy_label'] = float(malignant_label)
    df = pd.DataFrame([feature_dict])
    return df


if __name__ == "__main__":
    from tqdm import tqdm
    df_list = []
    for f in tqdm(jsonfiles):
        df = task(f)
        df_list.append(df)

    df_all = pd.concat(df_list)
    df_all = df_all.sort_values("basename")
    df_all.to_csv("D://yeyuxiang_workdir//data//lidc_data//feature_engineering//radiomics_features.csv", index=None)