import json

import SimpleITK as sitk
import numpy
import numpy as np
from glob import glob
import os
from loguru import logger
from radiomics import featureextractor



def get_bbox_fast(np_lbl, borders=0):
    """
    :param np_lbl: ????mask
    :return: mask?bbox? (ndim, 2)
    """
    if not np.any(np_lbl):
        return None
    ndim = np_lbl.ndim
    if isinstance(borders, int):
        borders = [borders] * ndim
    borders = np.array(borders)
    bbox = []
    for dim in range(ndim):
        axis = list(range(ndim))
        axis.pop(dim)
        nonzero_idx = np.nonzero(np.any(np_lbl, axis=tuple(axis)))[0]
        bbox.append([nonzero_idx.min(), nonzero_idx.max()])
    bbox = np.array(bbox)
    bbox[:, 0] = np.maximum(0, bbox[:, 0] - borders)
    bbox[:, 1] = np.minimum(np.array(np_lbl.shape) - 1, bbox[:, 1] + borders)
    return bbox


def write_arr_to_nii(f, arr, type, spacing, verbose):
    simage = sitk.GetImageFromArray(arr.astype(type))
    simage.SetSpacing(spacing)
    if verbose:
        sitk.WriteImage(simage, f)
    return simage


def task(f, verbose=False):
    try:
        dirname = os.path.dirname(f)
        simage = sitk.ReadImage(f)
        raw_arr = sitk.GetArrayFromImage(simage)
        logger.info("raw arr shape: {}".format(raw_arr.shape))

        spacexyz = simage.GetSpacing()
        for maskf in sorted(glob(os.path.join(dirname, "*.nii.gz"))):
            if "crop" in maskf:
                continue
            nodule_mask = sitk.GetArrayFromImage(sitk.ReadImage(maskf))
            bbox = get_bbox_fast(nodule_mask)
            crop_raw = raw_arr[tuple(map(slice, bbox[:, 0], bbox[:, 1] + 1))]
            crop_mask = nodule_mask[tuple(map(slice, bbox[:, 0], bbox[:, 1] + 1))]
            # logger.info("{} crop_raw shape: {}, spacexyz:{}".format(os.path.basename(maskf)[:-7], crop_mask.shape, spacexyz))
            # crop_raw = zoom(crop_raw, np.array(spacexyz[::-1]), order=1)
            # crop_mask = zoom(crop_mask, np.array(spacexyz[::-1]), order=0)

            simage_raw = write_arr_to_nii(maskf.replace(".nii.gz", "_crop_raw.nii.gz"), crop_raw, np.float32, spacexyz, verbose)
            simage_mask = write_arr_to_nii(maskf.replace(".nii.gz", "_crop_mask.nii.gz"), crop_mask, np.uint8, spacexyz, verbose)
            # logger.info("{} resized crop_raw shape: {}".format(os.path.basename(maskf)[:-7], crop_mask.shape))
            settings = {}
            settings['resampledPixelSpacing'] = [1, 1, 1]  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
            settings['interpolator'] = sitk.sitkBSpline

            extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
            radio_result = extractor.execute(simage_raw, simage_mask)
            for k, v in radio_result.items():
                if isinstance(v, (int,str,list, tuple, float)):
                    continue
                elif type(v) == numpy.ndarray:
                    radio_result[k] = float(v)

            with open(maskf.replace(".nii.gz", ".json"), "w", encoding="utf-8") as f:
                json.dump(radio_result, f, indent=4)
    except:
        print(f)


if __name__ == "__main__":
    from multiprocessing import Pool
    from tqdm import tqdm
    nrrds = sorted(glob("D://yeyuxiang_workdir//data//lidc_data//feature_engineering//new_nrrd_2//*//*.nrrd"))
    for i in tqdm(nrrds):
        task(i)

    # logger.critical("nrrds len : {}".format(len(nrrds)))
    # pool = Pool(8)
    # results = pool.starmap_async(task, [(f, ) for f in nrrds]).get()
    # pool.join()
    # pool.close()