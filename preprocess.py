from typing import Dict
import numpy as np
import nibabel as nib
import hcp_utils as hcp
import glob
import os
import logging

logging.basicConfig(filename="preprocess.log", 
                    level=logging.DEBUG, 
                    format="[%(asctime)s] [%(levelname)s] : [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

MMP_PATH = "mmp_1.0.npz"
HCP_BASE = "/project/nblab/takeru.abe/data/hcp/HCP_1200"

def def_output_path(source: str) -> str:
    """
    Define the output path for the given source.

    :param source: The source filepath.
    :return: The output path as a string.
    """
    
    subject_id = source.split('/')[7] 
    session = source.split('/')[10].split('_')[1]  # Extracts REST1 or REST2
    phase_encoding = source.split('/')[10].split('_')[2]  # Extracts LR or RL
    
    assert len(subject_id) == 6, "Subject id must be 6 charactors"
    assert session in ["REST1", "REST2"], "Session must be ether REST1 or REST2"
    assert phase_encoding in ["LR", "RL"], "Phase encoding must be ether LR or RL"
    logger.info(f"Subject id: {subject_id}, Session: {session}, Phase Encoding: {phase_encoding}")
    
    output_dir = f"data/HCP/npz/{subject_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"HCP_visual_voxel_{session}_{phase_encoding}.npz"
    output_path = os.path.join(output_dir, output_filename)
    return output_path


def extract_visual_voxels(Xn: np.ndarray, labels: np.ndarray, map_all: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract visual voxels for regions ['V1', 'V2', 'V3', 'V4'].

    :param Xn: Normalized voxel data as a NumPy array.
    :param labels: Label data as a NumPy array.
    :param map_all: Mapping of all voxel regions.
    :return: A dictionary containing concatenated visual voxel data for specified regions.
    """
    
    roi_voxels = {}
    for region in ['V1', 'V2', 'V3', 'V4']:
        L_region_id = np.where(labels == f'L_{region}')[0][0]
        R_region_id = np.where(labels == f'R_{region}')[0][0]
        roi_voxels[region] = np.concatenate([Xn[:, map_all == L_region_id], Xn[:, map_all == R_region_id]], axis=1)
    
    
    assert roi_voxels['V1'].shape[1] == 1618, "Number of voxels for V1 must be 1618"
    assert roi_voxels['V2'].shape[1] == 1220, "Number of voxels for V1 must be 1220"
    assert roi_voxels['V3'].shape[1] == 684, "Number of voxels for V1 must be 684"
    assert roi_voxels['V4'].shape[1] == 661, "Number of voxels for V1 must be 661"
    return roi_voxels
    
def main_worker(mmp_path: str, hcp_base: str) -> None:
    """
    Main worker function to process HCP data.

    :param mmp_path: Path to the MMP data file.
    :param hcp_base: Base path for the HCP dataset.
    """
    
    mmp_data = np.load(mmp_path)
    map_all = mmp_data['map_all']
    labels = mmp_data['labels']
    
    filepaths = glob.glob(hcp_base + "/*/MNINonLinear/Results/rfMRI_REST*_LR/rfMRI_REST*_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii") + \
            glob.glob(hcp_base + "/*/MNINonLinear/Results/rfMRI_REST*_RL/rfMRI_REST*_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii")
    
    for filepath in filepaths:
        logger.info(f"Start processing data for {filepath}")
        
        try:
            output_path = def_output_path(filepath)
        except Exception:
            logger.error("Failed to define output path", exc_info=True)
            
        
        img = nib.load(filepath)
        X = img.get_fdata()
        Xn = hcp.normalize(X)
        
        try:
            roi_voxels = extract_visual_voxels(Xn, labels, map_all)
        except Exception:
            logger.error("Fails to extract visual voxels", exc_info=True)
        
        np.savez(output_path, **roi_voxels)
        logger.info(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    main_worker(
        mmp_path=MMP_PATH,
        hcp_base=HCP_BASE
    )