import matplotlib.pyplot as plt
import matplotlib as mpl
from cellpose import io, models
from pathlib import Path
import tifffile
import skimage
from skimage.morphology import disk, dilation
import numpy as np

#%% Define cytoplasm generation function

def generate_cytoplasm_masks(masks, dilation=10):
    
    # initialize masks
    cytoplasm_masks = np.zeros_like(masks)
    nucleus_masks = masks.copy()
    
    # iterate through all unique mask id's, exclusing background (0)
    for cell_id in np.unique(masks)[1:]:
        
        # converts mask to binary mask that maps a single cell 
        nucleus_mask = (masks == cell_id).astype(np.uint8)
        
        # Dilate the nucleus mask to approximate the cell boundary
        dilated_cell = skimage.morphology.dilation(nucleus_mask, skimage.morphology.disk(dilation))
        
        # Subtract the nucleus to get the cytoplasm region
        cytoplasm_mask = dilated_cell - nucleus_mask
        
        # check for overlapping regions
        existing_areas = cytoplasm_masks > 0
        
        # if cytoplasm_mask overlaps, overlap_mask is true and mask is set to 0
        cytoplasm_mask[existing_areas] = 0
        
        # add to masks
        cytoplasm_masks[cytoplasm_mask > 0] = cell_id
        
        # remove any cytoplasm's in other nucleus regions
        cytoplasm_masks = cytoplasm_masks * (nucleus_masks == 0)
        
    return cytoplasm_masks, nucleus_masks


#%% Get Files

def get_all_tiff_files(path):
    return list(Path(path).rglob("*.tif*"))


#%% Load Images
def process_image(selected_image_path, diameter=50, cell_prob_threshold=0.5, flow_threshold=0.5, dilation=10):
    # Get the directory where this script is located
    HERE = Path(__file__).resolve().absolute().parent
    # Define the path to the Models directory relative to this script
    path_models = HERE / "Models" # Assumes Models is in the same dir as this .py file

    save_results = True
    
    filename_suffix = "cellpose" # 
    
    dict_Cellpose_params = {
        "gpu" : True,
        # Construct the full path to the specific model file
        "pretrained_model": str(path_models / "Organoid/OrganoidNuclei.zip"), # Assumes Organoid subfolder
        # Other model options (commented out):
        # 'model_type' : 'cyto2',
        # 'pretrained_model' : str(path_models / 'Adherent Cell/AdherentCells.zip'),
        # 'pretrained_model' : str(path_models / 'Adherent Cell/AdherentNuclei.zip'),
        # 'pretrained_model' : str(path_models / 'Organoid/OrganoidCells.zip'),
    }
    
    dict_eval_params = {
        'diameter' : diameter,
        'cellprob_threshold' : cell_prob_threshold,
        'flow_threshold' : flow_threshold # model match threshold on GUI
    }
        
    model = models.CellposeModel(**dict_Cellpose_params)
        
    channels = [[0,0]]
        
    img = tifffile.imread(selected_image_path)
        
    masks, flows, styles = model.eval(img, channels=channels, **dict_eval_params)

    cytoplasm_masks, nucleus_masks = generate_cytoplasm_masks(masks, dilation=dilation)

    return img, cytoplasm_masks
    
    # Save File
def save_masks(selected_image_path, masks):
    HERE = Path(__file__).resolve().absolute().parent
    output_dir = HERE / "masks_cellpose"
    output_dir.mkdir(exist_ok=True)
    filename_mask = f"{Path(selected_image_path).stem}_cellpose.tiff"
    output_path = output_dir / filename_mask
    tifffile.imwrite(output_path, masks)
    return output_path

        
    
        
    