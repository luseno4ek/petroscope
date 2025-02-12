import argparse
from pathlib import Path
import numpy as np
import petroscope.segmentation as segm
from petroscope.segmentation.utils.base import prepare_experiment
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import json

import argparse
from pathlib import Path

import petroscope.segmentation as segm
from petroscope.segmentation.utils.base import prepare_experiment

minerals_idx_name_dict = {
    0: "background",
    1: "chalcopyrite",
    2: "galena",
    3: "magnetite",
    4: "bornite",
    5: "pyrrhotite",
    6: "pyrite/marcasite",
    7: "pentlandite",
    8: "sphalerite",
    9: "arsenopyrite",
    10: "hematite",
    11: "tenantite-tetrahedrite group",
    12: "covelline"
}

def get_test_img_mask_pairs(ds_dir: Path):
    """
    Get paths to test images and corresponding masks from dataset directory.
    """
    test_img_mask_p = [
        (img_p, ds_dir / "masks" / "test" / f"{img_p.stem}.png")
        for img_p in sorted((ds_dir / "imgs" / "test").iterdir())
    ]
    return test_img_mask_p


def run_test(
    ds_dir: Path,
    out_dir: Path,
    device: str,
    void_pad=4,
    void_border_width=2,
    vis_segmentation=True,
    vis_plots=False,
    exp_name: str = None
):
    """
    Runs model on test images from dataset directory and
    saves results to output directory.
    """
    classes = segm.LumenStoneClasses.S1v1()
    model = segm.models.ResUNetTorch.trained("s1_x05_calib", device)
    tester = segm.SegmDetailedTester(
        out_dir=out_dir,
        classes=classes,
        void_pad=void_pad,
        void_border_width=void_border_width,
        vis_segmentation=vis_segmentation,
        vis_plots=vis_plots,
    )
    res, res_void = tester.test_on_set(
        get_test_img_mask_pairs(ds_dir),
        predict_func=model.predict_image,
        description="images",
        exp_name=exp_name
    )
    print("results without void borders:\n", res)
    print("results with void borders:\n", res_void)

def plot_error_heatmap(error_masks, exp_name = None, image_name = None):
    """
    Plot a heatmap based on segmentation error masks.

    Parameters:
    - error_masks: a list of binary NumPy arrays representing error masks for each color augmentation step.

    Returns:
    - None: The function directly plots the heatmap.
    """

    num_masks = len(error_masks)

    # Initialize an empty heatmap
    heatmap = np.zeros_like(error_masks[0], dtype=np.float32)

    # Assign decreasing weights for more robust error detection at earlier stages
    for idx, mask in enumerate(error_masks):
        # Weight is greatest for the first mask and decreases with index
        weight = num_masks - idx
        # Convert mask so that 1 represents an error (invert mask)
        error_weight = 1 - mask
        # Update the heatmap with weighted errors
        heatmap += error_weight * weight
    
    # Normalize the heatmap for better visualization
    max_value = heatmap.max() if heatmap.max() != 0 else 1  # Prevent division by zero
    heatmap_normalized = heatmap / max_value

    # Plot the heatmap
    plt.imshow(heatmap_normalized, interpolation='nearest')
    # plt.colorbar(label='Error Intensity')
    # plt.title('Segmentation Error Heatmap')
    plt.axis('off')
    path = f"./out/{exp_name}/error_heatmap"
    Path(path).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{path}/{image_name}.png", bbox_inches='tight') 
    np.save(f"{path}/{image_name}_raw.npy", heatmap_normalized)

def plot_error_heatmaps(ds_dir, exp_name, h = 10):
    img_mask_paths = get_test_img_mask_pairs(ds_dir)
    for img_mask_path in img_mask_paths:
        name = img_mask_path[0].stem
        # Skip all color changed files
        if '_' in name:
            continue
        error_masks = []
        for i in range(1,h+1):
            img_path = f"./out/{exp_name}/images/img_{name}_{i}_err.jpg"
            img = cv2.imread(img_path, 0)
            mask_1 = np.copy(img == 118)
            mask_2 = np.copy(img != 118)
            img[mask_1] = 0
            img[mask_2] = 1
            error_masks += [img]
        plot_error_heatmap(error_masks, exp_name=exp_name, image_name=name)

def get_test_error_radius_pairs(ds_dir: Path):
    """
    Get paths to test images and corresponding masks from dataset directory.
    """
    test_img_mask_p = [
        (img_p, ds_dir / "entropy_radius_map" / f"{img_p.stem}.npy")
        for img_p in sorted((ds_dir / "error_heatmap" ).iterdir()) if '_raw' in img_p.stem
    ]
    return test_img_mask_p

def normalize_map(x):
    return (x - x.min().item()) / (x.max().item() - x.min().item())

def weighted_mean(values, weights):
    return np.sum(weights * values) / np.sum(weights)

def weighted_covariance(X, Y, weights):
    mean_x = weighted_mean(X, weights)
    mean_y = weighted_mean(Y, weights)
    weighted_cov = np.sum(weights * (X - mean_x) * (Y - mean_y)) / np.sum(weights)
    return weighted_cov

def weighted_variance(values, weights):
    mean_val = weighted_mean(values, weights)
    weighted_var = np.sum(weights * (values - mean_val) ** 2) / np.sum(weights)
    return weighted_var

def weighted_correlation(X, Y, weights):
    cov_xy = weighted_covariance(X, Y, weights)
    var_x = weighted_variance(X, weights)
    var_y = weighted_variance(Y, weights)
    weighted_corr = cov_xy / np.sqrt(var_x * var_y)
    return weighted_corr

def get_radius_error_correlation(
        ds_dir, exp_name, shape = (2560, 3408), shape_small = (2547, 3396), normalize_radius_map = False, normalize_result_vectors = False):
    mineral_sq_err_dict = defaultdict(list)
    mineral_radius_dict = defaultdict(list)
    mineral_error_dict = defaultdict(list)

    path = Path(f"./out/{exp_name}")

    path_pairs = get_test_error_radius_pairs(path)

    for pair in path_pairs:
        name = pair[0].stem
        mask_path = f"{ds_dir}/masks/test/{name.split('_')[0]}.png"
        mask = cv2.imread(mask_path)
        mask = mask[:, :, 0]
        error = np.load(pair[0]).reshape(shape_small)
        x_dif_0 = (shape[0] - shape_small[0]) // 2
        x_dif_1 = x_dif_0 + (shape[0] - shape_small[0]) % 2
        y_dif_0 = (shape[1] - shape_small[1]) // 2
        y_dif_1 = y_dif_0 + (shape[1] - shape_small[1]) % 2
        radius = np.load(pair[1]).reshape(shape)[x_dif_0:shape[0]-x_dif_1, y_dif_0:shape[1]-y_dif_1]

        if normalize_radius_map:
            radius = normalize_map(radius)

        sq_err = np.square(error - radius)

        minerals = np.unique(mask)

        for mineral in minerals:
            values = sq_err[mask == mineral]
            radius_values = radius[mask == mineral]
            error_values = error[mask == mineral]
            mineral_sq_err_dict[mineral].append(values)
            mineral_radius_dict[mineral].append(radius_values)
            mineral_error_dict[mineral].append(error_values)

    mineral_mse_dict = {
        minerals_idx_name_dict[int(k)]  : 
        round(float(np.mean(np.concatenate(v))), 5) 
        for k, v in mineral_sq_err_dict.items()
        }

    mineral_correlation_dict = {
        minerals_idx_name_dict[int(k)]  : 
        round(float(
            np.corrcoef(np.concatenate(v), np.concatenate(mineral_error_dict[k]))[0, 1]
                ), 5) 
        for k, v in mineral_radius_dict.items()
    }

    mineral_weighted_correlation_dict = {
        minerals_idx_name_dict[int(k)]  : 
        round(float(
            weighted_correlation(np.concatenate(v), np.concatenate(mineral_error_dict[k]), np.concatenate(mineral_error_dict[k])**2)
                ), 5) 
        for k, v in mineral_radius_dict.items()
    }

    return mineral_mse_dict, mineral_correlation_dict, mineral_weighted_correlation_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to use for inference",
    )
    args = parser.parse_args()

    shape = (1280, 1712)
    shape_small = (1273, 1698)

    exp_name = 'S1_v1_yellowx05_calib_colordrift_10_s1x05calib'
    ds_dir=Path("./data/yellow_x05")

    run_test(
        ds_dir=ds_dir,
        out_dir=prepare_experiment(Path("./out"), exp_name=exp_name),
        device=args.device,
        exp_name=exp_name
    )

    plot_error_heatmaps(ds_dir=ds_dir, exp_name=exp_name, h=10)

    mse_dict, correlation_dict, weighted_correlation_dict = get_radius_error_correlation(ds_dir, exp_name, shape=shape, shape_small=shape_small)
    print("Not normalized radius map \n MSE = ", mse_dict, "\n Correlation = ", correlation_dict)
    with open(f'./out/{exp_name}/correlation_dict.json', 'wt') as f:
        json.dump(correlation_dict, f, indent=4)
    with open(f'./out/{exp_name}/mse_dict.json', 'wt') as f:
        json.dump(mse_dict, f, indent=4)
    with open(f'./out/{exp_name}/weighted_correlation_dict.json', 'wt') as f:
        json.dump(weighted_correlation_dict, f, indent=4)