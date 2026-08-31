import os
import SimpleITK as sitk
import numpy as np
import torch
import torchvision.transforms.functional as TF

output_folder = "I:/Segmentations/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def reorient_to_RAS(image: sitk.Image):

    # Get current orientation
    orientation_filter = sitk.DICOMOrientImageFilter()

    # Convert to RAS (Right-Anterior-Superior)
    orientation_filter.SetDesiredCoordinateOrientation('RAS')
    ras_image = orientation_filter.Execute(image)

    return ras_image


def resample_img(itk_image, out_spacing=[1.0, 1.0, 1.0],interpolator=sitk.sitkBSpline):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)  # FIXED: Use actual HU value
    resample.SetInterpolator(interpolator)

    return resample.Execute(itk_image)


def pad_resample_crop(image, original_spacing, target_spacing=(1, 1, 1),interpolator=sitk.sitkBSpline):
    pad_mm = 10.0

    print(f"Original shape: {image.GetSize()}")
    print(f"Original spacing: {original_spacing}")

    # Padding
    pad_voxels_lower = [int(np.ceil(pad_mm / sp)) for sp in original_spacing]
    padded_image = sitk.ConstantPad(image, pad_voxels_lower, pad_voxels_lower, 0)
    print(f"After padding: {padded_image.GetSize()}")

    # Resampling
    resampled_image = resample_img(padded_image, target_spacing,interpolator)
    print(f"After resampling: {resampled_image.GetSize()}")
    resampled_size=resampled_image.GetSize()
    # Cropping
    pad_voxels_resampled = [int(np.ceil(pad_mm / target_spacing[i])) for i in range(3)]
    new_size = [resampled_size[i] - 2 * pad_voxels_resampled[i] for i in range(3)]
    start_index = pad_voxels_resampled

    print(f"Crop start: {start_index}, Crop size: {new_size}")

    cropped_image = sitk.RegionOfInterest(resampled_image, new_size, start_index)
    print(f"Final shape: {cropped_image.GetSize()}")

    return cropped_image

def robust_normalize(window_data: np.ndarray) -> np.ndarray:
    # Match SynthStrip's inference pipeline exactly:
    # 1. Subtract minimum (shift to non-negative)
    # 2. Divide by 99th percentile
    # 3. Clip to [0, 1]
    shifted = window_data - window_data.min()
    p99 = np.percentile(shifted, 99)
    if p99 < 1e-6:
        return shifted
    return np.clip(shifted / p99, 0, 1)


def preprocess(series_id,out_path):
    try:
        image = sitk.ReadImage(os.path.join(output_folder,series_id+".nii"))
        image_mask=sitk.ReadImage(os.path.join(output_folder,series_id+"_cowseg.nii"))



        image=reorient_to_RAS(image)
        image_mask=reorient_to_RAS(image_mask)
        original_spacing = image.GetSpacing()


        image = pad_resample_crop(image, original_spacing)
        image_mask = pad_resample_crop(image_mask, original_spacing,interpolator=sitk.sitkNearestNeighbor)

        channel1 = torch.from_numpy(robust_normalize(sitk.GetArrayFromImage(image)))


        H_fixed, W_fixed = 256, 256
        resized_slices = []
        for slice_idx in range(channel1.shape[0]):
            slice_2d = channel1[slice_idx]  # (H, W)
            # Add channel dim, resize, remove channel
            slice_resized = TF.resize(slice_2d.unsqueeze(0), [H_fixed, W_fixed]).squeeze(0)
            resized_slices.append(slice_resized)
        channel1_resized = torch.stack(resized_slices)  # (D, H_fixed, W_fixed)

        # Similarly resize mask slices (nearest neighbor)
        mask = torch.from_numpy(sitk.GetArrayFromImage(image_mask)).float()  # (D, H, W)
        resized_masks = []
        for slice_idx in range(mask.shape[0]):
            mask_slice = mask[slice_idx]
            mask_resized = TF.resize(mask_slice.unsqueeze(0), [H_fixed, W_fixed],
                                     interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
            resized_masks.append(mask_resized)
        mask_resized = torch.stack(resized_masks)

        # Recompute slice_labels and bounding box on resized mask
        slice_has_vessel = (mask_resized.numpy().sum(axis=(1, 2)) > 0).astype(np.float32)


        mask_array=mask_resized.detach().clone().numpy()
        non_zero_indices = np.where(mask_array > 0)


        bbox_min = [
            max(0, np.min(non_zero_indices[0]) - 3),  # x_min
            max(0, np.min(non_zero_indices[1]) - 3),  # y_min
            max(0, np.min(non_zero_indices[2]) - 2)  # z_min
        ]
        bbox_max = [
            min(mask_array.shape[0], np.max(non_zero_indices[0]) + 3),  # x_max
            min(mask_array.shape[1], np.max(non_zero_indices[1]) + 3),  # y_max
            min(mask_array.shape[2], np.max(non_zero_indices[2]) + 2)  # z_max
        ]

        x1, x2, y1, y2, z1, z2 = bbox_min[0], bbox_max[0], bbox_min[1], bbox_max[1], bbox_min[2], bbox_max[2]

        print(x1, x2, y1, y2, z1, z2)

        d_resized, h_resized, w_resized = channel1_resized.shape  # (D, 256, 256)
        norm_coords = [x1 / d_resized, x2 / d_resized, y1 / h_resized, y2 / h_resized, z1 / w_resized, z2 / w_resized]



        dict={
            "tensors":channel1_resized,
            "slice_labels": torch.from_numpy(slice_has_vessel),
            "coordinates":norm_coords,

        }

        torch.save(dict,out_path)



    except Exception as e:
        raise e
        with open("Input_details/stage2_failed.txt", "a") as f:
            f.write(f"{series_id}\n")
        print("Failed preprocessing for series id ", series_id)


def visualise_preprocessing(image,mask,channel1_resized,mask_resized):
    # Convert to numpy and then to SimpleITK images
    resized_img_np = channel1_resized.numpy()  # shape (D, 256, 256)
    resized_mask_np = mask_resized.numpy()  # shape (D, 256, 256)

    resized_img_sitk = sitk.GetImageFromArray(resized_img_np)
    resized_mask_sitk = sitk.GetImageFromArray(resized_mask_np)


    original_H, original_W = mask.shape[1], mask.shape[2]  # from mask before resize
    new_spacing_x = 1.0 * (original_W / 256.0)
    new_spacing_y = 1.0 * (original_H / 256.0)
    new_spacing_z = 1.0  # depth spacing unchanged

    resized_img_sitk.SetSpacing((new_spacing_x, new_spacing_y, new_spacing_z))
    resized_mask_sitk.SetSpacing((new_spacing_x, new_spacing_y, new_spacing_z))


    resized_img_sitk.SetOrigin(image.GetOrigin())
    resized_img_sitk.SetDirection(image.GetDirection())
    resized_mask_sitk.SetOrigin(image.GetOrigin())
    resized_mask_sitk.SetDirection(image.GetDirection())


    vis_folder = "I:/Segmentation_visualization"
    os.makedirs(vis_folder, exist_ok=True)
    sitk.WriteImage(resized_img_sitk, f"visualise_resized.nii")
    sitk.WriteImage(resized_mask_sitk, f"visualise_resized_mask.nii")




#--- main preprocessing
series_ids = [f.split(".nii")[0] for f in os.listdir(output_folder) if not f.endswith("_cowseg.nii")]


for i in series_ids:
    out_path=os.path.join("I:\Segmentation_preprocessed",i+".pt" )
    preprocess(i, out_path)
    # if not os.path.exists(out_path):
    #     preprocess(i,out_path)


# --- main preprocessing