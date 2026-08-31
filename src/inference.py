import os
import json
import tempfile
import torch
import numpy as np
import SimpleITK as sitk
import torchvision.transforms.functional as TF
import torchvision.models as models
import torch.nn as nn
from flask import Flask, request, jsonify, send_file,render_template
from scipy.ndimage import label

CONFIG_PATH = "./config/config.json"

TEMP_DIR = tempfile.mkdtemp(prefix="inference_")

app = Flask(__name__)

model = None
threshold = None
device = None
CONTEXT = None
H_FIXED = None
W_FIXED = None
TARGET_SPACING = (1.0, 1.0, 1.0)
PADDING = 1
MIN_RUN_LENGTH = 3


class SliceClassifier(nn.Module):
    def __init__(self, dropout_rate=0.4, context=1):
        super().__init__()
        in_channels = 2 * context + 1
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        orig_weight = backbone.conv1.weight.data
        new_weight = orig_weight.repeat(1, in_channels, 1, 1)
        new_weight = new_weight[:, :in_channels, :, :] / (in_channels / 3.0)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = new_weight
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.encoder(x).flatten(1)
        return self.classifier(self.dropout(features))


def reorient_to_RAS(image: sitk.Image):
    orientation_filter = sitk.DICOMOrientImageFilter()
    orientation_filter.SetDesiredCoordinateOrientation('RAS')
    return orientation_filter.Execute(image)

def resample_img(itk_image, out_spacing=[1.0, 1.0, 1.0], interpolator=sitk.sitkBSpline):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(interpolator)
    return resample.Execute(itk_image)

def pad_resample_crop(image, original_spacing, target_spacing=(1, 1, 1), interpolator=sitk.sitkBSpline):

    pad_mm = 10.0
    pad_voxels_lower = [int(np.ceil(pad_mm / sp)) for sp in original_spacing]
    padded_image = sitk.ConstantPad(image, pad_voxels_lower, pad_voxels_lower, 0)
    resampled_image = resample_img(padded_image, target_spacing, interpolator)
    resampled_size = resampled_image.GetSize()

    pad_voxels_resampled = [int(np.ceil(pad_mm / target_spacing[i])) for i in range(3)]
    new_size = [resampled_size[i] - 2 * pad_voxels_resampled[i] for i in range(3)]
    start_index = pad_voxels_resampled

    return sitk.RegionOfInterest(resampled_image, new_size, start_index)

def robust_normalize(window_data: np.ndarray):
    shifted = window_data - window_data.min()
    p99 = np.percentile(shifted, 99)
    if p99 < 1e-6:
        return shifted
    return np.clip(shifted / p99, 0, 1)


def get_crop_indices(slice_probs, threshold, padding, min_run_length):
    predicted_positive = (slice_probs > threshold).astype(int)
    labeled, num_features = label(predicted_positive)

    for region_id in range(1, num_features + 1):
        region_slices = np.where(labeled == region_id)[0]
        if len(region_slices) < min_run_length:
            predicted_positive[region_slices] = 0

    positive_indices = np.where(predicted_positive == 1)[0]

    if len(positive_indices) == 0:
        return 0, len(slice_probs) - 1
    
    padded_start = max(0, positive_indices.min() - padding)
    padded_end = min(len(slice_probs) - 1, positive_indices.max() + padding)
    return int(padded_start), int(padded_end)


def run_inference_on_image(image_path, output_path=None):

    image = sitk.ReadImage(image_path)
    image = reorient_to_RAS(image)
    original_spacing = image.GetSpacing()
    image_preprocessed = pad_resample_crop(image, original_spacing, target_spacing=TARGET_SPACING)
    img_array = sitk.GetArrayFromImage(image_preprocessed)
    img_array = robust_normalize(img_array)


    channel = torch.from_numpy(img_array).float()
    resized_slices = []
    for s in range(channel.shape[0]):
        sl = TF.resize(channel[s].unsqueeze(0), [H_FIXED, W_FIXED]).squeeze(0)
        resized_slices.append(sl)
    volume = torch.stack(resized_slices)  # (D, H, W)
    D = volume.shape[0]


    stacks = []
    for i in range(D):
        indices = [max(0, min(D - 1, i + offset)) for offset in range(-CONTEXT, CONTEXT + 1)]
        stack = torch.stack([volume[idx] for idx in indices], dim=0).float()
        stacks.append(stack)


    batch_size = 64
    all_probs = []
    with torch.no_grad():
        for start in range(0, D, batch_size):
            batch = torch.stack(stacks[start:start + batch_size]).to(device)
            logits = model(batch).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 0:
                probs = np.array([float(probs)])
            all_probs.extend(probs.tolist())
    slice_probs = np.array(all_probs)


    start_idx, end_idx = get_crop_indices(slice_probs, threshold, PADDING, MIN_RUN_LENGTH)

    
    size = list(image_preprocessed.GetSize())
    region_start = [0, 0, start_idx]
    region_size = [size[0], size[1], end_idx - start_idx + 1]
    cropped_sitk = sitk.RegionOfInterest(image_preprocessed, region_size, region_start)

    if output_path:
        sitk.WriteImage(cropped_sitk, output_path)

    return cropped_sitk, (start_idx, end_idx), slice_probs



@app.route('/')
def index():
    return render_template('main.html', result=False)


def strip_nii_extension(filename):
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    elif filename.endswith('.nii'):
        return filename[:-4]
    return filename

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    input_path = os.path.join(TEMP_DIR, file.filename)
    file.save(input_path)

    basename = strip_nii_extension(os.path.basename(file.filename))
    output_filename = f"cropped_{basename}.nii.gz"
    output_path = os.path.join(TEMP_DIR, output_filename)
    cropped, (start, end), probs = run_inference_on_image(input_path, output_path)

    return jsonify({
        "start": start,
        "end": end,
        "n_slices": end - start + 1,
        "pos_raw": int((probs > threshold).sum()),
        "threshold": float(threshold),
        "filename": output_filename,
        "probs": probs.tolist()
    })

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True, download_name=filename)

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


def load_model_from_config():
    global model, threshold, device, CONTEXT, H_FIXED, W_FIXED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}. Please run training first to generate it.")

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    print("CONFIG CONTENT:", config)

    model_path = config.get("model_name", "best_slice_classifier.pt")
    CONTEXT = int(config.get("context_slices", 2))
    H_FIXED = int(config.get("resize_height", 256))
    W_FIXED = int(config.get("resize_width", 256))
    threshold = float(config.get("threshold", 0.5))
    print(f"Loaded training parameters: context={CONTEXT}, resize={H_FIXED}x{W_FIXED}")

    
    local_path = os.path.join("models",model_path)
    checkpoint_path = local_path

   
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SliceClassifier(dropout_rate=0.4, context=CONTEXT).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    threshold = float(config.get("threshold", checkpoint.get("threshold", 0.5)))

    print(f"Model loaded. Threshold = {threshold:.3f}")
    print(f"Validation F1 = {checkpoint.get('val_f1', 'N/A')}")


if __name__ == "__main__":
    load_model_from_config()
    app.run(host="0.0.0.0", port=5000, debug=False)
