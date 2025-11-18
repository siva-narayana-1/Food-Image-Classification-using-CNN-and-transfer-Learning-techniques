import os
import io
import json
import zipfile
import requests
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import cv2

from tensorflow.keras.models import load_model
from tensorflow.nn import softmax

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ==========================================================================================
#                         DROPBOX DIRECT DOWNLOAD ZIP LINKS
# ==========================================================================================

DROPBOX_LINKS = {
    "custom": "https://www.dropbox.com/scl/fi/6lla16txaovsbnkm8u842/custom_models.zip?dl=1",
    "resnet": "https://www.dropbox.com/scl/fi/l9oh5don6asitmfxxkyta/Resnet_models.zip?dl=1",
    "vgg":    "https://www.dropbox.com/scl/fi/1kxxujmxqr09hb27l9dys/Vgg_models.zip?dl=1",
}

MODEL_DIRS = {
    "custom": "custom_models",
    "resnet": "Resnet_models",
    "vgg": "vgg_models"
}

for d in MODEL_DIRS.values():
    os.makedirs(d, exist_ok=True)


# ==========================================================================================
#                     DOWNLOAD & EXTRACT ZIP FILE IF NOT PRESENT
# ==========================================================================================

def folder_has_h5(folder: str) -> bool:
    """Check recursively if any .h5 file exists inside a folder."""
    for root, dirs, files in os.walk(folder):
        if any(f.endswith(".h5") for f in files):
            return True
    return False


def download_and_extract_if_needed(model_type: str):
    folder = MODEL_DIRS[model_type]
    url = DROPBOX_LINKS[model_type]
    zip_path = f"{folder}.zip"

    # Skip if already extracted
    if folder_has_h5(folder):
        print(f"[INFO] {model_type} models already downloaded.")
        return

    print(f"[INFO] Downloading {model_type} models from Dropbox...")

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"[INFO] Downloaded ZIP for {model_type} to {zip_path}")
    except Exception as e:
        print(f"[ERROR] Dropbox download failed for {model_type}: {e}")
        return

    print(f"[INFO] Extracting ZIP for {model_type}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(folder)
        print(f"[INFO] Extracted {model_type} models successfully into {folder}")
    except Exception as e:
        print(f"[ERROR] ZIP extraction failed for {model_type}: {e}")


# Download models once at startup
download_and_extract_if_needed("custom")
download_and_extract_if_needed("resnet")
download_and_extract_if_needed("vgg")


# ==========================================================================================
#                             JSON FILES & CLASS MAPPING
# ==========================================================================================

CLASS_NAMES_FILE = 'class.json'
CUSTOM_JSON_FILE = 'model_evaluation_results.json'
RESNET_JSON_FILE = 'model_evaluation_results_resnet.json'
VGG_JSON_FILE = 'model_evaluation_results_vgg.json'

NUTRITION_JSON = json.load(open(CLASS_NAMES_FILE, "r"))  # you used class.json for nutrition + labels


def normalize(name: str) -> str:
    return name.lower().replace(" ", "_")


RAW_MODEL_CLASS_INDEX = {
    # CUSTOM MODELS
    "custom_model_1":  {'apple_pie': 0, 'baked_potato': 1, 'burger': 2},
    "custom_model_2":  {'butter_naan': 0, 'chai': 1, 'chapati': 2},
    "custom_model_3":  {'cheesecake': 0, 'chicken_curry': 1, 'chole_bhature': 2},
    "custom_model_4":  {'crispy_chicken': 0, 'dal_makhani': 1, 'dhokla': 2},
    "custom_model_5":  {'donut': 0, 'fried_rice': 1, 'fries': 2},
    "custom_model_6":  {'hot_dog': 0, 'ice_cream': 1, 'idli': 2},
    "custom_model_7":  {'jalebi': 0, 'kaathi_rolls': 1, 'kadai_paneer': 2},
    "custom_model_8":  {'kulfi': 0, 'masala_dosa': 1, 'momos': 2},
    "custom_model_9":  {'omelette': 0, 'paani_puri': 1, 'pakode': 2},
    "custom_model_10": {'pav_bhaji': 0, 'pizza': 1, 'samosa': 2},
    "custom_model_11": {'sandwich': 0, 'sushi': 1, 'taco': 2, 'taquito': 3},

    # RESNET MODELS
    "resnet_model_1":  {'apple_pie': 0, 'baked_potato': 1, 'burger': 2},
    "resnet_model_2":  {'butter_naan': 0, 'chai': 1, 'chapati': 2},
    "resnet_model_3":  {'cheesecake': 0, 'chicken_curry': 1, 'chole_bhature': 2},
    "resnet_model_4":  {'crispy_chicken': 0, 'dal_makhani': 1, 'dhokla': 2},
    "resnet_model_5":  {'donut': 0, 'fried_rice': 1, 'fries': 2},
    "resnet_model_6":  {'hot_dog': 0, 'ice_cream': 1, 'idli': 2},
    "resnet_model_7":  {'jalebi': 0, 'kaathi_rolls': 1, 'kadai_paneer': 2},
    "resnet_model_8":  {'kulfi': 0, 'masala_dosa': 1, 'momos': 2},
    "resnet_model_9":  {'omelette': 0, 'paani_puri': 1, 'pakode': 2},
    "resnet_model_10": {'pav_bhaji': 0, 'pizza': 1, 'samosa': 2},
    "resnet_model_11": {'sandwich': 0, 'sushi': 1, 'taco': 2, 'taquito': 3},

    # VGG MODELS
    "vgg_model_1":  {'apple_pie': 0, 'baked_potato': 1, 'burger': 2},
    "vgg_model_2":  {'butter_naan': 0, 'chai': 1, 'chapati': 2},
    "vgg_model_3":  {'cheesecake': 0, 'chicken_curry': 1, 'chole_bhature': 2},
    "vgg_model_4":  {'crispy_chicken': 0, 'dal_makhani': 1, 'dhokla': 2},
    "vgg_model_5":  {'donut': 0, 'fried_rice': 1, 'fries': 2},
    "vgg_model_6":  {'hot_dog': 0, 'ice_cream': 1, 'idli': 2},
    "vgg_model_7":  {'jalebi': 0, 'kaathi_rolls': 1, 'kadai_paneer': 2},
    "vgg_model_8":  {'kulfi': 0, 'masala_dosa': 1, 'momos': 2},
    "vgg_model_9":  {'omelette': 0, 'paani_puri': 1, 'pakode': 2},
    "vgg_model_10": {'pav_bhaji': 0, 'pizza': 1, 'samosa': 2},
    "vgg_model_11": {'sandwich': 0, 'sushi': 1, 'taco': 2, 'taquito': 3},
}

RAW_MODEL_CLASS_INDEX = {
    model: {normalize(cls): idx for cls, idx in mapping.items()}
    for model, mapping in RAW_MODEL_CLASS_INDEX.items()
}

MODEL_CLASS_INDEX = {m.lower(): v for m, v in RAW_MODEL_CLASS_INDEX.items()}

CLASS_NAMES = json.load(open(CLASS_NAMES_FILE))
CLASS_LIST = list(CLASS_NAMES.keys())


# ==========================================================================================
#                       JSON LOADERS AND MODEL FILE LOCATOR
# ==========================================================================================

def load_eval_json(model_type: str):
    if model_type == "custom_model":
        return json.load(open(CUSTOM_JSON_FILE))
    if model_type == "resnet_model":
        return json.load(open(RESNET_JSON_FILE))
    if model_type == "vgg_model":
        return json.load(open(VGG_JSON_FILE))
    return {}


def find_model_from_json(eval_json: dict, classname: str):
    cname = normalize(classname)
    for key, val in eval_json.items():
        if normalize(key) == cname:
            return val["model_used"], val
    return None, None


def get_model_path(model_type: str, model_used: str):
    model_used = model_used.replace(".h5", "")
    folder = MODEL_DIRS[model_type.replace("_model", "")]
    # search recursively in case ZIP contains subfolder
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".h5") and model_used in f:
                return os.path.join(root, f)
    return None


# ==========================================================================================
#                                 MODEL CACHE
# ==========================================================================================

_model_cache = {}

def load_model_cached(path: str):
    if path not in _model_cache:
        print(f"[INFO] Loading model → {path}")
        _model_cache[path] = load_model(path)
    return _model_cache[path]


# ==========================================================================================
#                              IMAGE PREPROCESSING
# ==========================================================================================

def preprocess_dynamic(img, w, h):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (w, h))
    img = np.expand_dims(img, 0).astype("float32") / 255.0
    return img


# ==========================================================================================
#                                     ROUTES
# ==========================================================================================

@app.route("/")
def index():
    return render_template("index.html", classes=CLASS_LIST)


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"})

    model_type = request.form.get("model_type")
    selected_class = request.form.get("selected_class")

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(filepath)

    eval_json = load_eval_json(model_type)
    model_used, class_entry = find_model_from_json(eval_json, selected_class)

    if not model_used:
        return jsonify({"success": False, "error": f"Model not found for class {selected_class}"})

    model_used = model_used.lower()
    model_path = get_model_path(model_type, model_used)

    if not model_path:
        return jsonify({"success": False, "error": f"Model file missing for {model_used}"})

    model = load_model_cached(model_path)

    _, h, w, _ = model.input_shape
    img = Image.open(filepath)
    x = preprocess_dynamic(img, w, h)

    pred = model.predict(x).squeeze()
    if np.sum(pred) == 0 or np.any(pred < 0):
        pred = softmax(pred).numpy()

    idx = int(np.argmax(pred))
    label = next((cls for cls, i in MODEL_CLASS_INDEX[model_used].items() if i == idx), None)
    conf = float(np.max(pred))

    nutrition = NUTRITION_JSON.get(normalize(selected_class), {})

    confusion_matrix_full = class_entry.get("confusion_matrix_full")
    model_labels_order = list(MODEL_CLASS_INDEX.get(model_used, {}).keys())

    return jsonify({
        "success": True,
        "selected_class": selected_class,
        "predicted_label": label,
        "confidence": conf,
        "model_used": model_used,
        "model_type": model_type,
        "class_metrics": class_entry,
        "nutrition": nutrition,
        "confusion_matrix_full": confusion_matrix_full,
        "model_labels_order": model_labels_order,
    })


# ==========================================================================================
#                                     MAIN
# ==========================================================================================

if __name__ == "__main__":
    app.run(debug=True)
