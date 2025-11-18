import os
import io
import json
import zipfile
import requests
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import cv2

from tensorflow.keras.models import load_model
from tensorflow.nn import softmax


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ==========================================================================================
#                              ONEDRIVE MODEL DOWNLOAD (ZIP FILE LINKS)
# ==========================================================================================

ONEDRIVE_LINKS = {
    "vgg": "https://npt3-my.sharepoint.com/:u:/g/personal/siva_madhala_npt3_onmicrosoft_com/IQCY-eAfpvtOQbl5AsNqoXhCASQek13PeHkzKl1e_PkMUN4?e=ncUQf1",
    "resnet": "https://npt3-my.sharepoint.com/:u:/g/personal/siva_madhala_npt3_onmicrosoft_com/IQCRrvHZOZNQRI9PWUY1m13UAWOcAPBGYNr5nI3J4YaTVsU?e=01YX6h",
    "custom": "https://npt3-my.sharepoint.com/:u:/g/personal/siva_madhala_npt3_onmicrosoft_com/IQA3_27GeetPRrHniHO3XmP8AY91Ya-SmrlUVZPkjedSEEM?e=bb8tnP"
}

MODEL_DIRS = {
    "vgg": "vgg_models",
    "resnet": "Resnet_models",
    "custom": "custom_models"
}

# Create folders if not exist
for d in MODEL_DIRS.values():
    os.makedirs(d, exist_ok=True)


def download_and_extract_if_needed(model_type):
    """
    Downloads ZIP from OneDrive and extracts all .h5 files.
    Works with direct file URLs (your :u:/ links).
    """
    folder = MODEL_DIRS[model_type]

    # Skip if already downloaded
    if any(f.endswith(".h5") for f in os.listdir(folder)):
        print(f"[INFO] {model_type} models already exist, skipping download.")
        return

    url = ONEDRIVE_LINKS[model_type]
    print(f"[INFO] Downloading {model_type} models from OneDrive...")

    try:
        response = requests.get(url, allow_redirects=True)
    except Exception as e:
        print(f"[ERROR] Failed to download {model_type}: {e}")
        return

    # If OneDrive returns HTML instead of ZIP
    if "text/html" in response.headers.get("Content-Type", ""):
        print(f"[ERROR] OneDrive returned HTML instead of ZIP for {model_type}.")
        return

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(folder)
            print(f"[INFO] Extracted {model_type} models successfully.")
    except zipfile.BadZipFile:
        print(f"[ERROR] Invalid ZIP for {model_type}. Check link.")
    except Exception as e:
        print(f"[ERROR] Extraction failed for {model_type}: {e}")


# Download all model sets (only on first boot)
download_and_extract_if_needed("vgg")
download_and_extract_if_needed("resnet")
download_and_extract_if_needed("custom")


# ==========================================================================================
#                                  JSON / CLASS MAPPINGS
# ==========================================================================================

CLASS_NAMES_FILE = 'class.json'
CUSTOM_JSON_FILE = 'model_evaluation_results.json'
RESNET_JSON_FILE = 'model_evaluation_results_resnet.json'
VGG_JSON_FILE = 'model_evaluation_results_vgg.json'

NUTRITION_JSON = json.load(open("class.json", "r"))

def normalize(name):
    return name.lower().replace(" ", "_")

# FULL CLASS-MODEL MAPPING (unchanged)
RAW_MODEL_CLASS_INDEX = {
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
    "vgg_model_11": {'sandwich': 0, 'sushi': 1, 'taco': 2, 'taquito': 3}
}

RAW_MODEL_CLASS_INDEX = {
    m: {normalize(c): i for c, i in mapping.items()}
    for m, mapping in RAW_MODEL_CLASS_INDEX.items()
}

MODEL_CLASS_INDEX = {m.lower(): v for m, v in RAW_MODEL_CLASS_INDEX.items()}

CLASS_NAMES = json.load(open(CLASS_NAMES_FILE))
CLASS_LIST = list(CLASS_NAMES.keys())


# ==========================================================================================
#                                JSON LOAD HELPERS
# ==========================================================================================

def load_eval_json(model_type):
    if model_type == "custom_model":
        return json.load(open(CUSTOM_JSON_FILE))
    if model_type == "resnet_model":
        return json.load(open(RESNET_JSON_FILE))
    if model_type == "vgg_model":
        return json.load(open(VGG_JSON_FILE))
    return {}


def find_model_from_json(eval_json, classname):
    cname = normalize(classname)
    for key, val in eval_json.items():
        if normalize(key) == cname:
            return val["model_used"], val
    return None, None


# ==========================================================================================
#                            MODEL FILE LOCATOR + LOADER
# ==========================================================================================

def get_model_path(model_type, model_used):
    model_used = model_used.replace(".h5", "")
    folder = MODEL_DIRS[model_type.replace("_model", "")]

    exact = os.path.join(folder, model_used + ".h5")
    if os.path.exists(exact):
        return exact

    for f in os.listdir(folder):
        if model_used.lower() in f.lower():
            return os.path.join(folder, f)

    return None


_model_cache = {}

def load_model_cached(path):
    if path not in _model_cache:
        print(f"[INFO] Loading model: {path}")
        _model_cache[path] = load_model(path)
    return _model_cache[path]


# ==========================================================================================
#                                IMAGE PREPROCESSING
# ==========================================================================================

def preprocess_dynamic(img, w, h):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (w, h))
    img = np.expand_dims(img, 0).astype("float32") / 255.0
    return img


def get_class_from_index(model_used, idx):
    mapping = MODEL_CLASS_INDEX.get(model_used.lower(), {})
    for cls, i in mapping.items():
        if i == idx:
            return cls
    return None


# ==========================================================================================
#                                      ROUTES
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
        return jsonify({"success": False, "error": f"No {model_type} model found for this class"})

    model_used = model_used.lower()
    model_path = get_model_path(model_type, model_used)

    if not model_path:
        return jsonify({"success": False, "error": f"Model file missing: {model_used}"})

    model = load_model_cached(model_path)

    _, h, w, _ = model.input_shape
    img = Image.open(filepath)
    x = preprocess_dynamic(img, w, h)

    pred = model.predict(x).squeeze()
    pred = softmax(pred).numpy() if np.sum(pred) == 0 else pred

    idx = int(np.argmax(pred))
    label = get_class_from_index(model_used, idx)
    conf = float(np.max(pred))

    nutrition = NUTRITION_JSON.get(normalize(selected_class), {})

    confusion_matrix_full = class_entry.get("confusion_matrix_full")
    model_labels_order = list(MODEL_CLASS_INDEX.get(model_used.lower(), {}).keys())

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
        "model_labels_order": model_labels_order
    })


@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)
