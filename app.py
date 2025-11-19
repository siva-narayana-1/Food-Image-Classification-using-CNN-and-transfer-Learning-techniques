import os
import json
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

# ===================================================================
#  APP INIT
# ===================================================================

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

TFLITE_DIR = "tflite_models"
os.makedirs(TFLITE_DIR, exist_ok=True)

# Google Cloud Storage bucket
GCS_BUCKET = "https://storage.googleapis.com/food-classification-models-bucket"

# External ResNet API
RESNET_API_URL = "https://resnet-api-creation.onrender.com/predict"

# ===================================================================
#  NORMALIZER
# ===================================================================

def normalize(x: str):
    """Lowercase and replace spaces with underscores."""
    return x.lower().strip().replace(" ", "_")

# ===================================================================
#  MODEL → CLASS INDEX MAPPING
#  (Only custom + vgg here; resnet runs in external API)
# ===================================================================

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

# Normalized mapping
MODEL_CLASS_INDEX = {
    m.lower(): {normalize(k): v for k, v in mapping.items()}
    for m, mapping in RAW_MODEL_CLASS_INDEX.items()
}

# ===================================================================
#  LOAD CLASS LIST
# ===================================================================

CLASS_JSON = json.load(open("class.json"))
CLASS_LIST = list(CLASS_JSON.keys())

# ===================================================================
#  LOAD METRIC JSONS (only custom + vgg; resnet handled by resnet API)
# ===================================================================

MODEL_EVAL_JSON = {
    "custom_model": json.load(open("model_evaluation_results.json")),
    "vgg_model": json.load(open("model_evaluation_results_vgg.json")),
}

# Normalize keys (apple_pie -> apple_pie, Fries -> fries, etc.)
for mtype, data in MODEL_EVAL_JSON.items():
    MODEL_EVAL_JSON[mtype] = {normalize(k): v for k, v in data.items()}

# ===================================================================
#  AUTO-DOWNLOAD TFLITE MODELS (ONLY 1 IN FOLDER)
# ===================================================================

tflite_cache = {}

def clear_tflite_folder():
    """Keep only one .tflite file in folder."""
    for f in os.listdir(TFLITE_DIR):
        if f.endswith(".tflite"):
            try:
                os.remove(os.path.join(TFLITE_DIR, f))
                print("[CLEAN] Removed old tflite:", f)
            except Exception:
                pass

def download_tflite_from_gcs(model_name: str):
    """
    Download model from Google Cloud Bucket → /tflite_models/
    Only 1 tflite is kept at any time.
    """
    clear_tflite_folder()

    url = f"{GCS_BUCKET}/{model_name}.tflite"
    save_path = os.path.join(TFLITE_DIR, f"{model_name}.tflite")

    print(f"[INFO] Downloading from GCS: {url}")

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(resp.content)
        print(f"[INFO] Saved: {save_path}")
        return save_path
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return None

def load_tflite_model(model_name: str):
    """
    Load or download + cache only 1 model in memory.
    """
    global tflite_cache

    # Always reset memory cache (we only keep one model)
    tflite_cache = {}

    model_path = os.path.join(TFLITE_DIR, model_name + ".tflite")

    # Download if missing
    if not os.path.exists(model_path):
        ok = download_tflite_from_gcs(model_name)
        if ok is None:
            return None

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    tflite_cache[model_name] = interpreter

    return interpreter

# ===================================================================
#  IMAGE PREPROCESS
# ===================================================================

def preprocess(img: Image.Image, size: int):
    img = img.resize((size, size))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

# ===================================================================
#  ROUTES
# ===================================================================

@app.route("/")
def index():
    return render_template("index.html", classes=CLASS_LIST)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"})

    img_file = request.files["file"]
    fname = secure_filename(img_file.filename)
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    img_file.save(fpath)

    try:
        model_type = request.form.get("model_type", "").lower()
        selected_class = request.form.get("selected_class", "").strip()
        cname = normalize(selected_class)

        # ======================================================
        # 1) RESNET → Forward to external API (no local TFLite)
        # ======================================================
        if model_type == "resnet_model":
            try:
                with open(fpath, "rb") as f:
                    files = {"file": (fname, f, img_file.mimetype)}
                    data = {
                        "model_type": model_type,
                        "selected_class": selected_class,
                    }
                    resp = requests.post(
                        RESNET_API_URL,
                        files=files,
                        data=data,
                        timeout=120,
                    )

                if resp.status_code != 200:
                    return jsonify({
                        "success": False,
                        "error": f"ResNet API error: {resp.status_code}"
                    }), 502

                try:
                    res_json = resp.json()
                except ValueError:
                    return jsonify({
                        "success": False,
                        "error": "ResNet API returned invalid JSON"
                    }), 502

                # Directly return ResNet API JSON (same fields expected by frontend)
                return jsonify(res_json)

            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"ResNet API request failed: {e}"
                }), 502

        # ======================================================
        # 2) CUSTOM / VGG → Local TFLite inference
        # ======================================================
        if model_type not in MODEL_EVAL_JSON:
            return jsonify({
                "success": False,
                "error": f"Unsupported model_type: {model_type}"
            }), 400

        # Get metrics entry for SELECTED class (only to find which model to load)
        class_info = MODEL_EVAL_JSON[model_type].get(cname)
        if class_info is None:
            return jsonify({
                "success": False,
                "error": f"Class not in JSON: {cname}"
            }), 400

        model_used = class_info["model_used"].lower()

        interpreter = load_tflite_model(model_used)
        if interpreter is None:
            return jsonify({
                "success": False,
                "error": "Model download/load failed"
            }), 500

        input_info = interpreter.get_input_details()[0]
        _, h, w, _ = input_info["shape"]

        img = Image.open(fpath).convert("RGB")
        x = preprocess(img, w)

        interpreter.set_tensor(input_info["index"], x)
        interpreter.invoke()

        output_info = interpreter.get_output_details()[0]
        preds = interpreter.get_tensor(output_info["index"]).squeeze().astype(float)
        preds = np.nan_to_num(preds)

        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        # Decode predicted label
        class_map = MODEL_CLASS_INDEX[model_used]
        predicted_label = next(k for k, v in class_map.items() if v == idx)
        predicted_norm = normalize(predicted_label)

        # Use METRICS of PREDICTED class
        predicted_info = MODEL_EVAL_JSON[model_type].get(predicted_norm, {})

        labels = list(class_map.keys())
        matrix = predicted_info.get("confusion_matrix_full", [])

        return jsonify({
            "success": True,
            "selected_class": selected_class,
            "predicted_label": predicted_label,
            "confidence": confidence,

            "accuracy": predicted_info.get("accuracy", "NA"),
            "precision": predicted_info.get("precision", "NA"),
            "recall": predicted_info.get("recall", "NA"),
            "f1_score": predicted_info.get("f1_score", "NA"),

            "confusion_matrix_labels": labels,
            "confusion_matrix_full": matrix,

            "model_used": model_used
        })

    finally:
        # ALWAYS delete uploaded image (temp only)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                print("[CLEANUP] Deleted uploaded file:", fpath)
            except Exception:
                pass

# ===================================================================
#  MAIN
# ===================================================================

if __name__ == "__main__":
    app.run(debug=True)
