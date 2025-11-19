import os
import json
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import time

# ===================================================================
#  APP INIT
# ===================================================================

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

TFLITE_DIR = "tflite_models"
os.makedirs(TFLITE_DIR, exist_ok=True)

# Google Cloud for Custom + VGG models
GCS_BUCKET = "https://storage.googleapis.com/food-classification-models-bucket"

# ResNet API Backend (Stored in another Render project)
RESNET_API_URL = "https://resnet-api-creation.onrender.com/predict"

# ===================================================================
#  NORMALIZE STRINGS
# ===================================================================

def normalize(x: str):
    return x.lower().strip().replace(" ", "_")

# ===================================================================
#  MODEL → CLASS MAPPING (Custom + VGG)
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

MODEL_CLASS_INDEX = {
    m.lower(): {normalize(k): v for k, v in mapping.items()}
    for m, mapping in RAW_MODEL_CLASS_INDEX.items()
}

# ===================================================================
#  CLASS JSON
# ===================================================================

CLASS_JSON = json.load(open("class.json"))
CLASS_LIST = list(CLASS_JSON.keys())

# ===================================================================
#  METRIC JSON (Custom + VGG)
# ===================================================================

MODEL_EVAL_JSON = {
    "custom_model": json.load(open("model_evaluation_results.json")),
    "vgg_model": json.load(open("model_evaluation_results_vgg.json")),
}

for mtype, data in MODEL_EVAL_JSON.items():
    MODEL_EVAL_JSON[mtype] = {normalize(k): v for k, v in data.items()}

# ===================================================================
#  TFLITE HANDLING (ONLY ONE MODEL KEPT)
# ===================================================================

tflite_cache = {}

def clear_tflite_folder():
    for f in os.listdir(TFLITE_DIR):
        if f.endswith(".tflite"):
            try:
                os.remove(os.path.join(TFLITE_DIR, f))
            except:
                pass

def download_tflite_from_gcs(model_name):
    clear_tflite_folder()

    url = f"{GCS_BUCKET}/{model_name}.tflite"
    save_path = os.path.join(TFLITE_DIR, f"{model_name}.tflite")

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return save_path
    except:
        return None

def load_tflite_model(model_name):
    global tflite_cache
    tflite_cache = {}

    model_path = os.path.join(TFLITE_DIR, f"{model_name}.tflite")

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

def preprocess(img, size):
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
        # 🔥 RESNET → EXTERNAL API WITH 90 SECOND WAIT
        # ======================================================
        if model_type == "resnet_model":

            for attempt in range(3):
                print(f"\n[ResNet] Attempt {attempt+1}/3 sending request...")

                try:
                    with open(fpath, "rb") as f:
                        resp = requests.post(
                            RESNET_API_URL,
                            files={"file": (fname, f, img_file.mimetype)},
                            data={"model_type": model_type,
                                  "selected_class": selected_class},
                            timeout=200
                        )
                except Exception as e:
                    print("[ResNet] Network error:", e)
                    print("[ResNet] Waiting 90 sec before retry...")
                    time.sleep(90)
                    continue

                # If server is waking up → 503
                if resp.status_code == 503:
                    print("[ResNet] API cold start → Waiting 90 sec...")
                    time.sleep(90)
                    continue

                if resp.status_code == 200:
                    try:
                        return jsonify(resp.json())
                    except:
                        return jsonify({
                            "success": False,
                            "error": "ResNet API returned invalid JSON format"
                        }), 502

                print(f"[ResNet] Server error: {resp.status_code}")
                print("[ResNet] Waiting 90 sec...")
                time.sleep(90)

            return jsonify({
                "success": False,
                "error": "ResNet API failed after 3 attempts"
            }), 502

        # ======================================================
        # CUSTOM + VGG → LOCAL TFLITE INFERENCE
        # ======================================================
        if model_type not in MODEL_EVAL_JSON:
            return jsonify({"success": False, "error": f"Invalid model_type: {model_type}"}), 400

        class_info = MODEL_EVAL_JSON[model_type].get(cname)
        if class_info is None:
            return jsonify({"success": False, "error": f"Class not found: {cname}"}), 400

        model_used = class_info["model_used"].lower()

        interpreter = load_tflite_model(model_used)
        if interpreter is None:
            return jsonify({"success": False, "error": "Failed to load model"}), 500

        input_info = interpreter.get_input_details()[0]
        _, h, w, _ = input_info["shape"]

        img = Image.open(fpath).convert("RGB")
        x = preprocess(img, w)

        interpreter.set_tensor(input_info["index"], x)
        interpreter.invoke()

        output = interpreter.get_output_details()[0]
        preds = interpreter.get_tensor(output["index"]).squeeze().astype(float)

        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        class_map = MODEL_CLASS_INDEX[model_used]
        predicted_label = next(k for k, v in class_map.items() if v == idx)

        predicted_norm = normalize(predicted_label)
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
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except:
                pass

# ===================================================================
#  MAIN
# ===================================================================

if __name__ == "__main__":
    app.run(debug=True)
