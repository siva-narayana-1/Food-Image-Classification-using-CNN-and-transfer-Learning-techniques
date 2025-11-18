import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

# ===================================================================
#  APP SETUP
# ===================================================================

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

TFLITE_DIR = "tflite_models"
os.makedirs(TFLITE_DIR, exist_ok=True)

# ===================================================================
#  NORMALIZER
# ===================================================================

def normalize(x: str):
    """Lowercase and replace spaces with underscores."""
    return x.lower().strip().replace(" ", "_")

# ===================================================================
#  MODEL → CLASS INDEX MAPPING
# ===================================================================

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
    "vgg_model_11": {'sandwich': 0, 'sushi': 1, 'taco': 2, 'taquito': 3},
}

# Normalize mapping
MODEL_CLASS_INDEX = {
    m.lower(): {normalize(k): v for k, v in mapping.items()}
    for m, mapping in RAW_MODEL_CLASS_INDEX.items()
}

# ===================================================================
#  LOAD CLASSES
# ===================================================================

CLASS_JSON = json.load(open("class.json"))
CLASS_LIST = list(CLASS_JSON.keys())

# ===================================================================
#  LOAD METRICS JSON + NORMALIZE KEYS
# ===================================================================

MODEL_EVAL_JSON = {
    "custom_model": json.load(open("model_evaluation_results.json")),
    "resnet_model": json.load(open("model_evaluation_results_resnet.json")),
    "vgg_model": json.load(open("model_evaluation_results_vgg.json"))
}

# Fix keys (lowercase + underscores)
for model_type, content in MODEL_EVAL_JSON.items():
    fixed = {}
    for key, val in content.items():
        fixed[normalize(key)] = val
    MODEL_EVAL_JSON[model_type] = fixed

# ===================================================================
#  TFLITE CACHING (ONLY 1 MODEL IN MEMORY)
# ===================================================================

tflite_cache = {}

def load_tflite_model(model_name):
    global tflite_cache

    # Clear if loading a different model
    if model_name not in tflite_cache:
        tflite_cache = {}
        path = os.path.join(TFLITE_DIR, model_name + ".tflite")

        if not os.path.exists(path):
            print("[ERROR] Missing TFLite file:", path)
            return None

        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        tflite_cache[model_name] = interpreter

    return tflite_cache[model_name]

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
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    img_file.save(file_path)

    model_type = request.form.get("model_type").lower()
    selected_class = request.form.get("selected_class", "").strip()
    cname = normalize(selected_class)

    # Load JSON metrics for selected class
    class_info = MODEL_EVAL_JSON[model_type].get(cname)
    if class_info is None:
        return jsonify({"success": False, "error": "Class not found in JSON", "class_lookup": cname})

    model_used = class_info["model_used"].lower()

    interpreter = load_tflite_model(model_used)
    if interpreter is None:
        return jsonify({"success": False, "error": "Model missing on server"})

    input_info = interpreter.get_input_details()[0]
    _, h, w, _ = input_info["shape"]

    img = Image.open(file_path).convert("RGB")
    x = preprocess(img, w)

    interpreter.set_tensor(input_info["index"], x)
    interpreter.invoke()

    output = interpreter.get_output_details()[0]
    preds = interpreter.get_tensor(output["index"]).squeeze().astype(float)
    preds = np.nan_to_num(preds)

    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    class_map = MODEL_CLASS_INDEX[model_used]
    predicted_label = next(key for key, val in class_map.items() if val == idx)
    predicted_label_norm = normalize(predicted_label)

    predicted_info = MODEL_EVAL_JSON[model_type].get(predicted_label_norm, {})

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
        "true_positive": predicted_info.get("true_positive", "NA"),
        "false_positive": predicted_info.get("false_positive", "NA"),
        "false_negative": predicted_info.get("false_negative", "NA"),
        "true_negative": predicted_info.get("true_negative", "NA"),

        "confusion_matrix_labels": labels,
        "confusion_matrix_full": matrix,

        "model_used": model_used
    })

# ===================================================================
#  MAIN
# ===================================================================

if __name__ == "__main__":
    app.run(debug=True)
