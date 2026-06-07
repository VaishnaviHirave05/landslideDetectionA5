import os
import uuid
import hashlib
import numpy as np
import mysql.connector
import requests
import re

from flask import Flask, request, render_template, redirect, url_for, session, flash
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps

# This helper class fixes legacy model configs where DepthwiseConv2D was serialized
# with an unsupported 'groups' argument on newer TensorFlow/Keras versions.
class DepthwiseConv2DFixed(DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("groups", None)
        return super().from_config(config)

CUSTOM_LOAD_KWARGS = {"custom_objects": {"DepthwiseConv2D": DepthwiseConv2DFixed}}

# -------------------- APP CONFIG --------------------
app = Flask(__name__)
app.secret_key = "123456"

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "landslide_model.h5"
LABELS_PATH = "labels.txt"

np.set_printoptions(suppress=True)

# -------------------- DATABASE --------------------

def get_db_connection():
    try:
        return mysql.connector.connect(
            host=os.getenv("MYSQLHOST"),
            user=os.getenv("MYSQLUSER", "root"),
            password=os.getenv("MYSQLPASSWORD"),
            database=os.getenv("MYSQLDATABASE"),
            port=int(os.getenv("MYSQLPORT", 3306))
        )
    except mysql.connector.Error as err:
        print("DB Error:", err)
        return None
# -------------------- LOAD MODEL --------------------
try:
    model = load_model(MODEL_PATH, compile=False, **CUSTOM_LOAD_KWARGS)
except Exception as e:
    print("⚠️ Failed to load primary model:", e)
    alt_path = "keras_model.h5"
    print(f"Attempting to load alternative model: {alt_path}")
    model = load_model(alt_path, compile=False, **CUSTOM_LOAD_KWARGS)
# Ensure model has a single input tensor (flatten list if needed)
# Ensure model has a single input tensor (flatten list if needed)
if isinstance(model.input, list) and len(model.input) > 1:
    model = tf.keras.models.Model(inputs=model.input[0], outputs=model.output)
# Keep raw lines (with newlines) exactly as Teachable Machine expects
class_names = open(LABELS_PATH, "r").readlines()

print("✅ Model & labels loaded")

UNET_PATHS = ["unet_final.h5", "landslide_unet_final.h5", "landslide_unet.h5"]
unet_model = None
for _p in UNET_PATHS:
    if os.path.exists(_p):
        try:
            unet_model = load_model(_p, compile=False, **CUSTOM_LOAD_KWARGS)
            print(f"✅ Loaded UNet model: {_p}")
            break
        except Exception as _e:
            print("⚠️ Failed to load UNet model", _p, _e)
if unet_model is None:
    print("ℹ️ No UNet model found or failed to load - segmentation overlay will be skipped")

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("index.html")

# -------------------- REGISTER --------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = hashlib.sha256(
            request.form["password"].encode()
        ).hexdigest()

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            # Check duplicate username or email
            cur.execute(
                "SELECT * FROM users WHERE username=%s OR email=%s",
                (username, email)
            )
            existing_user = cur.fetchone()

            if existing_user:
                flash('Username or Email already exists!', 'error')
                return redirect(url_for('register'))

            # Insert new user
            cur.execute(
                "INSERT INTO users(username,email,password) VALUES(%s,%s,%s)",
                (username, email, password)
            )
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

        finally:
            cur.close()
            conn.close()

    return render_template("register.html")

# -------------------- LOGIN --------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = hashlib.sha256(request.form["password"].encode()).hexdigest()

        conn = get_db_connection()
        if conn is None:
            flash("Database unavailable. Please try again later.", "error")
            return render_template("login.html")
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT * FROM users WHERE username=%s AND password=%s",
                (username, password)
            )
            user = cur.fetchone()
        finally:
            cur.close()
            conn.close()

        if user:
            session["loggedin"] = True
            session["id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))

        flash("Invalid login", "error")

    return render_template("login.html")

# -------------------- LOGOUT --------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# -------------------- DASHBOARD --------------------
@app.route('/dashboard')
def dashboard():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Total scans for this user
    cursor.execute(
        "SELECT COUNT(*) AS c FROM history WHERE user_id=%s",
        (session['id'],)
    )
    total_scans = cursor.fetchone()['c']

    # Landslides detected
    cursor.execute(
        "SELECT COUNT(*) AS c FROM history WHERE user_id=%s AND result LIKE %s",
        (session['id'], '%Landslide%')
    )
    danger_count = cursor.fetchone()['c']

    # Safe areas
    safe_count = total_scans - danger_count

    # Recent history (last 5 scans)
    cursor.execute(
        """
        SELECT prediction_date, result, confidence
        FROM history
        WHERE user_id=%s
        ORDER BY prediction_date DESC
        LIMIT 5
        """,
        (session['id'],)
    )
    recent_history = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template(
        'dashboard.html',
        total_scans=total_scans,
        danger_count=danger_count,
        safe_count=safe_count,
        recent_history=recent_history
    )


# -------------------- ANALYSIS (FIXED PREDICTION) --------------------
@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    if "loggedin" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        # ---------- IMAGE INPUT ----------
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            filename = str(uuid.uuid4())[:8] + "_" + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            file.save(filepath)

        elif request.form.get("image_url"):
            filename = str(uuid.uuid4())[:8] + "_url.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            response = requests.get(request.form["image_url"], stream=True)
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        else:
            flash("Please upload image", "error")
            return redirect(request.url)

        # ---------- PREDICTION (Teachable Machine style) ----------
        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Open and convert to RGB
        image = Image.open(filepath).convert("RGB")

        # Resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # Turn the image into a numpy array
        image_array = np.asarray(image)
        print("🔍 Image stats - shape:", image_array.shape, "dtype:", image_array.dtype,
              "min:", image_array.min(), "max:", image_array.max(), "mean:", float(image_array.mean()))

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        print("🔍 Normalized stats - min:", float(normalized_image_array.min()),
              "max:", float(normalized_image_array.max()), "mean:", float(normalized_image_array.mean()))

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        print("🔍 Raw prediction vector:", prediction)
        index = np.argmax(prediction)
        class_name = class_names[index]          # raw line e.g. "0 landslide\n"
        confidence_score = prediction[0][index]

        # Strip the leading index prefix ("0 " or "1 ") exactly as Teachable Machine outputs
        # labels.txt format: "0 landslide"  / "1 nonlandslide"
        raw = class_name.strip()  # remove newline
        if len(raw) > 2 and raw[0].isdigit() and raw[1] == " ":
            predicted_label = raw[2:].strip()
        else:
            predicted_label = raw
        confidence = f"{confidence_score * 100:.2f}"
        print("Class:", predicted_label, end="")
        print(" Confidence Score:", confidence_score)

        # ---------- RESULT ----------
        lower_label = predicted_label.lower()
        # If the label contains an explicit negative/neutral keyword, treat as safe
        if re.search(r'\b(no|not|safe|none)\b', lower_label):
            result = "Safe Area"
            alert_class = "status-success"
        elif re.search(r'\blandslide\b', lower_label):
            result = "Landslide Detected"
            alert_class = "status-danger"
        else:
            # Fallback: treat unknown labels as safe
            result = "Safe Area"
            alert_class = "status-success"
        print("🔍 Label:", predicted_label, "Decision:", result)

        # ---------- OPTIONAL: GENERATE SEGMENTATION MASK + OVERLAY ----------
        overlay_filename = None
        mask_filename = None
        try:
            if unet_model is not None:
                # Determine model input size (height, width)
                inp_shape = unet_model.input_shape
                # Typical shapes: (None, H, W, C) or (H, W, C)
                if inp_shape is None:
                    target_h, target_w = 256, 256
                elif len(inp_shape) == 4:
                    target_h, target_w = int(inp_shape[1] or 256), int(inp_shape[2] or 256)
                elif len(inp_shape) == 3:
                    target_h, target_w = int(inp_shape[0] or 256), int(inp_shape[1] or 256)
                else:
                    target_h, target_w = 256, 256

                mask_image = ImageOps.fit(Image.open(filepath).convert("RGB"), (target_w, target_h), Image.Resampling.LANCZOS)
                mask_arr = np.asarray(mask_image).astype(np.float32) / 255.0
                # If model expects single channel, pick appropriate slice
                x_in = np.expand_dims(mask_arr, axis=0)
                # Some unet models expect single-channel input; attempt channel-reduction if needed
                if unet_model.input_shape[-1] == 1 and x_in.shape[-1] == 3:
                    x_in = np.mean(x_in, axis=-1, keepdims=True)

                pred_mask = unet_model.predict(x_in)[0]
                # If multi-channel mask -> take argmax across channels
                if pred_mask.ndim == 3 and pred_mask.shape[-1] > 1:
                    mask_prob = np.max(pred_mask, axis=-1)
                else:
                    mask_prob = pred_mask[..., 0] if pred_mask.ndim == 3 else pred_mask

                # Normalize and threshold
                mask_prob = np.clip(mask_prob, 0.0, 1.0)
                mask_bin = (mask_prob >= 0.5).astype(np.uint8) * 255

                # Save mask image (grayscale)
                mask_filename = filename.rsplit('.', 1)[0] + "_mask.png"
                mask_path_full = os.path.join(UPLOAD_FOLDER, mask_filename)
                Image.fromarray(mask_bin).convert('L').save(mask_path_full)

                # Create colored overlay (red) and blend with original image
                orig_for_overlay = ImageOps.fit(Image.open(filepath).convert("RGBA"), (target_w, target_h), Image.Resampling.LANCZOS)
                mask_rgba = Image.fromarray(np.stack([mask_bin, np.zeros_like(mask_bin), np.zeros_like(mask_bin), (mask_bin * 0.6).astype(np.uint8)], axis=-1), mode='RGBA')
                overlay_img = Image.alpha_composite(orig_for_overlay, mask_rgba)

                overlay_filename = filename.rsplit('.', 1)[0] + "_overlay.png"
                overlay_path_full = os.path.join(UPLOAD_FOLDER, overlay_filename)
                overlay_img.save(overlay_path_full)

        except Exception as _e:
            print("⚠️ Failed to generate segmentation overlay:", _e)
            overlay_filename = None
            mask_filename = None

        # ---------- SAVE HISTORY ----------
        conn = get_db_connection()
        if conn is None:
            flash("Warning: could not save history (database unavailable).", "warning")
        else:
            cur = conn.cursor()
            try:
                cur.execute(
                    """INSERT INTO history 
                       (user_id, original_image, result, confidence)
                       VALUES (%s,%s,%s,%s)""",
                    (session["id"], filename, result, confidence)
                )
                conn.commit()
            finally:
                cur.close()
                conn.close()

        return render_template(
            "analysis.html",
            result=result,
            image_path=filename,
            confidence=confidence,
            alert_class=alert_class,
            overlay_path=(overlay_filename or filename),
            mask_path=(mask_filename or filename)
        )

    return render_template("analysis.html")

# -------------------- HISTORY --------------------
@app.route("/history")
def history():
    if "loggedin" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    if conn is None:
        flash("Database unavailable. Unable to load history.", "error")
        return redirect(url_for("dashboard"))
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            "SELECT * FROM history WHERE user_id=%s ORDER BY id DESC",
            (session["id"],)
        )
        data = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    return render_template("history.html", history_items=data)

# -------------------- RUN --------------------
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
