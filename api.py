import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer

# -----------------------------------------------------------
# PATHS
# -----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# -----------------------------------------------------------
# REQUEST MODEL
# -----------------------------------------------------------

class PredictRequest(BaseModel):
    text: str
    model: Optional[str] = None

# -----------------------------------------------------------
# FASTAPI APP INIT
# -----------------------------------------------------------

app = FastAPI(title="Jigsaw Toxic Rule Violation – API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# -----------------------------------------------------------
# MODEL REGISTRY
# -----------------------------------------------------------

MODEL_REGISTRY: Dict[str, Any] = {}
MODEL_METADATA: Dict[str, Any] = {}

# -----------------------------------------------------------
# LOAD MODELS SAFELY
# -----------------------------------------------------------

def _has_files(p: Path) -> bool:
    try:
        return p.exists() and any(p.iterdir())
    except Exception:
        return False


def safe_load_models():
    global MODEL_REGISTRY, MODEL_METADATA
    MODEL_REGISTRY = {}
    MODEL_METADATA = {}

    print("\n=== CARGANDO MODELOS ===")

    for folder in MODELS_DIR.iterdir():
        if not folder.is_dir():
            continue

        name = folder.name
        print(f"\n-> Cargando: {name}")

        info: Dict[str, Any] = {
            "type": None,
            "model": None,
            "tokenizer": None,
            "input_keys": None,
            "seq_len": None,
        }

        # ---- metadata.json ----
        metadata_path = folder / "metadata.json"
        if metadata_path.exists():
            try:
                MODEL_METADATA[name] = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                MODEL_METADATA[name] = None

        # ---- modelos sklearn (.joblib) ----
        joblib_files = list(folder.glob("*.joblib")) + list(folder.glob("*.pkl"))
        if joblib_files:
            try:
                info["model"] = joblib.load(joblib_files[0])
                info["type"] = "sklearn"
                MODEL_REGISTRY[name] = info
                print("   ✓ sklearn cargado")
            except Exception as e:
                print("   ✗ ERROR cargando sklearn:", e)
            continue

        # ---- BiLSTM deshabilitado (mantener por ahora) ----
        if name == "models_bilstm":
            print("   ✗ BiLSTM deshabilitado (Lambda sin output_shape)")
            continue

        # ---- TensorFlow SavedModel ----
        saved_model_path = folder / "saved_model.pb"
        if saved_model_path.exists():
            try:
                saved = tf.saved_model.load(str(folder))
                fn = saved.signatures["serving_default"]
                input_sig = fn.structured_input_signature[1]
                input_keys = list(input_sig.keys())

                # detectar seq_len fija (ej. 160)
                seq_len = None
                for spec in input_sig.values():
                    if len(spec.shape) == 2 and spec.shape[1] is not None:
                        seq_len = int(spec.shape[1])
                        break

                info["type"] = "tensorflow"
                info["model"] = fn
                info["input_keys"] = input_keys
                info["seq_len"] = seq_len

                print("   ✓ SavedModel cargado (tf.saved_model.load)")
                print(f"   · input keys: {input_keys}")
                if seq_len is not None:
                    print(f"   · seq_len detectado: {seq_len}")

            except Exception as e:
                print("   ✗ ERROR cargando SavedModel:", e)
                continue

            # tokenizer de HuggingFace (solo si existe y no está vacío)
            tok_dir = folder / "tokenizer"
            if _has_files(tok_dir):
                try:
                    info["tokenizer"] = AutoTokenizer.from_pretrained(
                        str(tok_dir),
                        local_files_only=True,
                    )
                    print("   ✓ tokenizer cargado")
                except Exception as e:
                    print("   ✗ ERROR cargando tokenizer:", e)
            else:
                print("   ✗ tokenizer no encontrado o vacío -> modelo omitido")
                continue

            MODEL_REGISTRY[name] = info
            continue

    print("\nModelos cargados:", list(MODEL_REGISTRY.keys()))
    print("=== FIN CARGA ===")

@app.on_event("startup")
def startup_event():
    safe_load_models()

# -----------------------------------------------------------
# PREDICCIÓN SKLEARN (versión robusta que usabas antes)
# -----------------------------------------------------------

def predict_sklearn(pipeline, text: str) -> Dict[str, Any]:
    try:
        # 1) predict_proba si existe
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba([text])[0]
        else:
            # 2) decision_function -> lo convertimos a probs
            if hasattr(pipeline, "decision_function"):
                df = pipeline.decision_function([text])
                df = np.array(df)
                if df.ndim == 1:
                    prob_pos = 1 / (1 + np.exp(-df[0]))
                    probs = np.array([1 - prob_pos, prob_pos])
                else:
                    exps = np.exp(df[0] - np.max(df[0]))
                    probs = exps / exps.sum()
            else:
                # 3) fallback: solo predict
                pred = pipeline.predict([text])[0]
                classes = getattr(pipeline, "classes_", None)
                if classes is not None:
                    probs = np.zeros(len(classes))
                    probs[list(classes).index(pred)] = 1.0
                else:
                    probs = np.array([1.0])

        # obtener classes_ desde el pipeline o algún step interno
        classes = getattr(pipeline, "classes_", None)
        if classes is None and hasattr(pipeline, "named_steps"):
            for _, step in pipeline.named_steps.items():
                if hasattr(step, "classes_"):
                    classes = step.classes_
                    break

        if isinstance(probs, np.ndarray) and probs.size > 1 and classes is not None:
            idx = int(np.argmax(probs))
            label = str(classes[idx])
            confidence = float(probs[idx])
            probs_list = probs.tolist()
        else:
            label = str(pipeline.predict([text])[0])
            confidence = 1.0
            probs_list = probs.tolist() if hasattr(probs, "tolist") else [float(probs)]

        return {
            "label": label,
            "label_index": int(label) if str(label).isdigit() else 0,
            "confidence": confidence,
            "probs": probs_list,
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# -----------------------------------------------------------
# PREDICCIÓN TENSORFLOW (maneja softmax y sigmoide)
# -----------------------------------------------------------

def predict_tensorflow(fn, tokenizer, input_keys, seq_len, text: str) -> Dict[str, Any]:
    try:
        # tokenización con longitud fija si el modelo la tiene
        if seq_len is not None:
            enc = tokenizer(
                text,
                return_tensors="tf",
                padding="max_length",
                truncation=True,
                max_length=seq_len,
            )
        else:
            enc = tokenizer(
                text,
                return_tensors="tf",
                padding=True,
                truncation=True,
            )

        enc_dict = {k: v for k, v in enc.items()}
        inputs_tf = {k: enc_dict[k] for k in input_keys if k in enc_dict}

        outputs = fn(**inputs_tf)

        # primera salida
        if "logits" in outputs:
            raw = outputs["logits"].numpy()
        else:
            raw = list(outputs.values())[0].numpy()

        raw = np.array(raw)

        # ---- caso sigmoide: salida [batch] o [batch, 1] ----
        if raw.ndim == 1 or (raw.ndim == 2 and raw.shape[1] == 1):
            if raw.ndim == 2:
                v = float(raw[0, 0])
            else:
                v = float(raw[0])

            # si ya está en [0,1] lo tomamos tal cual, si no aplicamos sigmoide
            if 0.0 <= v <= 1.0:
                p1 = v
            else:
                p1 = float(tf.sigmoid(v).numpy())

            p0 = 1.0 - p1
            probs = np.array([p0, p1], dtype=np.float32)

        # ---- caso softmax: [batch, num_classes] ----
        else:
            probs = tf.nn.softmax(raw, axis=-1).numpy()[0]

        idx = int(np.argmax(probs))

        return {
            "label": str(idx),
            "label_index": idx,
            "confidence": float(probs[idx]),
            "probs": probs.tolist(),
        }

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# -----------------------------------------------------------
# ROUTES
# -----------------------------------------------------------

@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"message": "Place static/index.html"}, status_code=404)

@app.get("/models")
def list_models():
    return {"models": list(MODEL_REGISTRY.keys()), "metadata": MODEL_METADATA}

@app.post("/predict")
def predict(req: PredictRequest):
    if not req.model:
        raise HTTPException(400, "Debes seleccionar un modelo")

    if req.model not in MODEL_REGISTRY:
        raise HTTPException(404, f"Modelo '{req.model}' no encontrado")

    info = MODEL_REGISTRY[req.model]

    # sklearn
    if info["type"] == "sklearn":
        result = predict_sklearn(info["model"], req.text)
        if "error" in result:
            raise HTTPException(500, result)
        return {"model": req.model, "result": result}

    # TensorFlow
    if info["type"] == "tensorflow":
        result = predict_tensorflow(
            fn=info["model"],
            tokenizer=info["tokenizer"],
            input_keys=info["input_keys"],
            seq_len=info["seq_len"],
            text=req.text,
        )
        if "error" in result:
            raise HTTPException(500, result)
        return {"model": req.model, "result": result}

    raise HTTPException(500, "Tipo de modelo no soportado")

# -----------------------------------------------------------
# STATIC
# -----------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
