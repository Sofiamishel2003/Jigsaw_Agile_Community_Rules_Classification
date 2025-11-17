from pathlib import Path
import json
import traceback
from typing import Dict, Any, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
STATIC_DIR = BASE_DIR / 'static'


class PredictRequest(BaseModel):
    text: str
    model: Optional[str] = None


app = FastAPI(title='Jigsaw Models API')

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def find_model_files(directory: Path) -> Dict[str, Path]:
    files = {}
    if not directory.exists():
        return files
    for p in directory.glob('*.joblib'):
        name = p.stem
        files[name] = p
    return files


def load_metadata(directory: Path) -> Dict[str, Any]:
    meta = {}
    if not directory.exists():
        return meta
    for p in directory.glob('*_metadata.json'):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            key = p.stem.replace('_metadata', '')
            meta[key] = data
        except Exception:
            continue
    return meta


_MODEL_REGISTRY: Dict[str, Any] = {}
_METADATA: Dict[str, Any] = {}


def safe_load_models():
    global _MODEL_REGISTRY, _METADATA
    _MODEL_REGISTRY = {}
    model_files = find_model_files(MODELS_DIR)
    for name, path in model_files.items():
        try:
            _MODEL_REGISTRY[name] = joblib.load(path)
        except Exception:
            _MODEL_REGISTRY[name] = None
    _METADATA = load_metadata(MODELS_DIR)


import json
import joblib
import tensorflow as tf
from transformers import AutoTokenizer
from pathlib import Path

def safe_load_models2():
    global _MODEL_REGISTRY, _METADATA
    _MODEL_REGISTRY = {}
    _METADATA = {}

    # Buscar carpetas dentro de MODELS_DIR
    for model_dir in Path(MODELS_DIR).glob("*"):
        if not model_dir.is_dir():
            continue

        name = model_dir.name
        model_info = {}

        # --- 1. Cargar metadata.json ---
        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf8") as f:
                    model_info["metadata"] = json.load(f)
            except:
                model_info["metadata"] = None

        # --- 2. Cargar modelo .joblib (model.pkl, model.joblib, etc.) ---
        joblib_files = list(model_dir.glob("*.joblib")) + list(model_dir.glob("*.pkl"))
        if joblib_files:
            try:
                model_info["model"] = joblib.load(joblib_files[0])
                model_info["type"] = "joblib"
            except:
                model_info["model"] = None

        # --- 3. Cargar modelo TensorFlow (.keras) ---
        keras_files = list(model_dir.glob("*.keras"))
        if keras_files:
            try:
                model_info["model"] = tf.keras.models.load_model(keras_files[0], compile=True)
                model_info["type"] = "tensorflow"
            except:
                model_info["model"] = None

        # --- 4. Cargar tokenizer de HuggingFace ---
        tokenizer_dir = model_dir / "tokenizer"
        if tokenizer_dir.exists():
            try:
                model_info["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_dir)
            except:
                model_info["tokenizer"] = None
        else:
            model_info["tokenizer"] = None

        # Registrar modelo
        _MODEL_REGISTRY[name] = model_info
        _METADATA[name] = model_info.get("metadata", None)

    print("Modelos cargados:", list(_MODEL_REGISTRY.keys()))



@app.on_event('startup')
def startup_event():
    safe_load_models2()


def predict_with_pipeline(pipeline, text: str) -> Dict[str, Any]:
    try:
        print(pipeline)
        # Try predict_proba
        if hasattr(pipeline, 'predict_proba'):
            probs = pipeline.predict_proba([text])[0]
        else:
            # Try decision_function
            if hasattr(pipeline, 'decision_function'):
                df = pipeline.decision_function([text])
                df = np.array(df)
                if df.ndim == 1:
                    prob_pos = 1 / (1 + np.exp(-df[0]))
                    probs = np.array([1 - prob_pos, prob_pos])
                else:
                    exps = np.exp(df[0] - np.max(df[0]))
                    probs = exps / exps.sum()
            else:
                pred = pipeline.predict([text])[0]
                classes = getattr(pipeline, 'classes_', None)
                if classes is not None:
                    probs = np.zeros(len(classes))
                    probs[list(classes).index(pred)] = 1.0
                else:
                    probs = np.array([1.0])

        classes = getattr(pipeline, 'classes_', None)
        if classes is None and hasattr(pipeline, 'named_steps'):
            for name, step in pipeline.named_steps.items():
                if hasattr(step, 'classes_'):
                    classes = step.classes_

        if isinstance(probs, np.ndarray) and probs.size > 1 and classes is not None:
            idx = int(np.argmax(probs))
            label = str(classes[idx])
            confidence = float(probs[idx])
            probs_list = probs.tolist()
        else:
            label = str(pipeline.predict([text])[0])
            confidence = 1.0
            probs_list = probs.tolist() if hasattr(probs, 'tolist') else [float(probs)]

        return {'label': label, 'confidence': confidence, 'probs': probs_list}
    except Exception as e:
        return {'error': str(e), 'trace': traceback.format_exc()}


@app.get('/')
def index():
    index_file = STATIC_DIR / 'index.html'
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({'message': 'Index not found. Place a static/index.html file.'}, status_code=404)


@app.get('/models')
def list_models():
    return {'models': list(_MODEL_REGISTRY.keys()), 'metadata': _METADATA}


@app.post('/predict')
def predict(req: PredictRequest):
    model_name = req.model or ('LinearSVC' if 'LinearSVC' in _MODEL_REGISTRY else None)
    if model_name is None:
        raise HTTPException(status_code=400, detail='No model specified and no default available')

    model_info = _MODEL_REGISTRY.get(model_name)
    if model_info is None:
        raise HTTPException(status_code=404, detail=f'Model "{model_name}" not found or failed to loaaaaad')

    model = model_info.get("model")
    tokenizer = model_info.get("tokenizer")
    model_type = model_info.get("type")

    if model is None:
        raise HTTPException(status_code=500, detail=f"Model '{model_name}' failed to loooooad")

    if model_type == "joblib":
        result = predict_with_pipeline(model, req.text)
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result)
        return {'model': model_name, 'text': req.text, 'result': result}

    if model_type == "tensorflow":
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="Tokenizer missing for TensorFlow model")

        inputs = tokenizer(req.text, return_tensors="tf", padding=True, truncation=True)
        probs = model.predict(inputs)
        probs = probs[0]

        label_id = int(np.argmax(probs))
        confidence = float(probs[label_id])

        return {
            'model': model_name,
            'text': req.text,
            'result': {
                'label': label_id,
                'confidence': confidence,
                'probs': probs.tolist()
            }
        }

    raise HTTPException(status_code=500, detail="Unsupported model type")



# Mount static files
if STATIC_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('api:app', host='0.0.0.0', port=8000, reload=True)
