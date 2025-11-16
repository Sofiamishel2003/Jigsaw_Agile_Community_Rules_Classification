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


@app.on_event('startup')
def startup_event():
    safe_load_models()


def predict_with_pipeline(pipeline, text: str) -> Dict[str, Any]:
    try:
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
    model = _MODEL_REGISTRY.get(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f'Model "{model_name}" not found or failed to load')

    result = predict_with_pipeline(model, req.text)
    if 'error' in result:
        raise HTTPException(status_code=500, detail=result)
    return {'model': model_name, 'text': req.text, 'result': result}


# Mount static files
if STATIC_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('api:app', host='0.0.0.0', port=8000, reload=True)
