import os
import shutil
from pathlib import Path
import tensorflow as tf
import json

from transformers import (
    AutoConfig,
    TFAutoModelForSequenceClassification,
    AutoTokenizer
)

BASE_DIR = Path(__file__).resolve().parent

# Rutas de origen
DISTILBERT_SRC = BASE_DIR / "Modelos" / "Modelo_1" / "best_rule_model"
DEBERTA_DIR = BASE_DIR / "Modelos" / "Modelo_3"
DEBERTA_WEIGHTS = DEBERTA_DIR / "best_weights"

EXPORT_DIR = BASE_DIR / "models"
EXPORT_DIR.mkdir(exist_ok=True)


# ========================================================
# FUNCIÓN AUXILIAR PARA CREAR metadata.json
# ========================================================
def write_metadata(path: Path, metadata: dict):
    meta_path = path / "metadata.json"
    with open(meta_path, "w", encoding="utf8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✔ metadata.json creado en {meta_path}")


# ========================================================
# 1) Exportar DistilBERT: modelo + tokenizer + metadata
# ========================================================
def export_distilbert():
    out = EXPORT_DIR / "distilbert"
    print("\n=== EXPORTANDO DISTILBERT ===")

    # Copiar SavedModel
    if not DISTILBERT_SRC.exists():
        raise FileNotFoundError(f"NO se encuentra el SavedModel en {DISTILBERT_SRC}")

    if not out.exists():
        shutil.copytree(DISTILBERT_SRC, out)
        print("✔ DistilBERT exportado correctamente.")
    else:
        print("✔ DistilBERT ya existe, omitiendo copia.")

    # Copiar tokenizer
    tokenizer_out = out / "tokenizer"
    tokenizer_out.mkdir(exist_ok=True)

    print("-> Guardando tokenizer de DistilBERT")
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tok.save_pretrained(tokenizer_out)
    print("✔ Tokenizer DistilBERT guardado.")

    # ---------- METADATA ----------
    metadata = {
        "name": "DistilBERT",
        "type": "tensorflow",
        "path": "distilbert",
        "description": "DistilBERT fine-tuned for toxic rule violation classification",
        "metrics": {
            "accuracy": 0.75,
            "precision": 0.765,
            "recall": 0.75,
            "f1": 0.7818574514038877,
            "confusion_matrix": [
                [124, 76],
                [25, 181]
            ]
        },
        "n_val": 406
    }
    write_metadata(out, metadata)


# ========================================================
# 2) Exportar DeBERTa V3 Small
# ========================================================
def export_deberta():
    out = EXPORT_DIR / "deberta_v3_small"
    print("\n=== EXPORTANDO DEBERTA V3 SMALL ===")

    # Config
    config_path = DEBERTA_DIR / "config.json"
    if not config_path.exists():
        print("⚠ No existe config.json — creando...")
        config = AutoConfig.from_pretrained("microsoft/deberta-v3-small")
        config.num_labels = 2
        with open(config_path, "w", encoding="utf8") as f:
            json.dump(json.loads(config.to_json_string()), f, indent=2)
        print("✔ config.json creado.")
    else:
        print("✔ config.json encontrado.")

    if out.exists():
        print("✔ deberta_v3_small ya existe, omitiendo creación.")
        return

    # Cargar configuración
    print("-> Cargando configuración...")
    config = AutoConfig.from_pretrained(str(config_path))

    # *** IMPORTANTE ***
    # Crear modelo SOLO desde config, sin PyTorch
    print("-> Construyendo arquitectura DeBERTa en TensorFlow...")
    model = TFAutoModelForSequenceClassification.from_config(config)

    # Cargar pesos de tu entrenamiento
    print("-> Cargando pesos entrenados...")
    model.load_weights(str(DEBERTA_WEIGHTS))

    # Guardar SavedModel
    print("-> Guardando modelo en SavedModel:", out)
    model.save(out)

    # Guardar tokenizer
    tokenizer_out = out / "tokenizer"
    tokenizer_out.mkdir(exist_ok=True)

    print("-> Guardando tokenizer DeBERTa...")
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    tok.save_pretrained(tokenizer_out)

    print("✔ DeBERTa exportada correctamente.")

    # metadata
    metadata = {
        "name": "DeBERTa_v3_small",
        "type": "tensorflow",
        "path": "deberta_v3_small",
        "description": "DeBERTa v3 Small fine-tuned for toxic rule violation classification",
        "metrics": {
            "f1": 0.78,
            "accuracy": 0.77,
            "precision": 0.77,
            "recall": 0.78,
            "confusion_matrix": [
            [154, 46],
            [46, 160]
            ],
            "n_val": 406
        }
    }
    write_metadata(out, metadata)

# ========================================================
# MAIN
# ========================================================
if __name__ == "__main__":
    print("\n=== INICIANDO EXPORTACIÓN DE MODELOS ===")
    export_distilbert()
    export_deberta()
    print("\n=== TODO LISTO ===")
