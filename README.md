## Jigsaw Community Rules 
Dashboard para clasificación automática de violaciones usando modelos entrenados.

### Requisitos
- Python 3.10+ (recomendado)
- Pip

### Instalación (Windows PowerShell)
```powershell
# 1. Clonar el repositorio 
git clone https://github.com/Sofiamishel2003/Jigsaw_Agile_Community_Rules_Classification.git
cd Jigsaw_Agile_Community_Rules_Classification

# 2. Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Instalar dependencias mínimas
pip install fastapi uvicorn joblib scikit-learn plotly
```

Si ya tienes un archivo `api_requirements.txt`, puedes sustituir el paso 3 por:
```powershell
pip install -r api_requirements.txt
```

### Ejecutar la UI
Lanzar el servidor FastAPI:
```powershell
uvicorn api:app --reload
```

### Acceso
- Abrir: http://127.0.0.1:8000
