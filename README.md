## Jigsaw Community Rules 
Dashboard para clasificación automática de violaciones usando modelos entrenados.
<img width="2559" height="1463" alt="image" src="https://github.com/user-attachments/assets/999771c0-e6d9-41d8-b0eb-afa78e2b96ab" />
<img width="2557" height="1465" alt="image" src="https://github.com/user-attachments/assets/78e6c8be-0363-4f4a-84f0-63aaf519e438" />
<img width="2548" height="1473" alt="image" src="https://github.com/user-attachments/assets/319828ad-9531-435b-b2d0-16bde9e22420" />
<img width="2559" height="1462" alt="image" src="https://github.com/user-attachments/assets/50864716-254d-41a2-a909-2cb9cf2fe670" />


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
