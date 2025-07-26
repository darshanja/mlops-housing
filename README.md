# 🏠 California Housing MLOps Pipeline

End-to-end MLOps project for predicting house prices using:

- ✅ Git + DVC for versioning data & code
- 📊 MLflow for experiment tracking
- 🧠 Scikit-learn for training models
- 🌐 FastAPI + Docker for serving
- 🔁 GitHub Actions for CI/CD

## 🔧 Local Run

```bash
python src/train.py
uvicorn src.app:app --reload
```

## 🐳 Docker

```bash
docker build -t housing-api .
docker run -p 8000:8000 housing-api
```

## 🔁 DVC Pipeline

```bash
dvc repro
```

## 🧪 API Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc":8.3,"HouseAge":20,"AveRooms":6,"AveBedrms":1,"Population":500,"AveOccup":2.5,"Latitude":37,"Longitude":-122}'
```
