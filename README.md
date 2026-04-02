# Workflow-CI

Repository CI/CD Pipeline untuk model Machine Learning Wine Quality menggunakan MLflow Project dan GitHub Actions.

## Struktur Repository
```
Workflow-CI/
├── .github/workflows/ci.yml
├── MLProject/
│   ├── modelling.py
│   ├── MLProject
│   ├── conda.yaml
│   ├── wine_quality_preprocessing.csv
│   └── dockerhub_link.txt
└── README.md
```

## Fitur CI Pipeline
1. **Train Model** - Melatih RandomForestClassifier via MLflow Project
2. **Upload Artifacts** - Menyimpan artefak ke GitHub (Skilled)
3. **Docker Build & Push** - Build Docker image dan push ke Docker Hub (Advanced)

## Docker Hub
Image: `sonyalfauzan/wine-quality-model`
Link: https://hub.docker.com/r/sonyalfauzan/wine-quality-model

## Menjalankan Lokal
```bash
cd MLProject
python modelling.py --n_estimators 200 --max_depth 15
```

## GitHub Secrets yang Diperlukan
- `DOCKERHUB_USERNAME` - Username Docker Hub
- `DOCKERHUB_TOKEN` - Access token Docker Hub

## Author
Sony Alfauzan
