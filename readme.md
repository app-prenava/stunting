# Stunting Risk Prediction Pipeline

Machine learning pipeline and API for early stunting risk prediction using IFLS5 dataset.

## Local Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

## Running the API Locally

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can test the endpoints interactively via the Swagger UI at `http://127.0.0.1:8000/docs`.

## Running with Docker

```bash
docker build -t stunting-api .
docker run -p 2727:8000 stunting-api
```

## API Endpoints

- `GET /` - Check API status.
- `GET /health` - Health check (validates that model artifacts are loaded).
- `GET /features` - Retrieve the list of expected input features for the model.
- `POST /predict` - Make a prediction.

### Example POST /predict payload

```json
{
  "child_gender": 1.0,
  "mother_education_level": 2.0,
  "mother_employment_status": 1.0,
  "mother_age_at_birth": 28.5,
  "is_teenage_mother": 0.0,
  "is_high_risk_mother_age": 0.0,
  "has_delivery_insurance": 1.0,
  "anc_clinic_midwife": 1.0,
  "anc_hospital": 0.0,
  "anc_traditional_other": 0.0,
  "anc_unknown": 0.0
}
```
