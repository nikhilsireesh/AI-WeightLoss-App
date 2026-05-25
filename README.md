# Diet AI — Personalized Nutrition Planning with HCMLP

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CPU-FF6F00?style=flat-square&logo=tensorflow)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat-square&logo=mongodb)

A production-deployed, ML-driven nutrition planning system. The core engine is a **Hierarchical Clustering Multilayer Perceptron (HCMLP)** — a hybrid model that first clusters users into dietary profiles using Agglomerative Clustering, then runs a Keras neural network to predict how likely a person is to adhere to the generated diet plan.

---

## How the Model Works

The ML pipeline inside `ml_model.py` follows these steps every time a user submits their details:

```
User Inputs (age, weight, height, gender, goal, activity)
        │
        ▼
 Mifflin-St Jeor BMR  →  TDEE  →  Calorie & Protein Targets
        │
        ▼
 15-Feature Vector Construction
 (protein target, calorie target, macro splits, activity multiplier,
  gender proxy, BMI proxy, meal frequency proxy, ...)
        │
        ▼
 MinMaxScaler  →  Agglomerative Clustering  →  OneHotEncode Cluster ID
        │
        ▼
 Concatenate scaled features + cluster encoding
        │
        ▼
 HCMLP Keras Model  →  Adherence Probability (0–1)  ×  100  =  Score %
        │
        ▼
 data_driven_food_selector()
 Filter food_database.csv → rank by protein density → fill calories
        │
        ▼
 Per-meal plan with macro targets (3 / 4 / 5 meal splits)
```

**Goal-based calorie adjustment:**
- Weight loss → TDEE − 500 kcal, protein = weight × 2.0 g
- Weight gain → TDEE + 500 kcal, protein = weight × 1.8 g
- Maintain → TDEE, protein = weight × 1.6 g

**Activity multipliers used:** Sedentary 1.2 · Light 1.375 · Moderate 1.55 · Active 1.725

---

## Application Pages

| Route | Page | Access |
|---|---|---|
| `/` | Home — landing page with CTA | Public |
| `/register` | Create account | Public (redirects if logged in) |
| `/login` | Login | Public (redirects if logged in) |
| `/app` | AI Planner Dashboard — form + plan results | Auth required |
| `/log` | Log Food — log meals and view history | Auth required |
| `/about` | Project info and team | Public |

The Navbar renders different links depending on auth state (`isAuth` prop). On logout, the JWT is removed from `localStorage` and the user is redirected to `/`.

---

## Backend API

Base URL (live): `https://weightloss-app-frontend.onrender.com`

All routes marked **Protected** require: `Authorization: Bearer <token>`

### Auth Routes
```
POST  /register     →  Create user (bcrypt hashed password stored in MongoDB)
POST  /login        →  Returns JWT access token (HS256, 24h expiry)
```

### ML Routes
```
POST  /predict          →  [Protected] Run HCMLP pipeline, save & return diet plan
GET   /latest_plan      →  [Protected] Fetch user's most recent saved prediction
```

### Food Logging Routes
```
POST  /log_food     →  [Protected] Save a meal entry (date, meal_type, food_item, calories, protein)
GET   /my_logs      →  [Protected] Return all logs for the current user, sorted newest first
```

### Sample `/predict` Payload
```json
{
  "age": 22,
  "gender": "male",
  "weight": 75.0,
  "height": 175.0,
  "goal": "lose",
  "activity_level": "moderate",
  "meals_per_day": 4,
  "food_preference": "chicken"
}
```

### Sample `/predict` Response
```json
{
  "adherence_score": 78.42,
  "recommended_calories": 2108.0,
  "recommended_protein": 150.0,
  "diet_plan": {
    "breakfast": { "macros": "527 kcal, 38g protein", "foods": ["..."] },
    "lunch":     { "macros": "738 kcal, 60g protein", "foods": ["..."] },
    "dinner":    { "macros": "633 kcal, 53g protein", "foods": ["..."] },
    "snacks":    { "macros": "211 kcal, 0g protein",  "foods": ["..."] }
  }
}
```

---

## Tech Stack

### ML & Data
- **TensorFlow / Keras** — HCMLP model saved as `.keras`
- **Scikit-learn** — `AgglomerativeClustering`, `MinMaxScaler`, `OneHotEncoder`
- **Pandas** — loads `food_database.csv` with `usecols` and `dtype` optimizations
- **NumPy** — feature vector construction
- **Joblib** — deserializes `.pkl` model artifacts at startup

### Backend
- **FastAPI** — async REST API, automatic `/docs` (Swagger UI)
- **Uvicorn** — ASGI server
- **Pydantic** — `UserInput`, `DietResponse`, `UserAuth`, `FoodLogEntry` schemas
- **Motor** — async MongoDB driver; 3 collections: `users`, `predictions`, `food_logs`
- **MongoDB Atlas** — cloud database
- **PyJWT** — HS256 token signing and decoding
- **passlib[bcrypt]** — password hashing and verification
- **python-dotenv** — `MONGO_URI` loaded from `.env`

### Frontend
- **React 18** — component-based UI
- **Vite 5** — dev server (`localhost:5173`) and production bundler
- **React Router DOM v6** — `BrowserRouter`, `Routes`, `Navigate` guards
- **Tailwind CSS v3** — utility-first styling (teal color theme)
- **PostCSS + Autoprefixer** — CSS build pipeline
- **Axios** — installed as dependency; `fetch` API used in components

### Infrastructure
- **Railway** — backend deployment
- **MongoDB Atlas** — database hosting

---

## Deployment Artifacts

The `backend/deployment_artifacts/` folder must be present before starting the server:

| File | Description |
|---|---|
| `hcmlp_model.keras` | Trained Keras neural network |
| `hc_cluster_model.pkl` | Fitted Agglomerative Clustering model |
| `cluster_encoder.pkl` | OneHotEncoder for cluster IDs |
| `minmax_scaler.pkl` | Fitted MinMaxScaler |
| `food_database.csv` | Real food nutrition database (~100 MB) |

---

## Running Locally

### Backend
```bash
cd backend
pip install -r requirements.txt

# .env file
echo "MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/" > .env

uvicorn main:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# App → http://localhost:5173
```

> For local dev, update the API base URL in `Dashboard.jsx` and `LogFood.jsx` from the Railway URL to `http://localhost:8000`.

---

## Database Collections

| Collection | Stores |
|---|---|
| `users` | `username`, hashed `password`, `created_at` |
| `predictions` | `username`, full `input`, full `prediction`, `timestamp` |
| `food_logs` | `username`, `date`, `meal_type`, `food_item`, `calories`, `protein`, `timestamp` |

---

## Python Dependencies

```
fastapi
uvicorn
pydantic
motor
pandas
numpy
scikit-learn
tensorflow-cpu
PyWavelets
scipy
python-dotenv
passlib[bcrypt]
PyJWT
```

---

## License

Academic capstone project. All rights reserved by the project team.
