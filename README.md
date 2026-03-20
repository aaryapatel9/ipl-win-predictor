# 🏏 IPL Win Probability Predictor

A machine learning project that predicts the **live win probability** for IPL teams during the 2nd innings of a match. Built using ball-by-ball delivery data from IPL 2017–2019, the model outputs real-time win/loss odds based on the current match situation.

---

## 📸 Demo

> Enter the batting team, bowling team, venue city, target score, current score, wickets fallen, and overs completed — and get instant win probability for both sides.

---

## 🧠 How It Works

The model is trained on **ball-by-ball data from the 2nd innings** only. For each ball, the following features are engineered and fed into a Logistic Regression pipeline:

| Feature | Description |
|---|---|
| `batting_team` | Team currently batting |
| `bowling_team` | Team currently bowling |
| `city` | Venue city |
| `runs_left` | Runs still needed to win |
| `balls_left` | Balls remaining in the innings |
| `wickets` | Wickets in hand (10 − wickets fallen) |
| `total_runs_x` | Target set by the 1st innings team |
| `crr` | Current Run Rate |
| `rrr` | Required Run Rate |

The target variable is binary: **1 = batting team wins**, **0 = batting team loses**.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Logistic Regression (`solver=liblinear`) |
| Accuracy | **~80.3%** on test set |
| Encoding | OneHotEncoder (`drop='first'`) on team & city columns |

---

## 🗂️ Project Structure

```
ipl-win-predictor/
│
├── matches.csv              # Match-level data (IPL 2017–2019)
├── deliveries.csv           # Ball-by-ball delivery data
├── notebook.ipynb           # Full EDA, feature engineering & model training
├── pipe.pkl                 # Trained sklearn Pipeline (saved model)
├── ipl_predictor.html       # Frontend UI (standalone HTML)
└── README.md
```

---

## ⚙️ Pipeline Architecture

```
Input Features
     │
     ▼
ColumnTransformer
  └── OneHotEncoder  → batting_team, bowling_team, city
  └── passthrough    → runs_left, balls_left, wickets, total_runs_x, crr, rrr
     │
     ▼
LogisticRegression (solver='liblinear')
     │
     ▼
predict_proba() → [P(lose), P(win)]
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ipl-win-predictor.git
cd ipl-win-predictor
```

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib
```

### 3. Run the notebook

Open `notebook.ipynb` in Jupyter and run all cells. This will:
- Load and merge `matches.csv` and `deliveries.csv`
- Engineer all features (runs_left, balls_left, wickets, crr, rrr)
- Train the pipeline and evaluate accuracy
- Save the model as `pipe.pkl`

### 4. Make a prediction

```python
import pickle
import pandas as pd

pipe = pickle.load(open('pipe.pkl', 'rb'))

input_df = pd.DataFrame({
    'batting_team': ['Mumbai Indians'],
    'bowling_team': ['Chennai Super Kings'],
    'city': ['Mumbai'],
    'runs_left': [60],
    'balls_left': [42],
    'wickets': [6],
    'total_runs_x': [180],
    'crr': [9.0],
    'rrr': [8.57]
})

prob = pipe.predict_proba(input_df)
print(f"Win probability: {round(prob[0][1] * 100, 1)}%")
```

### 5. Open the frontend

Simply open `ipl_predictor.html` in any browser — no server required.

---

## 📈 Match Progression Visualization

The notebook includes a `match_progression()` function that plots win/loss probability over each over for any match in the dataset, alongside runs scored and wickets fallen per over.

```python
temp_df, target = match_progression(X_df, match_id=1, pipe=pipe)
```

---

## 🏟️ Teams & Cities Supported

**Teams:** Sunrisers Hyderabad · Mumbai Indians · Royal Challengers Bangalore · Kolkata Knight Riders · Kings XI Punjab · Chennai Super Kings · Rajasthan Royals · Delhi Capitals

**Cities:** Hyderabad · Bangalore · Mumbai · Kolkata · Delhi · Chennai · Chandigarh · Jaipur · Pune · Indore · Nagpur · Dharamsala · Visakhapatnam · Ranchi · Ahmedabad · Cuttack · Abu Dhabi · Sharjah · Mohali · Bengaluru

---

## 📦 Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- pickle

---

## 🙌 Acknowledgements

Dataset sourced from publicly available IPL ball-by-ball data (IPL 2017–2019 seasons).
