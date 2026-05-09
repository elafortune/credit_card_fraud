# Credit Card Fraud Detection — ML Pipeline

Système complet de détection de fraude bancaire combinant une pipeline machine learning professionnelle et une interface web interactive.

---

## Pourquoi ce projet ?

La fraude bancaire représente des milliards de dollars de pertes chaque année. Les datasets de transactions réelles sont **extrêmement déséquilibrés** (moins de 0.2 % de fraudes), ce qui rend les modèles classiques inefficaces : un modèle qui prédit « légitime » à 100 % du temps atteint 99.8 % d'accuracy tout en ne détectant aucune fraude.

Ce projet répond à trois problèmes concrets :

| Problème | Solution apportée |
|---|---|
| Déséquilibre des classes | SMOTE + class\_weight sur chaque modèle |
| Choix du modèle optimal | Comparaison RandomForest / LogReg / XGBoost avec hyperparameter search |
| Glissement de données (data drift) | Détection KS-test par feature + réentraînement incrémental |

---

## Architecture

```
credit_card_fraud_project/
├── backend/          # API FastAPI (Python)
│   ├── src/
│   │   ├── eda.py            # Analyse exploratoire + visualisations
│   │   ├── preprocessing.py  # Split stratifié 70 / 15 / 15
│   │   ├── training.py       # Pipeline SMOTE → Scaler → Modèle
│   │   ├── evaluation.py     # Métriques & courbes comparatives
│   │   ├── state.py          # État global thread-safe
│   │   └── api/routes/       # Endpoints EDA, Training, Évaluation, Retraining
│   └── requirements.txt
└── frontend/         # Interface Vite + React + Tailwind CSS v4
    └── src/
        ├── pages/EDAPage.jsx         # Import CSV + 5 visualisations
        ├── pages/TrainingPage.jsx    # Lancement pipeline + résultats live
        ├── pages/EvaluationPage.jsx  # Évaluation sur nouveau dataset
        └── pages/RetrainingPage.jsx  # Drift detection + réentraînement
```

---

## Pipeline ML

### 1. Analyse Exploratoire (EDA)
- Distribution des classes, des montants et du temps
- Corrélation des features PCA avec la variable cible
- Nettoyage automatique (doublons, valeurs manquantes)

### 2. Entraînement (split 70 / 15 / 15)
```
X_train (70%)  → SMOTE + StandardScaler → RandomizedSearchCV (CV 3-fold)
X_test  (15%)  → évaluation après tuning (comparaison inter-modèles)
X_val   (15%)  → sélection finale du meilleur modèle (PR-AUC)
```

Trois modèles comparés :
- **Random Forest** — robuste, peu sensible aux outliers
- **Régression Logistique** — référence linéaire interprétable
- **XGBoost** — boosting par gradient, souvent le plus performant

### 3. Métriques ciblées
Les métriques d'*accuracy* sont trompeuses sur un dataset déséquilibré. Ce projet optimise :
- **Recall** — minimiser les fraudes manquées (faux négatifs)
- **F1-Score** — équilibre précision/recall
- **PR-AUC** — Precision-Recall AUC, métrique de référence pour les classes rares

### 4. Détection de Data Drift
Test de Kolmogorov-Smirnov (KS-test) sur chaque feature : si la distribution d'un nouveau dataset s'éloigne significativement des données d'entraînement (p < 0.05), un réentraînement est recommandé.

---

## Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284 807 transactions, 492 fraudes (0.173 %)
- Features V1–V28 : composantes PCA anonymisées
- Features originales : `Amount`, `Time`
- Cible : `Class` (0 = légitime, 1 = fraude)

---

## Installation & Démarrage

### Prérequis
- Python ≥ 3.13
- Node.js ≥ 18

### Backend
```bash
cd backend
pip install -r requirements.txt
python run.py
# API disponible sur http://localhost:8001
# Documentation Swagger : http://localhost:8001/docs
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Interface disponible sur http://localhost:5173
```

---

## Interface Web

| Page | Fonctionnalité |
|---|---|
| **EDA** | Import du CSV, statistiques descriptives, 5 graphiques automatiques |
| **Entraînement** | Lancement pipeline, barre de progression live, tableau comparatif des modèles |
| **Évaluation** | Tester le modèle sur un nouveau dataset avec les mêmes colonnes |
| **Réapprentissage** | Analyser le drift par feature (KS-test) puis réentraîner avec nouvelles données |

---

## Stack technique

**Backend** — FastAPI · scikit-learn · imbalanced-learn · XGBoost · pandas · matplotlib · scipy  
**Frontend** — Vite · React · Tailwind CSS v4 · Axios · React Router · Lucide Icons
