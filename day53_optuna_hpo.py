import streamlit as st
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("🎯 Day 53: Optuna-Based Hyperparameter Optimization")
st.write("""
This project demonstrates **automated hyperparameter tuning** using **Optuna**, 
which applies Bayesian optimization to find the best parameters for a model efficiently.
""")

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
st.write("### Dataset Preview")
st.write(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    return cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()

# Run optimization
if st.button("🚀 Run Hyperparameter Optimization"):
    with st.spinner("Optimizing... please wait ⏳"):
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
        st.success("Optimization Complete ✅")
        st.write("### Best Parameters Found:")
        st.json(best_params)

        # Train model with best params
        best_model = RandomForestClassifier(**best_params, random_state=42)
        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.metric("🎯 Accuracy on Test Set", f"{acc*100:.2f}%")

        st.write("### Optuna Optimization History")
        optuna.visualization.matplotlib.plot_optimization_history(study)
        st.pyplot()

st.write("💡 Optuna helps automate the model tuning process using intelligent search rather than manual trial and error!")
