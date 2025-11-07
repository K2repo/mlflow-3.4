# experiments/train_models.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from utils.data_loader import load_campaign_data

def train_open_rate_model():
    # Cargar datos
    df = load_campaign_data()
    
    # Features y target
    features = ['hour_sin', 'hour_cos', 'day_of_week', 'audience_size']
    X = df[features]
    y = df['open_rate']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Configurar experimento
    mlflow.set_experiment("whatsapp_campaign_optimization")
    
    with mlflow.start_run():
        # Entrenar modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluar
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Loggear parámetros y métricas
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metrics({"mae": mae, "r2_score": r2})
        
        # Loggear el modelo
        mlflow.sklearn.log_model(model, "open_rate_model")
        
        print(f"✅ Modelo entrenado - MAE: {mae:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    train_open_rate_model()