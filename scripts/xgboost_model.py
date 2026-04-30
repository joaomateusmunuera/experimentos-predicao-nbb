from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from experimentos import read_dados

def get_hyper_params_xgboost(X_train, y_train):
    # Definir os hiperparâmetros para o Grid Search
    param_grid = {
        'n_estimators': [50, 100, 150],  # Número de árvores
        'max_depth': [3, 5, 7],         # Profundidade máxima da árvore
        'learning_rate': [0.01, 0.1, 0.2],  # Taxa de aprendizado
        'subsample': [0.8, 1.0],        # Fração de amostras para cada árvore
        'colsample_bytree': [0.8, 1.0], # Fração de recursos para cada árvore
    }

    # Criar o modelo XGBoost
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

    # Configurar o Grid Search com validação cruzada
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=2, scoring='f1_weighted', verbose=2, n_jobs=-1)

    try:
        # Executar o Grid Search no conjunto de treino
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
    except ValueError as e:
        print(f"Erro durante o Grid Search: {e}")
        return None

    return best_params

def run_model_xgboost(treino_path, teste_path, useGridSearch=True):
    X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)

    if useGridSearch:
        # Obter os melhores hiperparâmetros usando Grid Search
        best_params = get_hyper_params_xgboost(X_train, y_train)

        if(best_params is None):
            return None, None, None

        # Criar e treinar o modelo com os melhores hiperparâmetros
        model = XGBClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            eval_metric='logloss',
            random_state=42
        )
    else:
        # Modelo com hiperparâmetros padrão
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric='logloss',
            random_state=42
        )
        best_params = []

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer previsões com o conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Para lidar com desbalanceamento de classes

    return accuracy, f1, best_params
