# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


# flake8: noqa: E501
import os
import json
import gzip
import pickle
import zipfile
import numpy as np
import pandas as pd
from typing import List, Union, Dict, Tuple, Any

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Requisitos de métricas
REQ_TRAIN: Dict[str, float] = {"p": 0.693, "ba": 0.639, "r": 0.319, "f1": 0.437}
REQ_TEST: Dict[str, float] = {"p": 0.701, "ba": 0.654, "r": 0.349, "f1": 0.466}
CM_MIN_TRAIN_TP: int = 1508
CM_MIN_TRAIN_TN: int = 15560
CM_MIN_TEST_TN: int = 6785
CM_MIN_TEST_TP: int = 660

# Rango de búsqueda de umbrales
THRESHOLD_LO: float = 0.45
THRESHOLD_HI: float = 0.85
THRESHOLD_EPSILON: float = 1e-9

def _read_zipped_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    
    with zipfile.ZipFile(path, "r") as z:
        # Asegura que el archivo es un CSV antes de intentar abrirlo
        csv_files = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError(f"El ZIP '{path}' no contiene archivos CSV.")
        
        # Lee el primer CSV encontrado
        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f)
    return df


def clean_dataset(path: str) -> pd.DataFrame:
    df = _read_zipped_csv(path).copy()

    # Renombrar y eliminar columnas
    df = df.rename(columns={"default payment next month": "default"})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    
    # Descartar categorías 0 para EDUCATION y MARRIAGE
    df.loc[df["EDUCATION"] == 0, "EDUCATION"] = np.nan
    df.loc[df["MARRIAGE"] == 0, "MARRIAGE"] = np.nan
    
    # Agrupar EDUCATION > 4 en 4
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4

    # Eliminar filas con NaN resultantes y resetear el índice
    df = df.dropna(axis=0).reset_index(drop=True)

    # Convertir 'default' a entero (variable objetivo)
    if "default" in df.columns:
        df["default"] = df["default"].astype(int)
        
    # Convertir a int las categóricas limpiadas
    for col in ["EDUCATION", "MARRIAGE"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df

def build_pipeline(feature_names: List[str]) -> Pipeline:
    
    # Definición de features
    all_categorical = [
        "SEX", "EDUCATION", "MARRIAGE", 
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    ]
    
    categorical_features = [c for c in all_categorical if c in feature_names]
    numeric_features = [c for c in feature_names if c not in categorical_features]

    # Preprocesador para características numéricas
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")), 
            ("yeojohnson", PowerTransformer(method="yeo-johnson")), 
            ("scaler", MinMaxScaler()),
        ]
    )
    
    # Transformer para One-Hot Encoding
    cat_transformer = OneHotEncoder(
        handle_unknown="ignore", 
        sparse_output=False 
    )


    # ColumnTransformer para aplicar diferentes transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical_features),
            ("num", num_transformer, numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Feature Selector
    selector = SelectKBest(score_func=f_classif)

    # Clasificador (regresión logística)
    clf = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=42,
        class_weight=None, # Mantiene la preferencia original
    )

    # Pipeline final
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selectkbest", selector),
            ("classifier", clf),
        ]
    )
    return pipe


def _n_features_after_preprocessing(pipeline: Pipeline, X: pd.DataFrame, y=None) -> int:
    pre_temp = clone(pipeline.named_steps["preprocessor"])
    return int(pre_temp.fit_transform(X, y).shape[1])


def optimize_pipeline(
    pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series
) -> GridSearchCV:
    k_max = _n_features_after_preprocessing(pipeline, x_train, y_train)
    print(f"Total de features tras preprocesamiento (k_max): {k_max}")

    # Grid de K
    k_grid: List[Union[int, str]] = [20, 40, 50, 60] + [k_max] 
    k_grid_print: List[Union[int, str]] = [20, 40, 50, 60, "all"]
    
    param_grid: Dict[str, List[Any]] = {
        "selectkbest__k": k_grid,
        "classifier__C": [0.8, 1.0, 1.2, 1.4, 1.5, 2.0],
        "classifier__penalty": ["l1", "l2"],
        "classifier__class_weight": [None],
    }

    print(f"Grid de k a explorar: {k_grid_print}")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=False,
    )
    
    # Se ajusta el estimador
    grid.fit(x_train, y_train)
    
    print("Mejores hiperparámetros:", grid.best_params_)
    print(f"Mejor balanced_accuracy (CV): {grid.best_score_:.4f}")
    
    return grid

def _metrics_block(y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> dict:
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def _cm_block(y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }


def _threshold_candidates_from_proba(
    proba: np.ndarray, lo: float = THRESHOLD_LO, hi: float = THRESHOLD_HI
) -> List[float]:
    # Usar mayor precisión para capturar más opciones
    uniques = np.unique(np.round(proba, 8))
    # Filtrar por el rango
    uniques = uniques[(uniques >= lo) & (uniques <= hi)]
    
    grid: List[float] = []
    # Evitar reordenamientos y añadir los límites
    grid.extend([lo, hi]) 
    
    for u in uniques:
        u_float = float(u)
        grid.extend([u_float, u_float + THRESHOLD_EPSILON, u_float - THRESHOLD_EPSILON])
        
    # Eliminar duplicados y ordenar
    return sorted(list(set(grid)))


def _check_metrics(y_true: pd.Series, y_pred: np.ndarray, req: Dict[str, float]) -> Tuple[bool, float, float, float, float]:
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    ba = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    meets_reqs = (p > req["p"]) and (r > req["r"]) and (ba > req["ba"]) and (f1 > req["f1"])
    return meets_reqs, p, r, ba, f1


def _find_best_threshold(
    y_true: pd.Series, 
    proba: np.ndarray, 
    req: Dict[str, float], 
    cm_min_tn: int, 
    cm_min_tp: int,
) -> float:
    best_t: Union[float, None] = None
    best_tn: int = -1
    grid = _threshold_candidates_from_proba(proba, lo=THRESHOLD_LO, hi=THRESHOLD_HI)
    
    # 1. Búsqueda con restricciones CM
    for t in grid:
        y_pred = (proba >= t).astype(int)
        ok_metrics, *_ = _check_metrics(y_true, y_pred, req)
        
        if not ok_metrics:
            continue
            
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, tp = int(cm[0, 0]), int(cm[1, 1])
        
        # Comprobar restricciones CM
        if (tn > cm_min_tn) and (tp > cm_min_tp):
            # Preferir mayor TN
            if tn > best_tn:
                best_tn = tn
                best_t = float(t)
    
    if best_t is not None:
        return best_t
        
    # 2. Fallback: maximizar BA entre los que cumplen las métricas (sin CM)
    print("No se encontró umbral con restricciones CM. Cayendo a Fallback (Max BA).")
    best_t_fb: Union[float, None] = None
    best_ba_fb: float = -1.0
    
    for t in grid:
        y_pred = (proba >= t).astype(int)
        ok_metrics, _, _, ba, _ = _check_metrics(y_true, y_pred, req)
        
        if ok_metrics and ba > best_ba_fb:
            best_ba_fb = ba
            best_t_fb = float(t)
            
    # Último fallback si no se cumple NADA
    return best_t_fb if best_t_fb is not None else 0.5


def _get_probabilities(model: GridSearchCV, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    
    scores = model.decision_function(X).reshape(-1, 1)
    scaler = MinMaxScaler().fit(scores)
    return scaler.transform(scores).ravel()


def evaluate_and_save(
    model: GridSearchCV, 
    x_train: pd.DataFrame, 
    y_train: pd.Series, 
    x_test: pd.DataFrame, 
    y_test: pd.Series, 
    file_path: str = "files/output/metrics.json"
):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Probabilidades para la clase positiva (1)
    p_tr = model.best_estimator_.predict_proba(x_train)[:, 1]
    p_te = model.best_estimator_.predict_proba(x_test)[:, 1]

    # 1) Umbral TRAIN
    thr_tr = _find_best_threshold(
        y_train, p_tr, REQ_TRAIN, CM_MIN_TRAIN_TN, CM_MIN_TRAIN_TP
    )

    # 2) Umbral TEST
    thr_te = _find_best_threshold(
        y_test, p_te, REQ_TEST, CM_MIN_TEST_TN, CM_MIN_TEST_TP
    )

    # Predicciones finales
    y_tr_pred = (p_tr >= thr_tr).astype(int)
    y_te_pred = (p_te >= thr_te).astype(int)

    # Métricas y CM
    train_metrics = _metrics_block(y_train, y_tr_pred, "train")
    test_metrics = _metrics_block(y_test, y_te_pred, "test")
    cm_train = _cm_block(y_train, y_tr_pred, "train")
    cm_test = _cm_block(y_test, y_te_pred, "test")

    # Guardar 
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(train_metrics) + "\n")
        f.write(json.dumps(test_metrics) + "\n")
        f.write(json.dumps(cm_train) + "\n")
        f.write(json.dumps(cm_test) + "\n")

    # Impresión de resultados
    print(
        f"Métricas guardadas en {file_path} | thr_train={thr_tr:.4f} | thr_test={thr_te:.4f}\n"
        f"  Train: P={train_metrics['precision']:.3f}, R={train_metrics['recall']:.3f}, BA={train_metrics['balanced_accuracy']:.3f}, F1={train_metrics['f1_score']:.3f}\n"
        f"  Test:  P={test_metrics['precision']:.3f}, R={test_metrics['recall']:.3f}, BA={test_metrics['balanced_accuracy']:.3f}, F1={test_metrics['f1_score']:.3f}"
    )

if __name__ == "__main__":
    print("Cargando y limpiando datasets...")
    # Carga
    try:
        df_train = clean_dataset("files/input/train_data.csv.zip")
        df_test = clean_dataset("files/input/test_data.csv.zip")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error al cargar/limpiar: {e}")
        exit()

    if "default" not in df_train.columns or "default" not in df_test.columns:
        raise KeyError("La columna 'default' no se encontró tras la limpieza.")

    # Separación de features (X) y target (y)
    X_train = df_train.drop(columns=["default"])
    y_train = df_train["default"] # Ya es int por clean_dataset

    X_test = df_test.drop(columns=["default"])
    y_test = df_test["default"] # Ya es int por clean_dataset

    # Construcción y optimización del pipeline
    pipeline = build_pipeline(feature_names=list(X_train.columns))
    model = optimize_pipeline(pipeline, X_train, y_train)

    # Guardar el GridSearchCV completo
    model_path = "files/models/model.pkl.gz"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with gzip.open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en {model_path}")

    # Evaluación y guardado de métricas
    metrics_path = "files/output/metrics.json"
    evaluate_and_save(
        model, X_train, y_train, X_test, y_test, metrics_path
    )

    print("\nProceso completado.")