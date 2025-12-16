import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuración
st.set_page_config(
    page_title="Predicción de la Rotación de Clientes",
    layout="centered"
)

st.title("Predicción de la Rotación de Clientes")

# ============================================
# 14 CARACTERÍSTICAS 
# ============================================
# Asegúrate que estas son EXACTAMENTE las que usó el scaler_14.pkl
FEATURES_14 = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges'
]

# ============================================
# 2. FUNCIÓN MEJORADA PARA CARGAR MODELOS
# ============================================
@st.cache_resource
def load_models():
    try:
        # Cargar modelos
        rf_14 = joblib.load("bagging_random_forest.joblib")
        cat_14 = joblib.load("boosting_catboost.joblib")
        voting_14 = joblib.load("voting_rf_svm.joblib")
        
        rf_full = joblib.load("Complet_bagging_random_forest.joblib")
        cat_full = joblib.load("Complet_boosting_catboost.joblib")
        voting_full = joblib.load("Complet_voting_rf_svm.joblib")
        
        # Cargar escaladores con verificación
        scaler_14 = joblib.load("scaler_14.pkl")
        scaler_full = joblib.load("scaler.pkl")
        
        # Verificar que el escalador_14 tiene 14 características
        if hasattr(scaler_14, 'n_features_in_'):
            if scaler_14.n_features_in_ != 14:
                st.warning(f"scaler_14.pkl esperaba 14 características, pero tiene {scaler_14.n_features_in_}")
        
        if hasattr(scaler_full, 'n_features_in_'):
            st.info(f"scaler.pkl  con {scaler_full.n_features_in_} características")
        
        return (
            rf_14, cat_14, voting_14,
            rf_full, cat_full, voting_full,
            scaler_14, scaler_full
        )
        
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        # Crear escaladores por defecto si no existen
        scaler_14 = StandardScaler()
        scaler_full = StandardScaler()
        return None, None, None, None, None, None, scaler_14, scaler_full

# ============================================
# CARGAR MODELOS
# ============================================
try:
    (
        rf_14, cat_14, voting_14,
        rf_full, cat_full, voting_full,
        scaler_14, scaler_full
    ) = load_models()
except Exception as e:
    st.error(f"Error al cargar modelos: {e}")
    st.stop()

# ============================================
# CARGAR DATOS
# ============================================
@st.cache_data
def load_csv():
    df = pd.read_csv("dato.csv")
    return df

try:
    df_raw = load_csv()
    have_csv = True
except Exception as e:
    st.error(f"No se encontró el archivo 'dato.csv': {e}")
    have_csv = False
    df_raw = None

# ============================================
# DEFINIR CARACTERÍSTICAS COMPLETAS
# ============================================
if have_csv:
    # Verificar que todas las FEATURES_14 están en los datos
    missing_features = [f for f in FEATURES_14 if f not in df_raw.columns]
    if missing_features:
        st.warning(f"Faltan características en los datos: {missing_features}")
    
    # Definir características completas (todas excepto Churn)
    FULL_FEATURES = [c for c in df_raw.columns if c != "Churn"]
    
    # Crear baseline (valores medios)
    baseline_full = df_raw[FULL_FEATURES].mean(numeric_only=True)
    baseline_full = baseline_full.reindex(FULL_FEATURES).fillna(0)
else:
    FULL_FEATURES = FEATURES_14
    baseline_full = pd.Series({c: 0 for c in FULL_FEATURES})

# ============================================
# FUNCIONES DE PREPROCESAMIENTO 
# ============================================
def preprocess_14_features(user_data):
    # Crear DataFrame con las 14 características exactas
    X_new = pd.DataFrame([user_data])[FEATURES_14]
    
    # Verificar que tenemos todas las características
    if X_new.shape[1] != 14:
        st.error(f"Error: Se esperaban 14 características, pero se tienen {X_new.shape[1]}")
        return None
    
    # Aplicar escalado
    try:
        X_scaled = scaler_14.transform(X_new)
        return X_scaled
    except Exception as e:
        st.error(f"Error en escalado (14 características): {e}")
        # Mostrar información de depuración
        if hasattr(scaler_14, 'feature_names_in_'):
            st.info(f"El escalador espera: {scaler_14.feature_names_in_}")
        st.info(f"Tus datos tienen: {X_new.columns.tolist()}")
        return None

def preprocess_full_features(user_data):
     # Crear fila con todas las características
    row_full = baseline_full.copy()
    
    # Actualizar con valores del usuario
    for feature, value in user_data.items():
        if feature in row_full.index:
            row_full[feature] = value
    
    # Crear DataFrame
    X_new = pd.DataFrame([row_full], columns=FULL_FEATURES)
    
    # Aplicar escalado si el escalador fue entrenado
    try:
        if hasattr(scaler_full, 'n_features_in_'):
            X_scaled = scaler_full.transform(X_new)
            return X_scaled
        else:
            return X_new.values
    except Exception as e:
        st.error(f"Error en escalado (características completas): {e}")
        return X_new.values

# FUNCIONES DE MÉTRICAS
def compute_metrics_14(model):
    if not have_csv or "Churn" not in df_raw.columns:
        return None
    
    try:
        # Seleccionar solo las 14 características
        X_eval = df_raw[FEATURES_14].copy()
        y_true = df_raw["Churn"].values
        
        # Escalar
        X_scaled = scaler_14.transform(X_eval)
        
        # Predecir
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_proba)
        }
    except Exception as e:
        st.error(f"Error calculando métricas (14 características): {e}")
        return None

def compute_metrics_full(model):
    """Calcular métricas para modelos completos"""
    if not have_csv or "Churn" not in df_raw.columns:
        return None
    
    try:
        X_eval = df_raw[FULL_FEATURES].copy()
        y_true = df_raw["Churn"].values
        
        # Escalar si es necesario
        if hasattr(scaler_full, 'n_features_in_'):
            X_scaled = scaler_full.transform(X_eval)
        else:
            X_scaled = X_eval.values
        
        # Predecir
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_proba)
        }
    except Exception as e:
        st.error(f"Error calculando métricas (características completas): {e}")
        return None

#  INTERFAZ DE USUARIO - BARRA LATERAL

st.sidebar.header("Configuración del modelo")

# Selección de grupo de modelos
grupo = st.sidebar.radio(
    "SELECCIONAR GRUPO DE MODELOS:",
    ["14 Características (Seleccionadas)", "Todas las Características"]
)

# Inicializar variables
current_model = None
current_is_full = False
modelo_nombre = ""
use_scaler = False

if grupo == "14 Características (Seleccionadas)":
    modelo_nombre = st.sidebar.selectbox(
        "Seleccionar Modelo:",
        ["Bagging - Random Forest", "Boosting - CatBoost", "Voting - RF + SVM (Soft)"]
    )
    
    if modelo_nombre == "Bagging - Random Forest":
        current_model = rf_14
        use_scaler = True  # RF de 14 características usa escalado
    elif modelo_nombre == "Boosting - CatBoost":
        current_model = cat_14
        use_scaler = True  # CatBoost de 14 características usa escalado
    else:
        current_model = voting_14
        use_scaler = True  # Voting de 14 características usa escalado
    
    current_is_full = False
    
else:  # Todas las Características
    modelo_nombre = st.sidebar.selectbox(
        "Seleccionar Modelo:",
        ["Bagging - Random Forest", "Boosting - CatBoost", "Voting - RF + SVM (Soft)"]
    )
    
    if modelo_nombre == "Bagging - Random Forest":
        current_model = rf_full
    elif modelo_nombre == "Boosting - CatBoost":
        current_model = cat_full
    else:
        current_model = voting_full
    
    current_is_full = True
    use_scaler = True  # Asumimos que modelos completos también usan escalado

# Mostrar información del modelo seleccionado


#  PESTAÑAS PRINCIPALES

tab_pred, tab_pca, tab_info = st.tabs([" Predicción", " Análisis PCA", "ℹ Información"])

# PREDICCIÓN

with tab_pred:
    st.subheader("Ingresar datos del cliente")
    
    # Verificar que tenemos un modelo
    if current_model is None:
        st.error("No se pudo cargar el modelo seleccionado")
        st.stop()
    
    # Crear columnas para inputs
    col1, col2 = st.columns(2)
    
    user_data = {}
    
    with col1:
        user_data['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1], 0)
        user_data['Partner'] = st.selectbox("Socio", [0, 1], 0)
        user_data['Dependents'] = st.selectbox("Dependents", [0, 1], 0)
        user_data['tenure'] = st.number_input("Meses en la empresa", 0, 100, 12)
        user_data['MultipleLines'] = st.selectbox("Varias líneas", [0, 1], 0)
        user_data['InternetService'] = st.selectbox(
            "Internet Service", 
            options=[0, 1, 2], 
            format_func=lambda x: ["DSL", "Fiber optic", "No"][x],
            index=1
        )
    
    with col2:
        user_data['OnlineSecurity'] = st.selectbox("Online Security", [0, 1], 0)
        user_data['OnlineBackup'] = st.selectbox("Online Backup", [0, 1], 0)
        user_data['DeviceProtection'] = st.selectbox("Device Protection", [0, 1], 0)
        user_data['TechSupport'] = st.selectbox("Tech Support", [0, 1], 0)
        user_data['Contract'] = st.selectbox(
            "Contract",
            options=[0, 1, 2],
            format_func=lambda x: ["Month-to-month", "One year", "Two years"][x],
            index=0
        )
        user_data['PaperlessBilling'] = st.selectbox("Paperless Billing", [0, 1], 1)
        user_data['PaymentMethod'] = st.selectbox(
            "Payment Method",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Electronic check", "Mailed check", "Bank transfer", "Credit card"][x],
            index=0
        )
        user_data['MonthlyCharges'] = st.number_input("Monthly Charges", 0.0, 200.0, 70.0, 0.1)
    
    # Botón de predicción
    if st.button(" Predecir Probabilidad de Salida", type="primary"):
        st.markdown("---")
        
        # Preprocesar datos según el tipo de modelo
        if not current_is_full:
            # Modelo de 14 características
            X_processed = preprocess_14_features(user_data)
        else:
            # Modelo completo
            X_processed = preprocess_full_features(user_data)
        
        if X_processed is not None:
            try:
                # Realizar predicción
                proba = current_model.predict_proba(X_processed)[0, 1]
                pred = current_model.predict(X_processed)[0]
                
                # Mostrar resultados
                st.subheader(" Resultados")
                
                # Barra de progreso para visualizar probabilidad
                st.progress(float(proba))
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric(
                        label="Probabilidad ", 
                        value=f"{proba:.1%}",
                        delta=f"Alta probabilidad" if proba > 0.7 else "Baja probabilidad" if proba < 0.3 else "Probabilidad media"
                    )
                
                with col_res2:
                    st.metric(
                        label="Predicción", 
                        value=" SALIDA PROBABLE" if pred == 1 else "PERMANECE",
                        delta="Revisión recomendada" if pred == 1 else "Cliente estable"
                    )
                
                # Interpretación
                st.markdown("---")
                st.subheader("Interpretación")
                if proba < 0.3:
                    st.success("**BAJO RIESGO:**Recomendado para retención estándar.")
                elif proba < 0.7:
                    st.warning("**RIESGO MODERADO:** Recomendado para programas de retención proactiva.")
                else:
                    st.error("**ALTO RIESGO:** Recomendado para intervención inmediata del equipo de retención.")
                
                # Calcular métricas globales
                st.markdown("---")
                st.subheader("Métricas del Modelo en el Dataset Completo")
                
                if not current_is_full:
                    metrics = compute_metrics_14(current_model)
                else:
                    metrics = compute_metrics_full(current_model)
                
                if metrics:
                    col_met1, col_met2 = st.columns(2)
                    with col_met1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                    with col_met2:
                        st.metric("AUC-ROC", f"{metrics['auc']:.2%}")
                    
                    # Interpretación de métricas
                    st.caption(f"*El modelo tiene un accuracy del {metrics['accuracy']:.1%} y un AUC-ROC de {metrics['auc']:.3f} *")
                
            except Exception as e:
                st.error(f"Error en la predicción: {e}")
        else:
            st.error("No se pudieron procesar los datos. Verifica los valores ingresados.")

# ============================================
# ANÁLISIS PCA
# ============================================
with tab_pca:
    if not have_csv:
        st.error("No se cargaron datos para análisis PCA")
    else:
        st.subheader("Análisis de Componentes Principales (PCA)")
        
        # Verificar que tenemos las 14 características
        missing_pca_features = [f for f in FEATURES_14 if f not in df_raw.columns]
        if missing_pca_features:
            st.error(f"Faltan características para PCA: {missing_pca_features}")
        else:
            # Preparar datos
            df_pca = df_raw[FEATURES_14].copy()
            
            # Escalar
            scaler_pca = StandardScaler()
            X_scaled = scaler_pca.fit_transform(df_pca)
            
            # Aplicar PCA
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)
            
            explained = pca.explained_variance_ratio_
            cum_explained = np.cumsum(explained)
            
            
            # Gráfico 2: Variables más importantes
            st.markdown("### Variables más importantes (Componente 1)")
            
            pc1 = pca.components_[0]
            weights = np.abs(pc1)
            idx = np.argsort(weights)[::-1]
            
            top_features = [FEATURES_14[i] for i in idx]
            top_values = weights[idx]
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.barh(top_features[::-1], top_values[::-1], color='coral')
            ax2.set_xlabel("Importancia en Componente Principal 1")
            ax2.set_title("Variables ordenadas por importancia en PCA")
            ax2.grid(True, alpha=0.3, axis='x')
            
            st.pyplot(fig2)


#  INFORMACIÓN

with tab_info:
    st.subheader("ℹInformación del Sistema")
    
    st.markdown("### Características utilizadas")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("**14 Características Seleccionadas:**")
        for i, feature in enumerate(FEATURES_14[:7], 1):
            st.write(f"{i}. {feature}")
    
    with col_info2:
        st.markdown("")
        for i, feature in enumerate(FEATURES_14[7:], 8):
            st.write(f"{i}. {feature}")
    
    st.markdown("---")
    
    st.markdown("### Modelos Disponibles")
    
    models_info = {
        "Random Forest": "Ensemble de árboles de decisión con bagging",
        "CatBoost": "Algoritmo de boosting optimizado para datos categóricos",
        "Voting Classifier": "Ensemble que combina Random Forest y SVM"
    }
    
    for model, desc in models_info.items():
        with st.expander(f"**{model}**"):
            st.write(desc)
    
    st.markdown("---")
    
    st.markdown("### Solución de Problemas")
    
    st.markdown("""
    ERROR
    """)

# ============================================
# 10. FOOTER
# ============================================
st.markdown("---")
st.caption("Sistema de Predicción de Rotación de Clientes ")