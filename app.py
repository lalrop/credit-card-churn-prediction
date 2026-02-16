import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Cargar artefactos del modelo ---
ruta = os.path.dirname(os.path.abspath(__file__))
modelo = joblib.load(os.path.join(ruta, 'modelo_churn.pkl'))
scaler = joblib.load(os.path.join(ruta, 'scaler_churn.pkl'))
feature_names = joblib.load(os.path.join(ruta, 'feature_names.pkl'))
modelo_nombre = joblib.load(os.path.join(ruta, 'modelo_nombre.pkl'))

# --- Configuración de página ---
st.set_page_config(
    page_title="Predictor de Churn - Credit Card",
    page_icon="💳",
    layout="wide"
)

st.title("Predictor de Riesgo de Churn")
st.caption(f"Modelo: {modelo_nombre}")

st.divider()

# =============================================================
# SIDEBAR - Inputs del cliente
# =============================================================
st.sidebar.header("Datos del Cliente")

# -- Demográficas --
st.sidebar.subheader("Demográficas")
customer_age = st.sidebar.slider("Edad", 18, 80, 45)
gender = st.sidebar.selectbox("Género", ["M", "F"])
dependent_count = st.sidebar.slider("Dependientes", 0, 5, 2)
education_level = st.sidebar.selectbox("Nivel Educativo", [
    "Unknown", "Uneducated", "High School", "College",
    "Graduate", "Post-Graduate", "Doctorate"
])
marital_status = st.sidebar.selectbox("Estado Civil", [
    "Divorced", "Married", "Single", "Unknown"
])
income_category = st.sidebar.selectbox("Categoría de Ingreso", [
    "Unknown", "Less than $40K", "$40K - $60K",
    "$60K - $80K", "$80K - $120K", "$120K +"
])

# -- Tarjeta --
st.sidebar.subheader("Tarjeta")
card_category = st.sidebar.selectbox("Tipo de Tarjeta", [
    "Blue", "Silver", "Gold", "Platinum"
])
months_on_book = st.sidebar.slider("Meses como cliente", 12, 60, 36)
total_relationship_count = st.sidebar.slider("Productos contratados", 1, 6, 3)

# -- Actividad --
st.sidebar.subheader("Actividad")
months_inactive = st.sidebar.slider("Meses inactivo (últimos 12)", 0, 6, 2)
contacts_count = st.sidebar.slider("Contactos con banco (últimos 12)", 0, 6, 2)

# -- Financieras --
st.sidebar.subheader("Financieras")
credit_limit = st.sidebar.number_input("Límite de crédito ($)", 1000, 40000, 8000, step=500)
total_revolving_bal = st.sidebar.number_input("Saldo revolving ($)", 0, 2600, 1000, step=100)
avg_open_to_buy = st.sidebar.number_input("Open to Buy ($)", 0, 40000, 7000, step=500)
total_amt_chng = st.sidebar.slider("Cambio monto Q4/Q1", 0.0, 3.5, 0.7, step=0.05)
total_trans_amt = st.sidebar.number_input("Monto total transacciones ($)", 500, 20000, 4000, step=500)
total_trans_ct = st.sidebar.slider("Cantidad total transacciones", 10, 140, 60)
total_ct_chng = st.sidebar.slider("Cambio cantidad trans Q4/Q1", 0.0, 4.0, 0.7, step=0.05)
avg_utilization = st.sidebar.slider("Ratio utilización promedio", 0.0, 1.0, 0.3, step=0.01)

# =============================================================
# PREPROCESAMIENTO (replica el pipeline de churn.py)
# =============================================================

# Encoding categóricas
gender_enc = 1 if gender == "M" else 0
card_enc = {"Blue": 0, "Silver": 1, "Gold": 2, "Platinum": 3}[card_category]
edu_enc = {"Unknown": -1, "Uneducated": 0, "High School": 1, "College": 2,
           "Graduate": 3, "Post-Graduate": 4, "Doctorate": 5}[education_level]
income_enc = {"Unknown": -1, "Less than $40K": 0, "$40K - $60K": 1,
              "$60K - $80K": 2, "$80K - $120K": 3, "$120K +": 4}[income_category]

# One-hot Marital_Status (drop_first=True, orden alfabético -> Divorced se elimina)
marital_married = 1 if marital_status == "Married" else 0
marital_single = 1 if marital_status == "Single" else 0
marital_unknown = 1 if marital_status == "Unknown" else 0

# Feature engineering
avg_trans_value = total_trans_amt / total_trans_ct if total_trans_ct > 0 else 0
activity_index = total_trans_ct / months_on_book if months_on_book > 0 else 0
contact_per_inactive = contacts_count / (months_inactive + 1)

# Construir diccionario con TODAS las features posibles
all_features = {
    'Customer_Age': customer_age,
    'Gender': gender_enc,
    'Dependent_count': dependent_count,
    'Education_Level': edu_enc,
    'Income_Category': income_enc,
    'Card_Category': card_enc,
    'Months_on_book': months_on_book,
    'Total_Relationship_Count': total_relationship_count,
    'Months_Inactive_12_mon': months_inactive,
    'Contacts_Count_12_mon': contacts_count,
    'Credit_Limit': credit_limit,
    'Total_Revolving_Bal': total_revolving_bal,
    'Avg_Open_To_Buy': avg_open_to_buy,
    'Total_Amt_Chng_Q4_Q1': total_amt_chng,
    'Total_Trans_Amt': total_trans_amt,
    'Total_Trans_Ct': total_trans_ct,
    'Total_Ct_Chng_Q4_Q1': total_ct_chng,
    'Avg_Utilization_Ratio': avg_utilization,
    'Marital_Status_Married': marital_married,
    'Marital_Status_Single': marital_single,
    'Marital_Status_Unknown': marital_unknown,
    'Avg_Trans_Value': avg_trans_value,
    'Activity_Index': activity_index,
    'Contact_per_Inactive': contact_per_inactive
}

# Seleccionar solo las features que el modelo espera
input_df = pd.DataFrame([{f: all_features[f] for f in feature_names}])

# Escalar
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

# =============================================================
# PREDICCIÓN
# =============================================================
proba = modelo.predict_proba(input_scaled)[0][1]
pred = modelo.predict(input_scaled)[0]
proba_pct = round(proba * 100, 2)

# Determinar segmento de riesgo
if proba <= 0.3:
    segmento = "Bajo"
    color = "#2ecc71"
    emoji = "🟢"
elif proba <= 0.6:
    segmento = "Medio"
    color = "#f39c12"
    emoji = "🟡"
elif proba <= 0.8:
    segmento = "Alto"
    color = "#e67e22"
    emoji = "🟠"
else:
    segmento = "Muy Alto"
    color = "#e74c3c"
    emoji = "🔴"

# =============================================================
# VISUALIZACIÓN DE RESULTADOS
# =============================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Probabilidad de Churn", f"{proba_pct}%")

with col2:
    st.markdown(
        f"<div style='text-align:center;'>"
        f"<span style='font-size:18px;'>Segmento de Riesgo</span><br>"
        f"<span style='font-size:48px; font-weight:bold; color:{color};'>"
        f"{emoji} {segmento}</span></div>",
        unsafe_allow_html=True
    )

with col3:
    estado = "Churn" if pred == 1 else "Existente"
    st.metric("Predicción del Modelo", estado)

# Barra de progreso visual
st.divider()
st.markdown("**Nivel de riesgo:**")
st.progress(min(proba, 1.0))

# Resumen del perfil
st.divider()
st.subheader("Resumen del Perfil")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**Demográfico**")
    st.write(f"Edad: {customer_age}")
    st.write(f"Género: {gender}")
    st.write(f"Dependientes: {dependent_count}")
    st.write(f"Educación: {education_level}")
    st.write(f"Estado civil: {marital_status}")
    st.write(f"Ingreso: {income_category}")

with c2:
    st.markdown("**Tarjeta**")
    st.write(f"Tipo: {card_category}")
    st.write(f"Meses cliente: {months_on_book}")
    st.write(f"Productos: {total_relationship_count}")

with c3:
    st.markdown("**Actividad**")
    st.write(f"Meses inactivo: {months_inactive}")
    st.write(f"Contactos: {contacts_count}")
    st.write(f"Transacciones: {total_trans_ct}")
    st.write(f"Monto trans.: ${total_trans_amt:,}")

with c4:
    st.markdown("**Financiero**")
    st.write(f"Límite crédito: ${credit_limit:,}")
    st.write(f"Saldo revolving: ${total_revolving_bal:,}")
    st.write(f"Utilización: {avg_utilization:.0%}")
