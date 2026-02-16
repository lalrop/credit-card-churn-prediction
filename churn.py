import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# ============================================================
# FASE 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================
ruta_script = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(ruta_script, 'bbdd.csv'), sep=';')

# --- 1.1 Información general del dataset ---
print("=" * 60)
print("INFORMACIÓN GENERAL DEL DATASET")
print("=" * 60)
print(f"\nDimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nColumnas:\n{list(df.columns)}")
print(f"\nTipos de datos:\n{df.dtypes}")
print(f"\nValores nulos por columna:\n{df.isnull().sum()}")
print(f"\nValores nulos totales: {df.isnull().sum().sum()}")
print(f"\nDuplicados: {df.duplicated().sum()}")
print(f"\nPrimeras filas:\n{df.head()}")

# --- 1.2 Distribución de la variable objetivo ---
print("\n" + "=" * 60)
print("DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
print("=" * 60)
conteo_target = df['Attrition_Flag'].value_counts()
porcentaje_target = df['Attrition_Flag'].value_counts(normalize=True) * 100
print(f"\nConteo:\n{conteo_target}")
print(f"\nPorcentaje:\n{porcentaje_target.round(2)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2ecc71', '#e74c3c']

conteo_target.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
axes[0].set_title('Distribución de Clientes (Churn vs No Churn)')
axes[0].set_xlabel('Estado del Cliente')
axes[0].set_ylabel('Cantidad')
axes[0].tick_params(axis='x', rotation=0)
for i, v in enumerate(conteo_target):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

axes[1].pie(conteo_target, labels=conteo_target.index, autopct='%1.1f%%',
            colors=colors, startangle=90, explode=(0, 0.05))
axes[1].set_title('Proporción Churn vs No Churn')
plt.tight_layout()
plt.savefig(os.path.join(ruta_script, 'fig_01_distribucion_target.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- 1.3 Estadísticas descriptivas ---
print("\n" + "=" * 60)
print("ESTADÍSTICAS DESCRIPTIVAS - VARIABLES NUMÉRICAS")
print("=" * 60)

cols_numericas = df.select_dtypes(include=[np.number]).columns.drop('CLIENTNUM')
print(f"\n{df[cols_numericas].describe().T}")
# Importante, se observar un desbalance en la variable objetivo, con una mayoria en los clientes existentes por sobre los clientes "Attrited". Esto es un punto a tener en cuenta para la fase de modelado, ya que podría requerir técnicas de balanceo o ajuste de métricas para evaluar el desempeño del modelo de manera adecuada. Además, se pueden observar algunas variables con valores máximos muy altos (como 'Credit_Limit' y 'Total_Revolving_Bal'), lo que sugiere la presencia de outliers que podrían afectar el análisis y modelado posterior.

# --- 1.4 Distribuciones de variables numéricas ---
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(cols_numericas):
    ax = axes[i]
    df[df['Attrition_Flag'] == 'Existing Customer'][col].hist(
        ax=ax, bins=30, alpha=0.6, color='#2ecc71', label='Existente', density=True)
    df[df['Attrition_Flag'] == 'Attrited Customer'][col].hist(
        ax=ax, bins=30, alpha=0.6, color='#e74c3c', label='Churn', density=True)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribución de Variables Numéricas por Clase', fontsize=14, y=1.02)
plt.tight_layout(h_pad=3.5)
plt.savefig(os.path.join(ruta_script, 'fig_02_distribuciones_numericas.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- 1.5 Variables categóricas ---
print("\n" + "=" * 60)
print("DISTRIBUCIÓN DE VARIABLES CATEGÓRICAS")
print("=" * 60)

cols_categoricas = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

for col in cols_categoricas:
    print(f"\n{col}:\n{df[col].value_counts()}")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, col in enumerate(cols_categoricas):
    ax = axes[i]
    ct = pd.crosstab(df[col], df['Attrition_Flag'], normalize='index') * 100
    ct.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
    ax.set_title(f'Tasa de Churn por {col}')
    ax.set_ylabel('Porcentaje (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Estado')

axes[-1].set_visible(False)
plt.suptitle('Variables Categóricas vs Churn', fontsize=14, y=1.02)
plt.tight_layout(h_pad=3.5)
plt.savefig(os.path.join(ruta_script, 'fig_03_categoricas_vs_churn.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- 1.6 Matriz de correlación ---
print("\n" + "=" * 60)
print("MATRIZ DE CORRELACIÓN")
print("=" * 60)

df_corr = df.copy()
df_corr['Churn'] = (df_corr['Attrition_Flag'] == 'Attrited Customer').astype(int)
correlacion = df_corr[list(cols_numericas) + ['Churn']].corr()

plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(correlacion, dtype=bool))
sns.heatmap(correlacion, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title('Matriz de Correlación (incluye variable Churn)')
plt.tight_layout()
plt.savefig(os.path.join(ruta_script, 'fig_04_correlacion.png'), dpi=150, bbox_inches='tight')
plt.show()

# Correlaciones más relevantes con Churn
print("\nCorrelación con Churn (ordenada):")
corr_churn = correlacion['Churn'].drop('Churn').sort_values()
print(corr_churn)

# --- 1.7 Análisis comparativo por clase ---
print("\n" + "=" * 60)
print("COMPARATIVA DE MEDIAS: CHURN vs NO CHURN")
print("=" * 60)

comparativa = df.groupby('Attrition_Flag')[cols_numericas].mean().T
comparativa['Diferencia_%'] = ((comparativa['Attrited Customer'] - comparativa['Existing Customer'])
                                / comparativa['Existing Customer'] * 100).round(2)
print(f"\n{comparativa}")

# Boxplots de features más discriminantes
top_features = corr_churn.abs().sort_values(ascending=False).head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(top_features):
    sns.boxplot(data=df, x='Attrition_Flag', y=col, ax=axes[i],
                hue='Attrition_Flag', palette={'Existing Customer': '#2ecc71', 'Attrited Customer': '#e74c3c'},
                legend=False)
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel('')

plt.suptitle('Top 6 Features más Correlacionadas con Churn', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(ruta_script, 'fig_05_top_features_boxplot.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("FIN DE FASE 1 - EDA COMPLETADO")
print("=" * 60)

# ============================================================
# FASE 2: PREPROCESAMIENTO
# ============================================================
print("\n\n" + "=" * 60)
print("FASE 2: PREPROCESAMIENTO")
print("=" * 60)

# --- 2.1 Preparación inicial ---
df_prep = df.copy()
df_prep = df_prep.drop(columns=['CLIENTNUM'])
df_prep['Churn'] = (df_prep['Attrition_Flag'] == 'Attrited Customer').astype(int)
df_prep = df_prep.drop(columns=['Attrition_Flag'])

print(f"\nColumnas tras eliminar CLIENTNUM y crear objetivo 'Churn'")
print(f"{list(df_prep.columns)}")

# --- 2.2 Codificación de variables categóricas ---
print("\n" + "-" * 40)
print("CODIFICACIÓN DE VARIABLES CATEGÓRICAS")
print("-" * 40)

# Gender: mapping binario
df_prep['Gender'] = df_prep['Gender'].map({'M': 1, 'F': 0})

# Card_Category: encoding ordinal
# Asignamos un orden logico a la variable Card_category, donde "blue" es el mas bajo y "platinum" es el mas alto

card_orden = {'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
df_prep['Card_Category'] = df_prep['Card_Category'].map(card_orden)

# Education_Level: encoding ordinal
# Al igual que con card_category, asignamos un orden logico a Education_level.

edu_orden = {'Unknown': -1, 'Uneducated': 0, 'High School': 1, 'College': 2,
             'Graduate': 3, 'Post-Graduate': 4, 'Doctorate': 5}
df_prep['Education_Level'] = df_prep['Education_Level'].map(edu_orden)

# Income_Category: encoding ordinal
income_orden = {'Unknown': -1, 'Less than $40K': 0, '$40K - $60K': 1,
                '$60K - $80K': 2, '$80K - $120K': 3, '$120K +': 4}
df_prep['Income_Category'] = df_prep['Income_Category'].map(income_orden)

# Marital_Status: one-hot encoding (no tiene orden natural)
df_prep = pd.get_dummies(df_prep, columns=['Marital_Status'],
                         drop_first=True, dtype=int)

print(f"\nColumnas después del encoding ({len(df_prep.columns)} total):")
print(f"{list(df_prep.columns)}")
print(f"\nFIN DE FASE 2 - PREPROCESAMIENTO COMPLETADO")

# ============================================================
# FASE 3: FEATURE ENGINEERING
# ============================================================
print("\n\n" + "=" * 60)
print("FASE 3: FEATURE ENGINEERING")
print("=" * 60)

n_features_antes = df_prep.shape[1] - 1  # -1 por Churn

# --- 3.1 Creación de nuevas features ---
print("\n" + "-" * 40)
print("CREACIÓN DE NUEVAS FEATURES")
print("-" * 40)

df_prep['Avg_Trans_Value'] = df_prep['Total_Trans_Amt'] / df_prep['Total_Trans_Ct']
df_prep['Activity_Index'] = df_prep['Total_Trans_Ct'] / df_prep['Months_on_book']
df_prep['Contact_per_Inactive'] = df_prep['Contacts_Count_12_mon'] / (df_prep['Months_Inactive_12_mon'] + 1)

print(f"\nNuevas features creadas:")
print(f"  - Avg_Trans_Value = Total_Trans_Amt / Total_Trans_Ct")
print(f"  - Activity_Index = Total_Trans_Ct / Months_on_book")
print(f"  - Contact_per_Inactive = Contacts_Count_12_mon / (Months_Inactive_12_mon + 1)")

# --- 3.2 Selección por correlación con target ---
print("\n" + "-" * 40)
print("SELECCIÓN POR CORRELACIÓN CON TARGET")
print("-" * 40)

corr_con_churn = df_prep.corr()['Churn'].drop('Churn').abs().sort_values(ascending=False)
print(f"\nCorrelación absoluta con Churn:")
print(corr_con_churn.to_string())

# --- 3.3 Eliminación de features redundantes ---
print("\n" + "-" * 40)
print("ELIMINACIÓN DE FEATURES REDUNDANTES")
print("-" * 40)

# Matriz de correlación entre features (sin Churn)
features_cols = [col for col in df_prep.columns if col != 'Churn']
corr_matrix = df_prep[features_cols].corr().abs()

# Extraer pares con alta correlación (umbral > 0.85)
umbral_corr = 0.85
mascara = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
pares_alta_corr = corr_matrix.where(mascara).stack()
pares_alta_corr = pares_alta_corr[pares_alta_corr > umbral_corr].sort_values(ascending=False)

print(f"\nPares de features con correlación > {umbral_corr}:")
if len(pares_alta_corr) > 0:
    for (f1, f2), corr_val in pares_alta_corr.items():
        print(f"  {f1} vs {f2}: {corr_val:.4f}")

    # Para cada par, eliminar la feature con menor correlación con Churn
    corr_con_target = df_prep.corr()['Churn'].abs()
    eliminadas = set()
    for (f1, f2), _ in pares_alta_corr.items():
        if f1 in eliminadas or f2 in eliminadas:
            continue
        if corr_con_target[f1] >= corr_con_target[f2]:
            eliminadas.add(f2)
            print(f"  -> Eliminada: {f2} (menor correlación con Churn que {f1})")
        else:
            eliminadas.add(f1)
            print(f"  -> Eliminada: {f1} (menor correlación con Churn que {f2})")
    df_prep = df_prep.drop(columns=list(eliminadas))
else:
    print(f"  No se encontraron pares altamente correlacionados")

umbral = 0.02
features_baja_corr = corr_con_churn[corr_con_churn < umbral].index.tolist()
# Filtrar solo las que aún existen (algunas pueden haber sido eliminadas en 3.3)
features_baja_corr = [f for f in features_baja_corr if f in df_prep.columns]
if features_baja_corr:
    print(f"\nFeatures eliminadas (corr < {umbral}): {features_baja_corr}")
    df_prep = df_prep.drop(columns=features_baja_corr)
else:
    print(f"\nNo se eliminaron features adicionales (todas tienen corr >= {umbral})")

# --- 3.4 Separación features / target ---
y = df_prep['Churn']
X = df_prep.drop(columns=['Churn'])

print(f"\nDimensiones X: {X.shape}")
print(f"Dimensiones y: {y.shape}")

# --- 3.5 División train/test (80/20 estratificado) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n" + "-" * 40)
print("DIVISIÓN TRAIN/TEST")
print("-" * 40)
print(f"\nTrain: {X_train.shape[0]} filas ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test:  {X_test.shape[0]} filas ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nDistribución en train:\n{y_train.value_counts()}")
print(f"\nDistribución en test:\n{y_test.value_counts()}")

# --- 3.6 Escalado de variables numéricas ---
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(f"\n" + "-" * 40)
print("ESCALADO (StandardScaler)")
print("-" * 40)
print(f"\nMedia de X_train_scaled (primeras 5):\n{X_train_scaled.mean().head()}")
print(f"\nStd de X_train_scaled (primeras 5):\n{X_train_scaled.std().head()}")

# --- 3.7 Balance de clases ---
# Se usará class_weight='balanced' directamente en los modelos (Fase 4)
# Esto penaliza más los errores en la clase minoritaria sin alterar los datos
print(f"\n" + "-" * 40)
print("BALANCE DE CLASES")
print("-" * 40)
print(f"\nEstrategia: class_weight='balanced' (se aplicará en Fase 4 - Modelado)")
print(f"  Clase 0 (Existente): {(y_train == 0).sum()}")
print(f"  Clase 1 (Churn):     {(y_train == 1).sum()}")
print(f"  Ratio: {(y_train == 0).sum() / (y_train == 1).sum():.2f}:1")

# --- Resumen Fase 3 ---
n_features_despues = X_train_scaled.shape[1]
print(f"\n" + "=" * 60)
print("RESUMEN FASE 3")
print("=" * 60)
print(f"\nFeatures antes: {n_features_antes}")
print(f"Features después: {n_features_despues}")
print(f"Features finales: {list(X_train_scaled.columns)}")
print(f"Train: {X_train_scaled.shape[0]} filas")
print(f"Test: {X_test_scaled.shape[0]} filas")
print(f"\nFIN DE FASE 3 - FEATURE ENGINEERING COMPLETADO")

# ============================================================
# FASE 4: MODELADO
# ============================================================
print("\n\n" + "=" * 60)
print("FASE 4: MODELADO")
print("=" * 60)

# --- 4.1 Definición de modelos ---
# Ratio para scale_pos_weight de XGBoost (clase mayoritaria / clase minoritaria)
ratio_clases = (y_train == 0).sum() / (y_train == 1).sum()

modelos = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
    ),
    'HistGradientBoosting': HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_depth=4,
        class_weight='balanced', random_state=42
    ),
    'SVM': SVC(
        class_weight='balanced', kernel='rbf', probability=True, random_state=42
    )
}

# --- 4.2 Entrenamiento y predicción ---
print("\n" + "-" * 40)
print("ENTRENAMIENTO DE MODELOS")
print("-" * 40)

resultados = {}

for nombre, modelo in modelos.items():
    print(f"\nEntrenando {nombre}...")
    modelo.fit(X_train_scaled, y_train)

    y_pred = modelo.predict(X_test_scaled)
    y_proba = modelo.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    resultados[nombre] = {
        'modelo': modelo,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc
    }
#     print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# --- 4.3 Validación cruzada ---
print("\n" + "-" * 40)
print("VALIDACIÓN CRUZADA (5-Fold, métrica=F1)")
print("-" * 40)

for nombre, modelo in modelos.items():
    cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5, scoring='f1')
    resultados[nombre]['CV_F1_mean'] = cv_scores.mean()
    resultados[nombre]['CV_F1_std'] = cv_scores.std()
    print(f"  {nombre}: F1 = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- 4.4 Tabla comparativa ---
print("\n" + "-" * 40)
print("COMPARATIVA DE MODELOS")
print("-" * 40)

metricas_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'CV_F1_mean']
df_resultados = pd.DataFrame(
    {nombre: {m: resultados[nombre][m] for m in metricas_cols}
     for nombre in resultados}
).T.sort_values('F1-Score', ascending=False)

print(f"\n{df_resultados.round(4).to_string()}")

# # --- 4.5 Selección del mejor modelo ---
mejor_nombre = df_resultados['F1-Score'].idxmax()
mejor_modelo = resultados[mejor_nombre]['modelo']

print(f"\nMejor modelo por F1-Score: {mejor_nombre}")
print(f"  F1-Score: {resultados[mejor_nombre]['F1-Score']:.4f}")
print(f"  AUC-ROC:  {resultados[mejor_nombre]['AUC-ROC']:.4f}")
print(f"  CV F1:    {resultados[mejor_nombre]['CV_F1_mean']:.4f} ± {resultados[mejor_nombre]['CV_F1_std']:.4f}")

# --- 4.6 Visualización comparativa ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Barplot de métricas
df_plot = df_resultados[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']]
df_plot.plot(kind='bar', ax=axes[0], edgecolor='black', width=0.8)
axes[0].set_title('Comparativa de Métricas por Modelo')
axes[0].set_ylabel('Score')
axes[0].set_ylim(0, 1.05)
axes[0].tick_params(axis='x', rotation=30)
axes[0].legend(loc='lower right', fontsize=8)

# Curvas ROC
for nombre in resultados:
    fpr, tpr, _ = roc_curve(y_test, resultados[nombre]['y_proba'])
    axes[1].plot(fpr, tpr, label=f"{nombre} (AUC={resultados[nombre]['AUC-ROC']:.3f})")

axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
axes[1].set_title('Curvas ROC')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(ruta_script, 'fig_06_comparativa_modelos.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- 4.7 Matrices de confusión ---
fig, axes = plt.subplots(1, len(modelos), figsize=(4 * len(modelos), 4))
for i, nombre in enumerate(df_resultados.index):
    ConfusionMatrixDisplay.from_predictions(
        y_test, resultados[nombre]['y_pred'], ax=axes[i],
        cmap='Blues', display_labels=['Existente', 'Churn']
    )
    axes[i].set_title(f'{nombre}\nF1={resultados[nombre]["F1-Score"]:.3f}', fontsize=10)

plt.suptitle('Matrices de Confusión', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(ruta_script, 'fig_07_matrices_confusion.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- Resumen Fase 4 ---
print(f"\n" + "=" * 60)
print("RESUMEN FASE 4")
print("=" * 60)
print(f"\nModelos entrenados: {len(modelos)}")
print(f"Mejor modelo: {mejor_nombre}")
print(f"  F1-Score: {resultados[mejor_nombre]['F1-Score']:.4f}")
print(f"  AUC-ROC:  {resultados[mejor_nombre]['AUC-ROC']:.4f}")
print(f"  Recall:   {resultados[mejor_nombre]['Recall']:.4f}")
print(f"\nFIN DE FASE 4 - MODELADO COMPLETADO")

# ============================================================
# FASE 5: EVALUACIÓN DETALLADA
# ============================================================
from sklearn.metrics import precision_recall_curve, average_precision_score

print("\n\n" + "=" * 60)
print("FASE 5: EVALUACIÓN DETALLADA")
print("=" * 60)

# --- 5.1 Classification Report del mejor modelo ---
print("\n" + "-" * 40)
print(f"CLASSIFICATION REPORT - {mejor_nombre}")
print("-" * 40)

y_pred_mejor = resultados[mejor_nombre]['y_pred']
y_proba_mejor = resultados[mejor_nombre]['y_proba']

print(classification_report(y_test, y_pred_mejor,
                            target_names=['Existente', 'Churn'], digits=4))

# --- 5.2 Análisis de errores ---
print("\n" + "-" * 40)
print("ANÁLISIS DE ERRORES")
print("-" * 40)

cm = confusion_matrix(y_test, y_pred_mejor)
tn, fp, fn, tp = cm.ravel()

print(f"\n  Verdaderos Negativos (TN): {tn}  - Existentes correctamente clasificados")
print(f"  Falsos Positivos    (FP): {fp}  - Existentes predichos como Churn")
print(f"  Falsos Negativos    (FN): {fn}  - Churn NO detectados (el error más costoso)")
print(f"  Verdaderos Positivos(TP): {tp}  - Churn correctamente detectados")

tasa_deteccion = tp / (tp + fn) * 100
tasa_falsa_alarma = fp / (fp + tn) * 100
print(f"\n  Tasa de detección de Churn: {tasa_deteccion:.1f}%")
print(f"  Tasa de falsa alarma: {tasa_falsa_alarma:.1f}%")

# --- 5.3 Optimización de umbral ---
print("\n" + "-" * 40)
print("OPTIMIZACIÓN DE UMBRAL DE DECISIÓN")
print("-" * 40)

precision_curve, recall_curve, umbrales_pr = precision_recall_curve(y_test, y_proba_mejor)
f1_scores_umbrales = 2 * (precision_curve[:-1] * recall_curve[:-1]) / (precision_curve[:-1] + recall_curve[:-1] + 1e-10)

# Mejor umbral por F1
idx_mejor_f1 = np.argmax(f1_scores_umbrales)
umbral_optimo_f1 = umbrales_pr[idx_mejor_f1]

# Mejor umbral priorizando Recall >= 0.80
mask_recall = recall_curve[:-1] >= 0.80
if mask_recall.any():
    f1_con_recall = np.where(mask_recall, f1_scores_umbrales, 0)
    idx_mejor_recall = np.argmax(f1_con_recall)
    umbral_optimo_recall = umbrales_pr[idx_mejor_recall]
else:
    umbral_optimo_recall = umbral_optimo_f1

print(f"\n  Umbral por defecto: 0.50")
print(f"  Umbral óptimo (max F1): {umbral_optimo_f1:.4f}")
print(f"  Umbral óptimo (Recall>=80%): {umbral_optimo_recall:.4f}")

# Aplicar umbral óptimo por F1
y_pred_optimizado = (y_proba_mejor >= umbral_optimo_f1).astype(int)
print(f"\n  Métricas con umbral optimizado ({umbral_optimo_f1:.4f}):")
print(f"    Accuracy:  {accuracy_score(y_test, y_pred_optimizado):.4f}")
print(f"    Precision: {precision_score(y_test, y_pred_optimizado):.4f}")
print(f"    Recall:    {recall_score(y_test, y_pred_optimizado):.4f}")
print(f"    F1-Score:  {f1_score(y_test, y_pred_optimizado):.4f}")

# --- 5.4 Visualizaciones de evaluación ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 5.4a Curva Precision-Recall
ap = average_precision_score(y_test, y_proba_mejor)
axes[0, 0].plot(recall_curve, precision_curve, color='#3498db', linewidth=2)
axes[0, 0].axvline(x=recall_curve[idx_mejor_f1], color='red', linestyle='--', alpha=0.7, label=f'Umbral óptimo={umbral_optimo_f1:.3f}')
axes[0, 0].set_title(f'Curva Precision-Recall (AP={ap:.3f})')
axes[0, 0].set_xlabel('Recall')
axes[0, 0].set_ylabel('Precision')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 5.4b F1 vs Umbral
axes[0, 1].plot(umbrales_pr, f1_scores_umbrales, color='#2ecc71', linewidth=2)
axes[0, 1].axvline(x=umbral_optimo_f1, color='red', linestyle='--', alpha=0.7, label=f'Óptimo={umbral_optimo_f1:.3f}')
axes[0, 1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default=0.5')
axes[0, 1].set_title('F1-Score vs Umbral de Decisión')
axes[0, 1].set_xlabel('Umbral')
axes[0, 1].set_ylabel('F1-Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 5.4c Distribución de probabilidades predichas
for label, color, nombre_clase in [(0, '#2ecc71', 'Existente'), (1, '#e74c3c', 'Churn')]:
    axes[1, 0].hist(y_proba_mejor[y_test == label], bins=50, alpha=0.6,
                     color=color, label=nombre_clase, density=True)
axes[1, 0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Umbral=0.5')
axes[1, 0].axvline(x=umbral_optimo_f1, color='red', linestyle='--', alpha=0.7, label=f'Óptimo={umbral_optimo_f1:.3f}')
axes[1, 0].set_title('Distribución de Probabilidades Predichas')
axes[1, 0].set_xlabel('Probabilidad de Churn')
axes[1, 0].set_ylabel('Densidad')
axes[1, 0].legend(fontsize=8)

# 5.4d Matriz de confusión con umbral optimizado
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_optimizado, ax=axes[1, 1],
    cmap='Blues', display_labels=['Existente', 'Churn']
)
axes[1, 1].set_title(f'Matriz de Confusión\n(Umbral optimizado: {umbral_optimo_f1:.3f})')

plt.suptitle(f'Evaluación Detallada - {mejor_nombre}', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(ruta_script, 'fig_08_evaluacion_detallada.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- Resumen Fase 5 ---
print(f"\n" + "=" * 60)
print("RESUMEN FASE 5")
print("=" * 60)
print(f"\nModelo evaluado: {mejor_nombre}")
print(f"  Umbral default (0.50)  -> F1: {resultados[mejor_nombre]['F1-Score']:.4f} | Recall: {resultados[mejor_nombre]['Recall']:.4f}")
print(f"  Umbral optimizado ({umbral_optimo_f1:.3f}) -> F1: {f1_score(y_test, y_pred_optimizado):.4f} | Recall: {recall_score(y_test, y_pred_optimizado):.4f}")
print(f"  Churn no detectados (FN): {fn} con umbral default")
cm_opt = confusion_matrix(y_test, y_pred_optimizado)
fn_opt = cm_opt[1, 0]
print(f"  Churn no detectados (FN): {fn_opt} con umbral optimizado")
print(f"\nFIN DE FASE 5 - EVALUACIÓN COMPLETADA")

# ============================================================
# FASE 6: INTERPRETABILIDAD
# ============================================================

# Para esta fase importamos SHAP (SHapley Additive Explanations) el cual es un metodo de interpretabilidad basado en teoria de juegos, que asigna a cada feature una contribución al resultado de la predicción. Es especialmente útil para modelos complejos como Random Forest o Gradient Boosting, ya que permite entender cómo cada feature influye en la decisión del modelo a nivel global y local (por muestra)
# A diferencia del Feature Importance nativo del modelo, este ademaas de mostrar la importancia global, muestra la direccion del impacto de cada feature.

import shap
from sklearn.inspection import permutation_importance

print("\n\n" + "=" * 60)
print("FASE 6: INTERPRETABILIDAD")
print("=" * 60)

# --- 6.1 Feature Importance del mejor modelo ---
print("\n" + "-" * 40)
print(f"FEATURE IMPORTANCE - {mejor_nombre}")
print("-" * 40)

feature_names = X_train_scaled.columns.tolist()

# Importancia nativa (si el modelo la tiene)
if hasattr(mejor_modelo, 'feature_importances_'):
    importancia = pd.Series(mejor_modelo.feature_importances_, index=feature_names)
    importancia = importancia.sort_values(ascending=False)
    print(f"\nImportancia nativa del modelo:")
    print(importancia.to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    importancia.sort_values().plot(kind='barh', ax=ax, color='#3498db', edgecolor='black')
    ax.set_title(f'Feature Importance - {mejor_nombre}')
    ax.set_xlabel('Importancia')
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_script, 'fig_09_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()

# --- 6.2 Permutation Importance ---
print("\n" + "-" * 40)
print("PERMUTATION IMPORTANCE (sobre test set)")
print("-" * 40)

perm_imp = permutation_importance(mejor_modelo, X_test_scaled, y_test,
                                   n_repeats=20, random_state=42, scoring='f1')
perm_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importancia': perm_imp.importances_mean,
    'Std': perm_imp.importances_std
}).sort_values('Importancia', ascending=False)

print(f"\n{perm_imp_df.to_string(index=False)}")

fig, ax = plt.subplots(figsize=(10, 6))
perm_imp_sorted = perm_imp_df.sort_values('Importancia')
ax.barh(perm_imp_sorted['Feature'], perm_imp_sorted['Importancia'],
        xerr=perm_imp_sorted['Std'], color='#e74c3c', edgecolor='black', capsize=3)
ax.set_title(f'Permutation Importance - {mejor_nombre}')
ax.set_xlabel('Disminución media en F1-Score')
plt.tight_layout()
plt.savefig(os.path.join(ruta_script, 'fig_10_permutation_importance.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- 6.3 SHAP Values ---
print("\n" + "-" * 40)
print("SHAP VALUES")
print("-" * 40)

# Usar muestra para eficiencia
X_shap = X_test_scaled.sample(n=min(200, len(X_test_scaled)), random_state=42)

# Seleccionar explainer según tipo de modelo
if hasattr(mejor_modelo, 'feature_importances_'):
    # Modelos basados en árboles (Random Forest, GradientBoosting, etc.)
    explainer = shap.TreeExplainer(mejor_modelo)
    shap_values = explainer(X_shap)
elif hasattr(mejor_modelo, 'coef_'):
    # Modelos lineales (Logistic Regression)
    explainer = shap.LinearExplainer(mejor_modelo, X_train_scaled)
    shap_values = explainer(X_shap)
else:
    # Cualquier otro modelo (SVM, etc.) - usa KernelExplainer (model-agnostic)
    background = shap.sample(X_train_scaled, 100)
    explainer = shap.KernelExplainer(mejor_modelo.predict_proba, background)
    shap_values_raw = explainer.shap_values(X_shap)
    # KernelExplainer devuelve una lista por clase, tomamos clase 1 (Churn)
    shap_values = shap.Explanation(
        values=shap_values_raw[:, :, 1],
        base_values=explainer.expected_value[1],
        data=X_shap.values,
        feature_names=feature_names
    )

print(f"\nSHAP values calculados para {X_shap.shape[0]} muestras")

# 6.3a SHAP Summary Plot (beeswarm)
# En este grafico se observa como cada punto representa un cliente, luego la posicion horozontal indica si la feature (del eje vertical) lo inclina hacia churn (derecha) o existente (izquierda). El color del punto indica el valor de la feature, rojo es un valor alto y azul bajo.
fig, ax = plt.subplots(figsize=(12, 7))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title(f'SHAP Summary Plot - {mejor_nombre}')
plt.tight_layout()
plt.savefig(os.path.join(ruta_script, 'fig_11_shap_summary.png'), dpi=150, bbox_inches='tight')
plt.show()

# 6.3b SHAP Bar Plot (importancia media absoluta)
# Esta grafica resume en parte lo mostrado con el Summary Plot, pero ordenando las features por su impacto absoluto medio en la prediccion.
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title(f'SHAP Feature Importance - {mejor_nombre}')
plt.tight_layout()
plt.savefig(os.path.join(ruta_script, 'fig_12_shap_bar.png'), dpi=150, bbox_inches='tight')
plt.show()

# 6.3c SHAP Dependence Plots (top 3 features)
# Muestra la relación directa entre el valor de una feature y su impacto SHAP para las 3 features más importantes
top_3_shap = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_abs_mean': np.abs(shap_values.values).mean(axis=0)
}).sort_values('SHAP_abs_mean', ascending=False).head(3)['Feature'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, feat in enumerate(top_3_shap):
    shap.plots.scatter(shap_values[:, feat], ax=axes[i], show=False)
    axes[i].set_title(f'SHAP Dependence: {feat}')

plt.suptitle('Dependencia SHAP - Top 3 Features', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(ruta_script, 'fig_13_shap_dependence.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- Resumen Fase 6 ---
print(f"\n" + "=" * 60)
print("RESUMEN FASE 6")
print("=" * 60)
print(f"\nModelo interpretado: {mejor_nombre}")
print(f"\nTop 5 features por SHAP (impacto absoluto medio):")
shap_ranking = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_abs_mean': np.abs(shap_values.values).mean(axis=0)
}).sort_values('SHAP_abs_mean', ascending=False)
for _, row in shap_ranking.head(5).iterrows():
    print(f"  {row['Feature']}: {row['SHAP_abs_mean']:.4f}")

print(f"\nTop 5 features por Permutation Importance:")
for _, row in perm_imp_df.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importancia']:.4f} ± {row['Std']:.4f}")

print(f"\nFIN DE FASE 6 - INTERPRETABILIDAD COMPLETADA")

# ============================================================
# FASE 7: EXPORTACIÓN DE DATOS PARA POWER BI
# ============================================================
print(f"\n{'=' * 60}")
print("FASE 7: EXPORTACIÓN PARA POWER BI")
print(f"{'=' * 60}")

carpeta_powerbi = os.path.join(ruta_script, 'powerbi_data')
os.makedirs(carpeta_powerbi, exist_ok=True)

# --- 7.0 Guardar modelo, scaler y features para la app de predicción ---
import joblib
joblib.dump(mejor_modelo, os.path.join(ruta_script, 'modelo_churn.pkl'))
joblib.dump(scaler, os.path.join(ruta_script, 'scaler_churn.pkl'))
joblib.dump(feature_names, os.path.join(ruta_script, 'feature_names.pkl'))
joblib.dump(mejor_nombre, os.path.join(ruta_script, 'modelo_nombre.pkl'))
print(f"\n  Modelo guardado: modelo_churn.pkl ({mejor_nombre})")
print(f"  Scaler guardado: scaler_churn.pkl")
print(f"  Features guardadas: {feature_names}")

# --- 7.1 Predicciones de TODOS los clientes ---
print(f"\n--- 7.1 Predicciones de todos los clientes ---")

# Escalar todo el dataset con el scaler ya entrenado
X_all = df_prep.drop(columns=['Churn'])
y_all = df_prep['Churn']
X_all_scaled = pd.DataFrame(scaler.transform(X_all), columns=X_all.columns, index=X_all.index)

# Predecir con el mejor modelo sobre TODOS los clientes
y_pred_all = mejor_modelo.predict(X_all_scaled)
y_proba_all = mejor_modelo.predict_proba(X_all_scaled)[:, 1]

# Construir tabla con features ORIGINALES legibles + predicciones
df_predicciones = df[['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender',
                       'Dependent_count', 'Education_Level', 'Marital_Status',
                       'Income_Category', 'Card_Category', 'Months_on_book',
                       'Total_Relationship_Count', 'Months_Inactive_12_mon',
                       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                       'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
                       'Avg_Utilization_Ratio']].copy()

df_predicciones['Churn_Real'] = y_all.values
df_predicciones['Churn_Predicho'] = y_pred_all
df_predicciones['Probabilidad_Churn'] = y_proba_all
df_predicciones['Probabilidad_Churn_%'] = (y_proba_all * 100).round(2)

# Segmentar riesgo por probabilidad
df_predicciones['Segmento_Riesgo'] = pd.cut(
    df_predicciones['Probabilidad_Churn'],
    bins=[0, 0.3, 0.6, 0.8, 1.0],
    labels=['Bajo', 'Medio', 'Alto', 'Muy Alto']
)

# Clasificación del resultado
df_predicciones['Clasificacion'] = df_predicciones.apply(
    lambda row: 'Verdadero Positivo' if row['Churn_Real'] == 1 and row['Churn_Predicho'] == 1
    else 'Falso Negativo' if row['Churn_Real'] == 1 and row['Churn_Predicho'] == 0
    else 'Falso Positivo' if row['Churn_Real'] == 0 and row['Churn_Predicho'] == 1
    else 'Verdadero Negativo', axis=1
)

# Estado legible
df_predicciones['Estado_Real'] = df_predicciones['Churn_Real'].map({0: 'Existente', 1: 'Churn'})
df_predicciones['Estado_Predicho'] = df_predicciones['Churn_Predicho'].map({0: 'Existente', 1: 'Churn'})

df_predicciones.to_csv(os.path.join(carpeta_powerbi, 'predicciones_clientes.csv'), index=False)
print(f"  Exportado: predicciones_clientes.csv ({df_predicciones.shape[0]} clientes - DATASET COMPLETO)")

# --- 7.2 Comparativa de modelos ---
print(f"\n--- 7.2 Comparativa de modelos ---")
df_resultados_export = df_resultados.copy()
df_resultados_export.index.name = 'Modelo'
df_resultados_export = df_resultados_export.reset_index()
df_resultados_export['Mejor_Modelo'] = (df_resultados_export['Modelo'] == mejor_nombre).astype(int)
df_resultados_export.to_csv(os.path.join(carpeta_powerbi, 'comparativa_modelos.csv'), index=False)
print(f"  Exportado: comparativa_modelos.csv ({len(df_resultados_export)} modelos)")

# --- 7.3 SHAP values por cliente ---
print(f"\n--- 7.3 SHAP values por cliente ---")
df_shap = pd.DataFrame(shap_values.values, columns=feature_names)
df_shap['SHAP_base_value'] = shap_values.base_values if hasattr(shap_values.base_values, '__len__') else shap_values.base_values
df_shap.to_csv(os.path.join(carpeta_powerbi, 'shap_values_clientes.csv'), index=False)
print(f"  Exportado: shap_values_clientes.csv ({df_shap.shape[0]} clientes x {len(feature_names)} features)")

# --- 7.4 Importancia de features (consolidado) ---
print(f"\n--- 7.4 Importancia de features ---")
shap_importance = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_abs_mean': np.abs(shap_values.values).mean(axis=0)
}).sort_values('SHAP_abs_mean', ascending=False).reset_index(drop=True)

df_importancia = shap_importance.merge(
    perm_imp_df[['Feature', 'Importancia']].rename(columns={'Importancia': 'Permutation_Importance'}),
    on='Feature', how='left'
)
# Agregar feature importance del modelo si existe
if hasattr(mejor_modelo, 'feature_importances_'):
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Model_Feature_Importance': mejor_modelo.feature_importances_
    })
    df_importancia = df_importancia.merge(fi_df, on='Feature', how='left')

df_importancia['Ranking_SHAP'] = range(1, len(df_importancia) + 1)
df_importancia.to_csv(os.path.join(carpeta_powerbi, 'importancia_features.csv'), index=False)
print(f"  Exportado: importancia_features.csv ({len(df_importancia)} features)")

# --- 7.5 Distribución del dataset original ---
print(f"\n--- 7.5 Distribución general ---")
df_distribucion = df_prep.copy()
df_distribucion.to_csv(os.path.join(carpeta_powerbi, 'dataset_procesado.csv'), index=False)
print(f"  Exportado: dataset_procesado.csv ({df_distribucion.shape[0]} filas x {df_distribucion.shape[1]} columnas)")

# --- 7.6 Curva ROC (puntos para graficar en Power BI) ---
print(f"\n--- 7.6 Curvas ROC ---")
roc_data_list = []
for nombre, data in resultados.items():
    fpr, tpr, umbrales = roc_curve(y_test, data['y_proba'])
    temp = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    temp['Modelo'] = nombre
    temp['AUC'] = data['AUC-ROC']
    roc_data_list.append(temp)
df_roc = pd.concat(roc_data_list, ignore_index=True)
df_roc.to_csv(os.path.join(carpeta_powerbi, 'curvas_roc.csv'), index=False)
print(f"  Exportado: curvas_roc.csv ({len(df_roc)} puntos)")

# --- Resumen final ---
print(f"\n" + "=" * 60)
print("EXPORTACIÓN COMPLETADA")
print("=" * 60)
print(f"\nCarpeta: {carpeta_powerbi}")
print(f"\nArchivos generados:")
print(f"  1. predicciones_clientes.csv  -> TODOS los clientes con features originales + predicciones + riesgo")
print(f"  2. comparativa_modelos.csv    -> Métricas de todos los modelos")
print(f"  3. shap_values_clientes.csv   -> Valores SHAP por cliente y feature")
print(f"  4. importancia_features.csv   -> Ranking de importancia (SHAP + Permutation)")
print(f"  5. dataset_procesado.csv      -> Dataset completo procesado")
print(f"  6. curvas_roc.csv             -> Puntos para graficar curvas ROC")


