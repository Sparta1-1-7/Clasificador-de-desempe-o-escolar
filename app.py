import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Predicción de Rendimiento Académico", layout="centered")

st.title("🎓 Predicción de Rendimiento Académico Universitario")
st.subheader("Seminario de Sistemas II - Modelo de Neurona Artificial")

# Funciones de activación
def sigmoide(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def derivada_sigmoide(p):
    return p * (1 - p)

tab1, tab2 = st.tabs(["🏋️ Modo 1: Carga de Dataset y Entrenamiento CSV", "⚙️ Modo 2: Evaluación Manual"])

# ---------------------------------------------------------
# PESTAÑA 1: ENTRENAMIENTO MEDIANTE CSV
# ---------------------------------------------------------
with tab1:
    st.header("📂 1. Carga del Dataset y Entrenamiento Supervisado")
    st.write("Sube un archivo `.csv` con los datos históricos de los estudiantes para entrenar la neurona paso a paso [3, 4].")

    archivo_cargado = st.file_uploader("Selecciona tu archivo CSV (ej. estudiantes.csv)", type=["csv"])

    if archivo_cargado is not None:
        df = pd.read_csv(archivo_cargado)
        st.subheader("📋 Vista previa del Dataset Cargado")
        st.dataframe(df.head(10))

        columnas_entradas = ['Asistencia', 'Actividades', 'Medio_Curso', 'Ordinario', 'Proyecto']
        columna_objetivo = 'Aprobado'

        if all(col in df.columns for col in columnas_entradas + [columna_objetivo]):
            st.success("✅ Estructura del archivo correcta. Listo para entrenar.")
            
            col_c1, col_c2 = st.columns(2)
            epocas = col_c1.slider("Épocas de Entrenamiento", 100, 2000, 500, 100)
            tasa_aprendizaje = col_c2.select_slider("Tasa de Aprendizaje (LR)", [0.01, 0.05, 0.1, 0.5, 1.0], value=0.1)

            if 'W' not in st.session_state:
                st.session_state.W = None
            if 'b' not in st.session_state:
                st.session_state.b = None

            if st.button("🚀 Iniciar Entrenamiento de la Neurona"):
                X = df[columnas_entradas].values
                y = df[[columna_objetivo]].values

                np.random.seed(42)
                W = np.random.randn(5, 1) * 0.1
                b = np.random.randn(1) * 0.1
                historial_error = []

                # Barras de estado visual durante el entrenamiento
                progreso = st.progress(0)
                estado_texto = st.empty()

                for epoca in range(epocas):
                    # Forward propagation
                    z = np.dot(X, W) + b
                    predicciones = sigmoide(z)

                    error = y - predicciones
                    perdida = np.mean(error ** 2)
                    historial_error.append(perdida)

                    # Backpropagation (Ajuste de pesos y bias)
                    d_z = error * derivada_sigmoide(predicciones)
                    W += tasa_aprendizaje * np.dot(X.T, d_z) / len(X)
                    b += tasa_aprendizaje * np.sum(d_z) / len(X)

                    # Actualización visual de la barra de progreso
                    if (epoca + 1) % (epocas // 10) == 0 or epoca == epocas - 1:
                        porcentaje = int(((epoca + 1) / epocas) * 100)
                        progreso.progress(porcentaje)
                        estado_texto.text(f"🧠 Entrenando... Época {epoca + 1}/{epocas} | Pérdida (Error): {perdida:.4f}")
                        time.sleep(0.02) # Pequeña pausa para hacer visible el proceso de carga

                st.session_state.W = W
                st.session_state.b = b
                estado_texto.empty()
                progreso.empty()

                st.success("🎉 ¡Proceso de entrenamiento completado!")
                
                # Evaluación de precisión
                predicciones_finales = (sigmoide(np.dot(X, W) + b) >= 0.5).astype(int)
                precision = np.mean(predicciones_finales == y) * 100
                st.metric("Precisión Alcanzada (Accuracy)", f"{precision:.1f}%")

                st.subheader("📉 Curva de Aprendizaje (Reducción de Error)")
                fig, ax = plt.subplots(figsize=(7, 2.5))
                ax.plot(historial_error, color='#1E88E5', linewidth=2)
                ax.set_xlabel("Épocas")
                ax.set_ylabel("Pérdida (MSE)")
                st.pyplot(fig)

                st.subheader("⚖️ Pesos ($W$) y Sesgo ($b$) Calculados por la Neurona")
                pesos_dict = {
                    "Peso Asistencia (w1)": [round(float(W[0][0]), 4)],
                    "Peso Actividades (w2)": [round(float(W[1][0]), 4)],
                    "Peso Medio Curso (w3)": [round(float(W[2][0]), 4)],
                    "Peso Ordinario (w4)": [round(float(W[3][0]), 4)],
                    "Peso Proyecto (w5)": [round(float(W[4][0]), 4)],
                    "Sesgo / Bias (b)": [round(float(b[0]), 4)]
                }
                st.table(pd.DataFrame(pesos_dict))

        else:
            st.error("El archivo CSV no contiene los nombres de columnas requeridos.")
    else:
        st.info("👆 Por favor sube el archivo `estudiantes.csv` para comenzar.")

    st.markdown("---")
    st.subheader("🔮 2. Probar la Neurona Entrenada con un Estudiante")
    if st.session_state.W is None:
        st.warning("⚠️ Primero debes entrenar la neurona cargando el CSV arriba.")
    else:
        col_a, col_b = st.columns(2)
        x1_a = col_a.slider("Asistencia (0-100%)", 0, 100, 85, key="t1_x1") / 100
        x2_a = col_a.slider("Act. Fundamentales (0-100%)", 0, 100, 90, key="t1_x2") / 100
        x3_a = col_a.slider("Examen Medio Curso (0-100%)", 0, 100, 75, key="t1_x3") / 100
        x4_a = col_b.slider("Examen Ordinario (0-100%)", 0, 100, 70, key="t1_x4") / 100
        x5_a = col_b.slider("Proyecto Integrador (0-100%)", 0, 100, 95, key="t1_x5") / 100

        x_in = np.array([x1_a, x2_a, x3_a, x4_a, x5_a]).reshape(1, -1)
        z_a = np.dot(x_in, st.session_state.W) + st.session_state.b
        prob_a = sigmoide(z_a)[0][0]

        if prob_a >= 0.5:
            st.success(f"🟢 **Resultado Predicho**: APROBADO (Probabilidad: {prob_a*100:.1f}%)")
        else:
            st.error(f"🔴 **Resultado Predicho**: RIESGO DE REPROBACIÓN (Probabilidad: {prob_a*100:.1f}%)")

# ---------------------------------------------------------
# PESTAÑA 2: CONFIGURACIÓN MANUAL
# ---------------------------------------------------------
with tab2:
    st.header("⚙️ Evaluación Manual (Criterio Personalizado)")
    st.write("Permite ajustar ponderaciones manualmente sin necesidad de entrenamiento [2].")

    st.subheader("1. Ponderación de la Materia (Pesos w)")
    col_w1, col_w2 = st.columns(2)
    w1_m = col_w1.number_input("Peso Asistencia (%)", 0, 100, 5, key="m_w1") / 100
    w2_m = col_w1.number_input("Peso Actividades (%)", 0, 100, 30, key="m_w2") / 100
    w3_m = col_w1.number_input("Peso Medio Curso (%)", 0, 100, 25, key="m_w3") / 100
    w4_m = col_w2.number_input("Peso Ordinario (%)", 0, 100, 25, key="m_w4") / 100
    w5_m = col_w2.number_input("Peso Proyecto (%)", 0, 100, 15, key="m_w5") / 100

    st.subheader("2. Calificaciones del Estudiante (Entradas x)")
    col_x1, col_x2 = st.columns(2)
    x1_m = col_x1.slider("Asistencia (%)", 0, 100, 80, key="m_x1") / 100
    x2_m = col_x1.slider("Actividades (%)", 0, 100, 70, key="m_x2") / 100
    x3_m = col_x1.slider("Medio Curso (%)", 0, 100, 60, key="m_x3") / 100
    x4_m = col_x2.slider("Ordinario (%)", 0, 100, 65, key="m_x4") / 100
    x5_m = col_x2.slider("Proyecto (%)", 0, 100, 90, key="m_x5") / 100

    calif_m = (x1_m*w1_m + x2_m*w2_m + x3_m*w3_m + x4_m*w4_m + x5_m*w5_m) * 100
    st.metric("Calificación Ponderada Final", f"{calif_m:.1f} / 100 pts")

    if calif_m >= 70:
        st.success("🟢 **Resultado**: Estudiante Aprobado")
    else:
        st.error("🔴 **Resultado**: Estudiante en Riesgo de Reprobación")