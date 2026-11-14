"""
App de clasificación de residuos - Datatón 03 (Applied Machine Learning)
Ejecutar con:  streamlit run app.py
Necesita la carpeta modelos/ (exportada desde Colab) al lado de este archivo.
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

st.set_page_config(page_title="Clasificador de Residuos", page_icon="♻️", layout="wide")

# ------------------------------------------------------------
# Recomendaciones de segregación (colores según la NTP 900.058:2019)
# ------------------------------------------------------------
RECOMENDACIONES = {
    "Cartón": ("🟦 Tacho AZUL (papel y cartón)",
               "Aplánalo para que ocupe menos espacio. Debe estar seco y sin restos de comida o grasa."),
    "Papel": ("🟦 Tacho AZUL (papel y cartón)",
              "Que esté limpio y seco. El papel higiénico o las servilletas usadas van al tacho negro."),
    "Plástico": ("⬜ Tacho BLANCO (plásticos)",
                 "Enjuaga la botella o envase, quítale la tapa y aplástalo antes de botarlo."),
    "Vidrio": ("⬛ Tacho PLOMO (vidrio)",
               "Enjuaga el envase y no lo rompas. Los vidrios rotos envuélvelos en papel para que nadie se corte."),
    "Metal": ("🟨 Tacho AMARILLO (metales)",
              "Enjuaga las latas y aplástalas. Si es un objeto grande de metal, llévalo a un punto de acopio."),
    "Basura general": ("⬛ Tacho NEGRO (no aprovechables)",
                       "No se puede reciclar. Ciérralo bien en una bolsa antes de entregarlo al camión."),
}

# ------------------------------------------------------------
# Valores por defecto de la configuración (se guardan en session_state)
# ------------------------------------------------------------
DEFAULTS = {"carpeta": "modelos", "umbral": 0.60, "top_k": 3, "dispositivo": "cpu", "modelo_sel": None}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ------------------------------------------------------------
# Funciones de apoyo
# ------------------------------------------------------------
def leer_fichas(carpeta):
    """Busca todas las subcarpetas que tengan ficha.json y modelo.pt."""
    fichas = {}
    base = Path(carpeta)
    if not base.exists():
        return fichas
    for sub in sorted(base.iterdir()):
        if (sub / "ficha.json").exists() and (sub / "modelo.pt").exists():
            ficha = json.loads((sub / "ficha.json").read_text(encoding="utf-8"))
            ficha["ruta"] = str(sub / "modelo.pt")
            fichas[ficha["nombre"]] = ficha
    return fichas


@st.cache_resource
def cargar_modelo(ruta, dispositivo):
    """Carga el modelo TorchScript una sola vez (queda en caché)."""
    modelo = torch.jit.load(ruta, map_location=dispositivo)
    modelo.eval()
    return modelo


def preprocesar(imagen, ficha):
    """Misma transformación que validación/test en Colab: redimensionar + normalizar."""
    tf = transforms.Compose([
        transforms.Resize((ficha["img_size"], ficha["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ficha["media"], ficha["std"]),
    ])
    return tf(imagen.convert("RGB")).unsqueeze(0)


def predecir(imagen, ficha, dispositivo):
    modelo = cargar_modelo(ficha["ruta"], dispositivo)
    x = preprocesar(imagen, ficha).to(dispositivo)
    inicio = time.time()
    with torch.no_grad():
        probs = torch.softmax(modelo(x), dim=1)[0].cpu()
    ms = (time.time() - inicio) * 1000
    return probs, ms


fichas = leer_fichas(st.session_state.carpeta)
if st.session_state.modelo_sel not in fichas and fichas:
    # por defecto el modelo con mejor Puntaje
    st.session_state.modelo_sel = max(fichas, key=lambda n: fichas[n]["metricas"]["Puntaje"])

# ------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------
st.title("♻️ Clasificador de Residuos")
st.caption("Toma una foto de un residuo y la app te dice qué tipo es y en qué tacho va. "
           "Prototipo para la Municipalidad · Datatón 03")

tab_conf, tab_modelo, tab_clasif, tab_metricas = st.tabs(
    ["⚙️ Configuración", "🧠 Selección de modelo", "📷 Clasificación y resultado", "📊 Métricas"])

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
with tab_conf:
    st.header("⚙️ Configuración")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.carpeta = st.text_input("Carpeta de modelos", st.session_state.carpeta,
                                                 help="Carpeta donde descomprimiste modelos.zip")
        st.session_state.umbral = st.slider("Umbral de confianza", 0.0, 1.0, st.session_state.umbral, 0.05,
                                            help="Si la confianza es menor a este valor, la app pide revisar manualmente")
    with col2:
        st.session_state.top_k = st.slider("Top-k (cuántas opciones mostrar)", 1, 6, st.session_state.top_k)
        opciones_disp = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        st.session_state.dispositivo = st.radio("Dispositivo", opciones_disp, horizontal=True,
                                                index=opciones_disp.index(st.session_state.dispositivo)
                                                if st.session_state.dispositivo in opciones_disp else 0)

    if fichas:
        st.success(f"Se encontraron {len(fichas)} modelos: {', '.join(fichas)}")
    else:
        st.error(f"No se encontraron modelos en la carpeta '{st.session_state.carpeta}'. "
                 "Descomprime modelos.zip al lado de app.py.")

    st.info(f"Configuración actual → umbral: **{st.session_state.umbral:.0%}** · top-k: **{st.session_state.top_k}** · "
            f"dispositivo: **{st.session_state.dispositivo}**")

if not fichas:
    st.stop()

# ============================================================
# 2. SELECCIÓN DE MODELO
# ============================================================
with tab_modelo:
    st.header("🧠 Selección de modelo")
    nombres = list(fichas)
    st.session_state.modelo_sel = st.selectbox("Modelo a usar", nombres,
                                               index=nombres.index(st.session_state.modelo_sel))
    ficha = fichas[st.session_state.modelo_sel]
    m = ficha["metricas"]

    st.subheader(f"Ficha del modelo: {ficha['nombre']}")
    st.write(ficha.get("descripcion", ""))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Puntaje (test)", f"{m['Puntaje']:.2f}")
    c2.metric("Accuracy", f"{m['Accuracy']:.1%}")
    c3.metric("F1 macro", f"{m['F1 macro']:.3f}")
    c4.metric("Parámetros", f"{m['Parámetros'] / 1e6:.2f} M")

    c5, c6, c7 = st.columns(3)
    c5.metric("Familia", m["Familia"])
    c6.metric("Tamaño de imagen", f"{ficha['img_size']} px")
    c7.metric("Entrenamiento", f"{m['Segundos entrenamiento']} s")
    st.caption("Clases: " + ", ".join(ficha["clases"]))

# ============================================================
# 3. CLASIFICACIÓN Y RESULTADO
# ============================================================
with tab_clasif:
    st.header("📷 Clasificación y resultado")
    ficha = fichas[st.session_state.modelo_sel]
    st.caption(f"Modelo en uso: **{ficha['nombre']}** · umbral {st.session_state.umbral:.0%} · top-{st.session_state.top_k}")

    fuente = st.radio("¿Cómo quieres ingresar la foto?", ["Subir una foto", "Tomar foto con la cámara"], horizontal=True)
    archivo = (st.file_uploader("Sube la foto del residuo", type=["jpg", "jpeg", "png"])
               if fuente == "Subir una foto" else st.camera_input("Toma la foto del residuo"))

    if archivo is not None:
        imagen = Image.open(archivo)
        col_img, col_res = st.columns([1, 1.3])
        col_img.image(imagen, caption="Foto ingresada", use_container_width=True)

        # TODO 1: predecir y mostrar el top-k
        probs, ms = predecir(imagen, ficha, st.session_state.dispositivo)
        k = min(st.session_state.top_k, len(ficha["clases"]))
        top_p, top_i = torch.topk(probs, k)
        clase = ficha["clases"][top_i[0].item()]
        confianza = top_p[0].item()

        with col_res:
            # TODO 2: alerta de baja confianza y recomendación de segregación
            if confianza < st.session_state.umbral:
                st.warning(f"⚠️ **El modelo no está seguro** (confianza {confianza:.0%}, menor al umbral de "
                           f"{st.session_state.umbral:.0%}). Toma otra foto con mejor luz y con el residuo solo, "
                           f"o revisa las opciones de abajo antes de botarlo.")
                st.markdown(f"Posible clase: **{clase}**")
            else:
                st.success(f"### Es **{clase}** ({confianza:.0%} de confianza)")
                tacho, consejo = RECOMENDACIONES.get(clase, ("—", ""))
                st.markdown(f"**¿Dónde va?** {tacho}")
                st.markdown(f"**Consejo:** {consejo}")

            st.markdown(f"**Top-{k} del modelo:**")
            top_df = pd.DataFrame({"Clase": [ficha["clases"][i] for i in top_i.tolist()],
                                   "Probabilidad": top_p.tolist()})
            st.dataframe(top_df.style.format({"Probabilidad": "{:.1%}"}), hide_index=True, use_container_width=True)
            st.bar_chart(top_df.set_index("Clase"))
            st.caption(f"Tiempo de predicción: {ms:.0f} ms en {st.session_state.dispositivo}")
    else:
        st.info("Sube o toma una foto para clasificarla.")

# ============================================================
# 4. MÉTRICAS
# ============================================================
with tab_metricas:
    st.header("📊 Métricas de los modelos")

    # TODO 3: tabla comparativa de los modelos
    tabla = pd.DataFrame([f["metricas"] for f in fichas.values()]).sort_values("Puntaje", ascending=False)
    cols = ["Modelo", "Familia", "Img", "Puntaje", "F1 macro", "Accuracy", "Precision macro", "Recall macro",
            "Parámetros", "Entrenables", "Segundos entrenamiento"]
    st.subheader("Comparación de modelos (test)")
    st.dataframe(tabla[[c for c in cols if c in tabla.columns]], hide_index=True, use_container_width=True)
    st.bar_chart(tabla.set_index("Modelo")["Puntaje"])

    elegido = st.selectbox("Ver el detalle de:", list(fichas), index=list(fichas).index(st.session_state.modelo_sel))
    f = fichas[elegido]
    col_cm, col_f1 = st.columns(2)

    # TODO 4: matriz de confusión y F1 por clase
    with col_cm:
        st.subheader("Matriz de confusión (test)")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(f["matriz_confusion"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=f["clases"], yticklabels=f["clases"], ax=ax)
        ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    with col_f1:
        st.subheader("F1 por clase")
        f1 = pd.Series(f["f1_por_clase"]).sort_values()
        st.bar_chart(f1)
        peor = f1.idxmin()
        st.caption(f"La clase más difícil para {elegido} es **{peor}** (F1 = {f1[peor]:.3f}).")

    # TODO 5: curvas de entrenamiento
    st.subheader("Curvas de entrenamiento")
    h = f.get("historial", {})
    if h.get("train_loss"):
        curvas = pd.DataFrame({"Loss train": h["train_loss"], "Loss validación": h["val_loss"]},
                              index=range(1, len(h["train_loss"]) + 1))
        c_loss, c_acc = st.columns(2)
        c_loss.line_chart(curvas)
        c_acc.line_chart(pd.DataFrame({"Accuracy validación": h["val_acc"]}, index=curvas.index))
        st.caption("Eje X: época. Si la loss de validación deja de bajar mientras la de train sigue bajando, hay sobreajuste.")
    else:
        st.info("Este modelo no tiene historial guardado.")
