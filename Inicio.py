import streamlit as st
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import pandas as pd
import numpy as np

# ─── Lexicón de sentimientos en español ───────────────────────
# Palabras positivas y negativas comunes en español
PALABRAS_POSITIVAS = {
    "bueno", "buena", "buenos", "buenas", "excelente", "excelentes",
    "maravilloso", "maravillosa", "increíble", "increíbles", "fantástico",
    "fantástica", "genial", "positivo", "positiva", "feliz", "alegre",
    "contento", "contenta", "satisfecho", "satisfecha", "perfecto", "perfecta",
    "mejor", "mejora", "mejorado", "mejorar", "prometedor", "prometedora",
    "exitoso", "exitosa", "éxito", "logro", "logros", "avance", "avances",
    "innovación", "innovador", "innovadora", "potencial", "oportunidad",
    "oportunidades", "beneficio", "beneficios", "útil", "útiles", "eficiente",
    "eficientes", "efectivo", "efectiva", "poderoso", "poderosa", "capaz",
    "capaces", "brillante", "brillantes", "sobresaliente", "notable",
    "admirable", "valioso", "valiosa", "importante", "fascinante", "fascinantes",
    "interesante", "interesantes", "fácil", "rápido", "rápida", "preciso",
    "precisa", "seguro", "segura", "confiable", "confiables", "robusto",
    "robusta", "gran", "grande", "grandes", "mejor", "óptimo", "óptima",
    "gusta", "encanta", "amo", "amas", "aman", "recomienda", "recomendamos",
    "amor", "pasión", "entusiasmo", "esperanza", "progreso", "crecimiento",
    "transformación", "mejorar", "superar", "ganar", "triunfar", "destacar",
    "liderar", "empoderar", "inspirar"
}

PALABRAS_NEGATIVAS = {
    "malo", "mala", "malos", "malas", "terrible", "terribles", "horrible",
    "horribles", "pésimo", "pésima", "peor", "deficiente", "deficientes",
    "problema", "problemas", "error", "errores", "falla", "fallas", "falló",
    "fracaso", "fracasos", "fracasó", "grave", "graves", "negativo", "negativa",
    "difícil", "difíciles", "imposible", "imposibles", "complicado", "complicada",
    "preocupante", "preocupantes", "peligroso", "peligrosa", "riesgo", "riesgos",
    "desafío", "desafíos", "obstáculo", "obstáculos", "limitación", "limitaciones",
    "lento", "lenta", "ineficiente", "ineficientes", "inútil", "inútiles",
    "obsoleto", "obsoleta", "anticuado", "anticuada", "desactualizado",
    "frustrante", "frustrantes", "decepcionante", "decepcionantes",
    "insatisfecho", "insatisfecha", "insuficiente", "insuficientes",
    "inadecuado", "inadecuada", "incorrecto", "incorrecta", "equivocado",
    "equivocada", "complejo", "compleja", "confuso", "confusa", "molesto",
    "molesta", "triste", "tristeza", "miedo", "temor", "angustia", "ansiedad",
    "desastre", "catástrofe", "crisis", "deterioro", "declive", "caída",
    "pérdida", "perdida", "daño", "daños", "aburrido", "aburrida", "mediocre"
}

# Modificadores de intensidad
INTENSIFICADORES = {"muy", "bastante", "extremadamente", "totalmente", "completamente",
                    "absolutamente", "increíblemente", "enormemente", "sumamente"}
NEGACIONES = {"no", "nunca", "jamás", "tampoco", "ni", "ningún", "ninguna", "sin"}


def analizar_con_lexico(texto: str) -> dict:
    """Análisis de sentimiento usando lexicón español propio — funciona offline."""
    tokens = texto.lower().split()
    score = 0.0
    total_sentiment_words = 0
    negacion_activa = False

    for i, token in enumerate(tokens):
        # Limpiar puntuación del token
        token_limpio = ''.join(c for c in token if c.isalpha() or c == 'é' or c == 'á'
                               or c == 'í' or c == 'ó' or c == 'ú' or c == 'ü' or c == 'ñ')

        if token_limpio in NEGACIONES:
            negacion_activa = True
            continue

        intensificador = 1.0
        if i > 0:
            prev = ''.join(c for c in tokens[i-1] if c.isalpha())
            if prev in INTENSIFICADORES:
                intensificador = 1.5

        if token_limpio in PALABRAS_POSITIVAS:
            delta = 1.0 * intensificador
            score += -delta if negacion_activa else delta
            total_sentiment_words += 1
            negacion_activa = False
        elif token_limpio in PALABRAS_NEGATIVAS:
            delta = -1.0 * intensificador
            score += -delta if negacion_activa else delta
            total_sentiment_words += 1
            negacion_activa = False
        else:
            negacion_activa = False  # la negación caduca si no hay palabra de sentimiento

    # Normalizar entre -1 y 1
    if total_sentiment_words > 0:
        polaridad = max(-1.0, min(1.0, score / (total_sentiment_words * 1.5)))
    else:
        polaridad = 0.0

    # Subjetividad aproximada: proporción de palabras de sentimiento
    num_palabras = len([t for t in tokens if len(t) > 2])
    subjetividad = min(1.0, total_sentiment_words / max(num_palabras, 1) * 3)

    return round(polaridad, 4), round(subjetividad, 4)

# ─── Descarga de recursos NLTK ────────────────────────────────
@st.cache_resource
def descargar_nltk():
    for recurso in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.download(recurso, quiet=True)
        except:
            pass

descargar_nltk()

# ─── Configuración de página ──────────────────────────────────
st.set_page_config(
    page_title="Análisis de Sentimientos",
    page_icon="🧠",
    layout="centered"
)

# ─── Estilos CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fondo */
.stApp {
    background: #0d0d12;
    color: #e8e8f0;
}

/* Header principal */
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.main-header p {
    color: #6b7280;
    font-size: 1rem;
    font-weight: 300;
}

/* Tarjeta de resultado */
.result-card {
    border-radius: 18px;
    padding: 2rem;
    margin: 1.5rem 0;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
}

/* Etiqueta de sentimiento grande */
.sentiment-badge {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    padding: 0.5rem 1.8rem;
    border-radius: 50px;
    margin-bottom: 1rem;
    letter-spacing: -0.5px;
}
.badge-positivo { background: linear-gradient(135deg, #059669, #34d399); color: #fff; }
.badge-negativo { background: linear-gradient(135deg, #dc2626, #f87171); color: #fff; }
.badge-neutro   { background: linear-gradient(135deg, #4b5563, #9ca3af); color: #fff; }

/* Métricas */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.metric-box {
    flex: 1;
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(255,255,255,0.06);
    text-align: center;
}
.metric-box .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6b7280;
    margin-bottom: 0.3rem;
}
.metric-box .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
}

/* Tabla de oraciones */
.oracion-row {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    background: rgba(255,255,255,0.03);
    border-left: 3px solid transparent;
}
.oracion-positivo { border-color: #34d399; }
.oracion-negativo { border-color: #f87171; }
.oracion-neutro   { border-color: #6b7280; }

/* Textarea */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    resize: vertical;
}

/* Botón */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.65rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
}

/* Sección de ejemplos */
.ejemplo-chip {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    cursor: pointer;
    margin: 0.2rem;
    color: #9ca3af;
}

/* Separador */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* Ajuste de sidebar/expander */
.streamlit-expanderHeader {
    font-family: 'Syne', sans-serif;
    color: #a78bfa !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Funciones ────────────────────────────────────────────────

def analizar_sentimiento_oracion(oracion: str) -> dict:
    """
    Análisis de sentimiento multicapa para español:
    1. Intenta traducir con deep-translator → TextBlob en inglés
    2. Si falla, usa lexicón propio en español (siempre disponible offline)
    """
    pol, sub = None, None
    metodo = "lexicón"

    # ── Capa 1: deep-translator + TextBlob ────────────────────
    try:
        from deep_translator import GoogleTranslator
        texto_en = GoogleTranslator(source='es', target='en').translate(oracion)
        if texto_en:
            blob = TextBlob(texto_en)
            pol_tb = blob.sentiment.polarity
            sub_tb = blob.sentiment.subjectivity
            # Solo usar si dio un valor no nulo o el texto original es neutro
            if pol_tb != 0.0 or sub_tb != 0.0:
                pol, sub = pol_tb, sub_tb
                metodo = "traducción"
    except Exception:
        pass

    # ── Capa 2: TextBlob directo en español (backup) ──────────
    if pol is None:
        try:
            blob_es = TextBlob(oracion)
            pol_es = blob_es.sentiment.polarity
            sub_es = blob_es.sentiment.subjectivity
            if pol_es != 0.0:
                pol, sub = pol_es, sub_es
                metodo = "textblob-es"
        except Exception:
            pass

    # ── Capa 3: Lexicón propio en español (siempre funciona) ──
    if pol is None or pol == 0.0:
        pol_lex, sub_lex = analizar_con_lexico(oracion)
        # Combinar si hay señal del lexicón
        if pol_lex != 0.0:
            if pol is not None and pol != 0.0:
                pol = (pol + pol_lex) / 2
                sub = (sub + sub_lex) / 2
            else:
                pol, sub = pol_lex, sub_lex
                metodo = "lexicón"

    pol = pol if pol is not None else 0.0
    sub = sub if sub is not None else 0.0

    if pol > 0.1:
        etiqueta, emoji, css_clase = "Positivo", "🟢", "positivo"
    elif pol < -0.1:
        etiqueta, emoji, css_clase = "Negativo", "🔴", "negativo"
    else:
        etiqueta, emoji, css_clase = "Neutro", "⚪", "neutro"

    return {
        "polaridad": round(pol, 4),
        "subjetividad": round(sub, 4),
        "etiqueta": etiqueta,
        "emoji": emoji,
        "css": css_clase,
        "metodo": metodo
    }


def obtener_tokens_limpios(texto: str) -> list:
    stop_es = set(stopwords.words('spanish'))
    tokens = word_tokenize(texto, language='spanish')
    return [
        t.lower() for t in tokens
        if t.lower() not in stop_es and t.isalpha() and len(t) > 2
    ]


def gauge_chart(polaridad: float) -> plt.Figure:
    """Genera un gauge semicircular de polaridad."""
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw={'aspect': 'equal'})
    fig.patch.set_facecolor('#0d0d12')
    ax.set_facecolor('#0d0d12')

    # Arco de fondo (gris)
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='#1f2937', linewidth=18, solid_capstyle='round')

    # Color según polaridad
    if polaridad > 0.1:
        color = '#34d399'
    elif polaridad < -0.1:
        color = '#f87171'
    else:
        color = '#9ca3af'

    # Arco coloreado (de π hasta el ángulo correspondiente)
    angulo_fin = np.pi - (polaridad + 1) / 2 * np.pi
    theta_fill = np.linspace(np.pi, angulo_fin, 200)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill),
            color=color, linewidth=18, solid_capstyle='round', alpha=0.9)

    # Aguja
    angulo_aguja = angulo_fin
    ax.plot([0, 0.6 * np.cos(angulo_aguja)],
            [0, 0.6 * np.sin(angulo_aguja)],
            color='white', linewidth=3, zorder=5)
    ax.add_patch(plt.Circle((0, 0), 0.06, color='white', zorder=6))

    # Etiquetas
    ax.text(-1.0, -0.22, 'Negativo', color='#f87171', fontsize=8, ha='center',
            fontfamily='monospace')
    ax.text(0, -0.22, 'Neutro', color='#9ca3af', fontsize=8, ha='center',
            fontfamily='monospace')
    ax.text(1.0, -0.22, 'Positivo', color='#34d399', fontsize=8, ha='center',
            fontfamily='monospace')

    # Valor central
    ax.text(0, 0.25, f'{polaridad:+.3f}', color='white',
            fontsize=16, ha='center', va='center', fontweight='bold',
            fontfamily='monospace')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.2)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


def barras_oraciones(oraciones_datos: list) -> plt.Figure:
    """Gráfica de barras de polaridad por oración."""
    fig, ax = plt.subplots(figsize=(8, max(3, len(oraciones_datos) * 0.7)))
    fig.patch.set_facecolor('#0d0d12')
    ax.set_facecolor('#0d0d12')

    etiquetas = [f"Oración {d['num']}" for d in oraciones_datos]
    valores   = [d['polaridad'] for d in oraciones_datos]
    colores   = ['#34d399' if v > 0.1 else '#f87171' if v < -0.1 else '#6b7280'
                 for v in valores]

    bars = ax.barh(etiquetas[::-1], valores[::-1], color=colores[::-1],
                   height=0.55, edgecolor='none')
    ax.axvline(0, color=(1, 1, 1, 0.15), linewidth=1)
    ax.axvline(0.1, color='#34d399', linewidth=0.7, linestyle='--', alpha=0.4)
    ax.axvline(-0.1, color='#f87171', linewidth=0.7, linestyle='--', alpha=0.4)

    for bar, val in zip(bars, valores[::-1]):
        ax.text(val + (0.02 if val >= 0 else -0.02), bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}', va='center',
                ha='left' if val >= 0 else 'right',
                color='white', fontsize=8, fontfamily='monospace')

    ax.set_xlabel('Polaridad', color='#6b7280', fontsize=9)
    ax.set_xlim(-1.15, 1.15)
    ax.tick_params(colors='#6b7280', labelsize=9)
    ax.spines[:].set_visible(False)
    ax.grid(axis='x', color=(1, 1, 1, 0.06), linewidth=0.5)
    plt.tight_layout()
    return fig


# ─── INTERFAZ ─────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧠 Análisis de Sentimientos</h1>
    <p>Detecta la polaridad emocional de textos en español · NLP con NLTK & TextBlob</p>
</div>
""", unsafe_allow_html=True)

# Ejemplos rápidos
EJEMPLOS = {
    "😊 Positivo": "La inteligencia artificial es una tecnología maravillosa que está transformando el mundo de manera increíble y prometedora.",
    "😞 Negativo": "Este sistema tiene errores graves y un rendimiento terrible. La implementación fue un fracaso total, muy frustrante.",
    "😐 Neutro": "El modelo procesa los datos de entrada y genera una salida numérica que representa la probabilidad de cada clase.",
    "📰 Mixto": "Colombia tiene un gran potencial tecnológico. Sin embargo, aún existen desafíos importantes en infraestructura y acceso a educación de calidad.",
}

st.markdown("**Ejemplos rápidos:**")
cols = st.columns(len(EJEMPLOS))
texto_ejemplo = None
for i, (label, txt) in enumerate(EJEMPLOS.items()):
    if cols[i].button(label, key=f"ej_{i}"):
        texto_ejemplo = txt

# Área de texto
valor_inicial = texto_ejemplo if texto_ejemplo else ""
texto_usuario = st.text_area(
    "Escribe o pega tu texto en español:",
    value=valor_inicial,
    height=150,
    placeholder="Ej: La educación en Colombia ha mejorado significativamente...",
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)
analizar = st.button("✨  Analizar sentimiento")

# ─── RESULTADO ────────────────────────────────────────────────
if analizar and texto_usuario.strip():

    with st.spinner("Analizando..."):

        # Análisis global
        res = analizar_sentimiento_oracion(texto_usuario)
        pol = res["polaridad"]
        sub = res["subjetividad"]

        # Tokens para estadísticas
        try:
            tokens_limpios = obtener_tokens_limpios(texto_usuario)
            oraciones = sent_tokenize(texto_usuario, language='spanish')
        except:
            tokens_limpios = []
            oraciones = [texto_usuario]

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Resultado principal ──────────────────────────────────
    badge_class = f"badge-{res['css']}"
    label_sub = "Objetivo" if sub < 0.35 else "Subjetivo" if sub > 0.65 else "Mixto"
    metodo_icons = {"traducción": "🌐 Traducción → TextBlob", "textblob-es": "📝 TextBlob español", "lexicón": "📖 Lexicón español"}
    metodo_label = metodo_icons.get(res.get("metodo", "lexicón"), "📖 Lexicón español")

    st.markdown(f"""
    <div class="result-card">
        <div style="text-align:center; margin-bottom:1rem;">
            <span class="sentiment-badge {badge_class}">{res['emoji']} {res['etiqueta']}</span>
            <div style="margin-top:0.5rem;font-size:0.75rem;color:#4b5563">
                Método: <span style="color:#a78bfa">{metodo_label}</span>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-box">
                <div class="label">Polaridad</div>
                <div class="value" style="color:{'#34d399' if pol>0.1 else '#f87171' if pol<-0.1 else '#9ca3af'}">
                    {pol:+.3f}
                </div>
            </div>
            <div class="metric-box">
                <div class="label">Subjetividad</div>
                <div class="value" style="color:#60a5fa">{sub:.3f}</div>
            </div>
            <div class="metric-box">
                <div class="label">Tono</div>
                <div class="value" style="font-size:1.1rem;padding-top:0.4rem;color:#e8e8f0">{label_sub}</div>
            </div>
            <div class="metric-box">
                <div class="label">Oraciones</div>
                <div class="value" style="color:#a78bfa">{len(oraciones)}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Gauge ───────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig_gauge = gauge_chart(pol)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

    # ── Análisis por oración ─────────────────────────────────
    if len(oraciones) > 1:
        st.markdown("### 📋 Análisis por oración")

        datos_oraciones = []
        for i, oracion in enumerate(oraciones, 1):
            r = analizar_sentimiento_oracion(oracion)
            datos_oraciones.append({**{'num': i, 'texto': oracion}, **r})

            st.markdown(f"""
            <div class="oracion-row oracion-{r['css']}">
                <span style="font-size:1.2rem;min-width:24px">{r['emoji']}</span>
                <div>
                    <div style="font-size:0.9rem;color:#e8e8f0;line-height:1.5">{oracion}</div>
                    <div style="font-size:0.75rem;color:#6b7280;margin-top:0.3rem">
                        Polaridad: <b style="color:{'#34d399' if r['polaridad']>0.1 else '#f87171' if r['polaridad']<-0.1 else '#9ca3af'}">{r['polaridad']:+.3f}</b>
                        &nbsp;|&nbsp; Subjetividad: <b style="color:#60a5fa">{r['subjetividad']:.3f}</b>
                        &nbsp;|&nbsp; <b>{r['etiqueta']}</b>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Gráfica de barras
        st.markdown("<br>", unsafe_allow_html=True)
        fig_bar = barras_oraciones(datos_oraciones)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    # ── Palabras más frecuentes ──────────────────────────────
    if tokens_limpios:
        with st.expander("📊 Palabras más frecuentes"):
            fdist = FreqDist(tokens_limpios)
            top_words = fdist.most_common(10)

            fig_freq, ax = plt.subplots(figsize=(7, 3.5))
            fig_freq.patch.set_facecolor('#0d0d12')
            ax.set_facecolor('#0d0d12')

            palabras = [w for w, _ in top_words]
            freqs    = [f for _, f in top_words]
            colores_freq = plt.cm.cool(np.linspace(0.2, 0.9, len(palabras)))

            ax.barh(palabras[::-1], freqs[::-1], color=colores_freq[::-1],
                    height=0.6, edgecolor='none')
            ax.set_xlabel('Frecuencia', color='#6b7280', fontsize=9)
            ax.tick_params(colors='#6b7280', labelsize=9)
            ax.spines[:].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_freq, use_container_width=True)
            plt.close(fig_freq)

    # ── Exportar resultados ──────────────────────────────────
    with st.expander("📥 Exportar resultados"):
        datos_export = {
            "texto": texto_usuario,
            "polaridad_global": pol,
            "subjetividad_global": sub,
            "etiqueta": res['etiqueta'],
            "num_oraciones": len(oraciones),
            "num_tokens_limpios": len(tokens_limpios)
        }
        df_export = pd.DataFrame([datos_export])
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Descargar CSV",
            data=csv,
            file_name="resultado_sentimiento.csv",
            mime="text/csv"
        )

elif analizar and not texto_usuario.strip():
    st.warning("⚠️ Por favor escribe o pega un texto antes de analizar.")

# ─── Footer ───────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#374151;font-size:0.78rem;font-family:'DM Sans',sans-serif">
    Análisis de Sentimientos · NLTK + TextBlob · Universidad EAFIT<br>
    <span style="color:#1f2937">Polaridad: -1.0 (muy negativo) → 0 (neutro) → +1.0 (muy positivo)</span>
</div>
""", unsafe_allow_html=True)
