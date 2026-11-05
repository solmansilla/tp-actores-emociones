"""App del TP de IA y Aprendizaje Automático II (UCA Rosario).
Reconoce qué actor aparece en una foto y qué emoción muestra, con la CNN propia y con EfficientNetB0.
Resultados (datos/*.csv, datos/*.json): los del notebook TP_CNN_actor_emocion_v2_FINAL.ipynb.
Redes: las mismas que guardó ese notebook. La app las busca en modelos/ y, si no están, las baja una sola vez
de Google Drive con los links de datos/config.json ("links_modelos").
"""
import base64, html, io, json, os, re, shutil, threading, urllib.request
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
MODELOS, DATOS = os.path.join(BASE, 'modelos'), os.path.join(BASE, 'datos')
INTEGRANTES = ['Juan Ignacio Flores', 'Sol Mansilla', 'María Florencia Romero']
MAX_BYTES = 15 * 1024 * 1024          # foto más grande que esto no se procesa
MAX_PIXELES = 40_000_000               # 40 megapíxeles: más que eso puede tirar la app por memoria

# Paleta. El ámbar es el del recuadro de enfoque de una cámara: se usa solo para marcar caras.
TINTA, TINTA_SUAVE, LINEA, FONDO2 = '#172033', '#556079', '#DDE2E8', '#F1F3F6'
ENFOQUE = '#F0A202'
COLOR = {'cnn_propia': '#2F4BFF', 'efficientnetb0': '#0E9F8A', 'trivial': '#9AA5B8'}
COLOR_EMO = {'alegria': '#E0A526', 'enojo': '#C8553D', 'neutral': '#9AA5B8', 'sorpresa': '#7A5CC4',
             'tristeza': '#3C6E91'}

st.set_page_config(page_title='Actores y Emociones', layout='wide')
# La página ya está en castellano. Streamlit la marca como inglés y el traductor de Chrome la "traduce":
# al cambiarle el texto rompe la página (error "removeChild"). Esto marca la página como castellano y no traducible.
st.html("""<script>
const raiz = document.documentElement;
raiz.setAttribute('lang', 'es');
raiz.setAttribute('translate', 'no');
raiz.classList.add('notranslate');
if (!document.querySelector('meta[name="google"]')) {
  const meta = document.createElement('meta');
  meta.name = 'google'; meta.content = 'notranslate';
  document.head.appendChild(meta);
}
</script>""", unsafe_allow_javascript=True)

st.markdown(f"""
<style>
:root {{
  --tinta: {TINTA}; --tinta-suave: {TINTA_SUAVE}; --linea: {LINEA}; --fondo2: {FONDO2};
  --enfoque: {ENFOQUE}; --cnn: {COLOR['cnn_propia']}; --tl: {COLOR['efficientnetb0']}; --gris: {COLOR['trivial']};
}}
.block-container {{ max-width: 1180px; padding-top: 2.5rem; padding-bottom: 4rem; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ---------- Encabezado: el título dentro de un recuadro de enfoque ---------- */
.visor {{ position: relative; width: fit-content; max-width: 100%; padding: 2.1rem 2.6rem 2rem; margin: 0 0 1.6rem; }}
.visor .esq {{ position: absolute; width: 38px; height: 38px; border: 0 solid var(--enfoque); }}
.visor .a {{ top: 0; left: 0; border-top-width: 4px; border-left-width: 4px; animation: enf-a .7s cubic-bezier(.2,.9,.3,1.2) both; }}
.visor .b {{ top: 0; right: 0; border-top-width: 4px; border-right-width: 4px; animation: enf-b .7s cubic-bezier(.2,.9,.3,1.2) both; }}
.visor .c {{ bottom: 0; left: 0; border-bottom-width: 4px; border-left-width: 4px; animation: enf-c .7s cubic-bezier(.2,.9,.3,1.2) both; }}
.visor .d {{ bottom: 0; right: 0; border-bottom-width: 4px; border-right-width: 4px; animation: enf-d .7s cubic-bezier(.2,.9,.3,1.2) both; }}
@keyframes enf-a {{ from {{ transform: translate(-18px,-18px); opacity: 0; }} }}
@keyframes enf-b {{ from {{ transform: translate(18px,-18px); opacity: 0; }} }}
@keyframes enf-c {{ from {{ transform: translate(-18px,18px); opacity: 0; }} }}
@keyframes enf-d {{ from {{ transform: translate(18px,18px); opacity: 0; }} }}
@media (prefers-reduced-motion: reduce) {{ .visor .esq {{ animation: none; }} }}
.visor .titulo {{ font-family: 'Archivo', sans-serif; font-stretch: 120%; font-weight: 800;
  font-size: clamp(2.3rem, 6.2vw, 4.8rem); line-height: .96; letter-spacing: -0.015em; color: var(--tinta); }}
.visor .bajada {{ font-size: 1.18rem; line-height: 1.5; color: var(--tinta); max-width: 60ch; margin: 1rem 0 0; }}
.visor .firma {{ font-size: .93rem; line-height: 1.5; color: var(--tinta-suave); max-width: 72ch; margin: .8rem 0 0; }}
@media (max-width: 640px) {{ .visor {{ padding: 1.5rem 1.3rem 1.4rem; }} .visor .esq {{ width: 26px; height: 26px; }} }}

/* ---------- Pestañas ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid var(--linea); }}
.stTabs [data-baseweb="tab"] {{ padding: .55rem 0; }}
.stTabs [data-baseweb="tab"] p {{ font-family: 'Archivo', sans-serif; font-size: 1.02rem; font-weight: 600; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.6rem; }}

/* ---------- Texto ---------- */
.lectura {{ max-width: 72ch; font-size: 1.04rem; line-height: 1.6; }}
.lectura p {{ margin: 0 0 .9rem; }}
.chico {{ font-size: .9rem; color: var(--tinta-suave); line-height: 1.5; }}
[data-testid="stMetricValue"] {{ font-family: 'Archivo', sans-serif; font-weight: 700; }}
[data-testid="stMetricLabel"] p {{ color: var(--tinta-suave); }}

/* ---------- Marcador de resultados (Inicio) ---------- */
.marcador {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2.4rem; margin: .4rem 0 .6rem; }}
@media (max-width: 760px) {{ .marcador {{ grid-template-columns: 1fr; gap: 1.6rem; }} }}
.tarea-titulo {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.35rem; color: var(--tinta);
  padding-bottom: .5rem; border-bottom: 2px solid var(--tinta); margin-bottom: .8rem; }}
.tarea-titulo span {{ font-family: 'IBM Plex Sans', sans-serif; font-weight: 400; font-size: .92rem; color: var(--tinta-suave); margin-left: .5rem; }}
.fila {{ display: grid; grid-template-columns: 9.5rem 1fr 3.2rem; align-items: center; gap: .8rem; padding: .42rem 0; }}
.fila .nom {{ font-size: .95rem; color: var(--tinta); }}
.fila .nom.ref {{ color: var(--tinta-suave); }}
.fila .pista {{ height: 12px; background: var(--fondo2); border-radius: 2px; overflow: hidden; }}
.fila .barra {{ height: 100%; border-radius: 2px; }}
.fila .val {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.05rem; text-align: right;
  font-variant-numeric: tabular-nums; color: var(--tinta); }}
.dos-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2.4rem; margin-top: 1.6rem; }}
@media (max-width: 760px) {{ .dos-col {{ grid-template-columns: 1fr; gap: 1.2rem; }} }}
.dos-col h4 {{ font-family: 'Archivo', sans-serif; font-size: 1.1rem; font-weight: 700; margin: 0 0 .35rem; padding: 0; }}
.dos-col p {{ margin: 0; line-height: 1.55; max-width: 60ch; }}

/* ---------- Probar con una foto ---------- */
.veredicto {{ font-size: 1.05rem; margin: 1.6rem 0 .9rem; padding-left: .9rem; border-left: 4px solid var(--enfoque); line-height: 1.45; }}
.respuesta {{ border-top: 4px solid var(--c); padding-top: .9rem; }}
.respuesta .red {{ font-size: .92rem; font-weight: 600; color: var(--c); }}
.respuesta .actor {{ font-family: 'Archivo', sans-serif; font-weight: 800; font-stretch: 110%; font-size: clamp(1.6rem, 3vw, 2.1rem);
  line-height: 1.1; color: var(--tinta); margin: .35rem 0 .15rem; }}
.respuesta .emo {{ font-size: 1.08rem; color: var(--tinta); }}
.respuesta .conf {{ color: var(--tinta-suave); font-variant-numeric: tabular-nums; }}
.respuesta .grupo {{ margin-top: 1rem; }}
.respuesta .grupo-tit {{ font-size: .85rem; color: var(--tinta-suave); margin-bottom: .2rem; }}
.respuesta .fila {{ grid-template-columns: 9rem 1fr 2.9rem; padding: .22rem 0; }}
.respuesta .fila .pista {{ height: 8px; }}
.respuesta .fila .val {{ font-family: 'IBM Plex Sans', sans-serif; font-weight: 500; font-size: .9rem; }}
.respuesta .fila .nom {{ font-size: .9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
</style>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="visor">
  <span class="esq a"></span><span class="esq b"></span><span class="esq c"></span><span class="esq d"></span>
  <div class="titulo">Actores y<br>Emociones</div>
  <p class="bajada">Dos redes convolucionales miran una cara y responden quién es, entre 20 actores y figuras
  argentinas, y qué emoción muestra.</p>
  <p class="firma">{', '.join(INTEGRANTES[:-1])} y {INTEGRANTES[-1]}. Inteligencia Artificial y Aprendizaje
  Automático II, Licenciatura en Ciencia de Datos, UCA Rosario, 2026.</p>
</div>""", unsafe_allow_html=True)


# ============================== Carga de datos y modelos ==============================
def leer_csv(nombre, **kw):
    ruta = os.path.join(DATOS, nombre)
    return pd.read_csv(ruta, **kw) if os.path.exists(ruta) else None


def leer_json(nombre):
    ruta = os.path.join(DATOS, nombre)
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding='utf-8') as f:
        return json.load(f)


CONFIG = leer_json('config.json')
RESUMEN = leer_json('resumen_dataset.json') or {}
METRICAS = leer_csv('metricas_test.csv')


# Nombres con los que el notebook guarda cada red (cualquiera sirve: son la misma red)
ARCHIVOS = {'cnn_propia': ['cnn_propia.keras', 'CNN_PROPIA_FINAL.keras', 'maxpool_flatten_s7.keras'],
            'efficientnetb0': ['efficientnetb0.keras', 'TL_EFFICIENTNETB0_FINAL.keras', 'tl_fase2.keras']}
URL_DRIVE = os.environ.get('URL_DRIVE', 'https://drive.usercontent.google.com/download?id={}&export=download&confirm=t')


def id_drive(link):
    """Saca el id de un link de Google Drive (o acepta el id solo)."""
    link = (link or '').strip()
    m = (re.search(r'/d/([\w-]{20,})', link) or re.search(r'[?&]id=([\w-]{20,})', link)
         or re.fullmatch(r'([\w-]{20,})', link))
    return m.group(1) if m else None


def bajar_de_drive(fid, destino):
    parcial = destino + '.parcial'
    pedido = urllib.request.Request(URL_DRIVE.format(fid), headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(pedido, timeout=300) as r, open(parcial, 'wb') as f:
        shutil.copyfileobj(r, f, 1 << 20)
    with open(parcial, 'rb') as f:
        firma = f.read(2)
    if firma != b'PK':                   # un .keras es un zip; si no, Drive devolvió una página (archivo no público)
        os.remove(parcial)
        raise ValueError('Drive no devolvió el archivo. Revisar que esté compartido como '
                         '"Cualquier persona con el enlace".')
    os.replace(parcial, destino)


@st.cache_resource(show_spinner='Cargando las redes (la primera vez se bajan de Google Drive, tarda un minuto)...')
def _cargar_modelos(links):
    import keras
    modelos, problemas = {}, {}
    for clave, nombres in ARCHIVOS.items():
        ruta = next((os.path.join(MODELOS, n) for n in nombres if os.path.exists(os.path.join(MODELOS, n))), None)
        try:
            if ruta is None:
                fid = id_drive(dict(links).get(clave))
                if fid is None:
                    raise ValueError('no está en modelos/ y falta su link de Google Drive en datos/config.json.')
                os.makedirs(MODELOS, exist_ok=True)
                ruta = os.path.join(MODELOS, nombres[0])
                bajar_de_drive(fid, ruta)
            modelos[clave] = keras.models.load_model(ruta, compile=False)
        except Exception as e:
            problemas[clave] = str(e) or type(e).__name__
    if not modelos:                      # sin ninguna red no se guarda nada en caché: se reintenta en la próxima visita
        raise FileNotFoundError(problemas)
    return modelos, problemas


def cargar_modelos():
    """Devuelve ({clave: red}, {clave: problema}). Las redes se cargan una vez y se comparten entre visitas."""
    links = tuple(sorted((CONFIG or {}).get('links_modelos', {}).items()))
    try:
        return _cargar_modelos(links)
    except FileNotFoundError as e:
        return {}, e.args[0] if e.args and isinstance(e.args[0], dict) else {}


@st.cache_resource(show_spinner=False)
def candado():
    # Las redes y el detector se comparten entre todas las personas que usan la app: de a una por vez
    return threading.Lock()


@st.cache_resource(show_spinner=False)
def cargar_detector():
    import cv2
    xml = 'haarcascade_frontalface_default.xml'
    candidatos = [os.path.join(MODELOS, xml)]
    if hasattr(cv2, 'data'):
        candidatos.insert(0, os.path.join(cv2.data.haarcascades, xml))
    for ruta in candidatos:
        if os.path.exists(ruta):
            det = cv2.CascadeClassifier(ruta)
            if not det.empty():
                return det
    return None


NOMBRE = {'cnn_propia': 'CNN propia', 'efficientnetb0': 'EfficientNetB0 (transferencia)', 'trivial': 'Trivial'}


def lindo(nombre):
    return nombre.replace('_', ' ').title()


# ============================== Recorte de la cara (igual que el notebook) ==============================
def detectar_caras(det, img_rgb):
    import cv2
    gris = cv2.equalizeHist(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
    lado = min(gris.shape[:2])
    caras = det.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=6, minSize=(max(24, lado // 20),) * 2)
    return sorted([tuple(map(int, c)) for c in caras], key=lambda c: -c[2] * c[3])


def recortar_rostro(img_rgb, det, tam, umbral, escala):
    """Misma función que recortar_rostro del notebook: devuelve (imagen tam×tam, caja o None, estado)."""
    import cv2
    h, w = img_rgb.shape[:2]
    caras = detectar_caras(det, img_rgb)
    if not caras:
        lado = min(h, w)
        y0, x0 = (h - lado) // 2, (w - lado) // 2
        recorte, caja, estado = img_rgb[y0:y0 + lado, x0:x0 + lado], None, 'sin_cara'
    else:
        x, y, cw, ch = caras[0]
        estado_extra = 'ambigua' if len(caras) > 1 and caras[1][2] >= 0.75 * cw else None
        if cw / min(h, w) >= umbral and h == w:
            recorte, caja, estado = img_rgb, None, 'ya_recortada'
        else:
            lado = int(min(max(cw * escala, 32), min(h, w)))
            cx, cy = x + cw / 2, y + ch / 2 + 0.05 * ch
            x0 = int(np.clip(cx - lado / 2, 0, max(w - lado, 0)))
            y0 = int(np.clip(cy - lado / 2, 0, max(h - lado, 0)))
            recorte, caja, estado = img_rgb[y0:y0 + lado, x0:x0 + lado], (x0, y0, lado), 'recortada'
        if estado_extra:
            estado = estado_extra
    recorte = cv2.resize(np.ascontiguousarray(recorte), (tam, tam), interpolation=cv2.INTER_AREA)
    return recorte, caja, estado


class ImagenInvalida(Exception):
    pass


def abrir_imagen(datos):
    """Igual que cargar_imagen del notebook: corrige la rotación del celular y pasa a RGB, sin cambiar el tamaño."""
    if len(datos) > MAX_BYTES:
        raise ImagenInvalida('La imagen pesa más de 15 MB.')
    try:
        img = Image.open(io.BytesIO(datos))
    except Exception:
        raise ImagenInvalida('Eso no es una imagen. Si pegaste un link, tiene que ser el de la imagen '
                             '(clic derecho sobre la foto y "Copiar dirección de la imagen"), no el de la página.')
    if img.width * img.height > MAX_PIXELES:
        raise ImagenInvalida(f'La imagen es demasiado grande ({img.width} × {img.height} píxeles). '
                             'Probá con una de menos de 40 megapíxeles.')
    try:
        img = ImageOps.exif_transpose(img)                    # corrige la rotación de las fotos de celular
        if img.mode.startswith('I'):                          # PNG de 16 bits: se lleva a 8 bits
            img = Image.fromarray((np.asarray(img, dtype='float64') / 257).clip(0, 255).astype('uint8'))
        return np.array(img.convert('RGB'))
    except Exception:
        raise ImagenInvalida('No se pudo leer la imagen. Probá guardarla como JPG o PNG.')


@st.cache_data(max_entries=20, show_spinner=False)
def bajar_imagen(url):
    url = url.strip()
    m = re.match(r'^data:image/[\w.+-]+;base64,(.+)$', url, re.S)   # Google a veces copia la imagen como data:
    if m:
        return base64.b64decode(m.group(1))
    if not url.startswith(('http://', 'https://')):
        raise ValueError('El link tiene que empezar con http:// o https://')
    pedido = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(pedido, timeout=20) as r:
        datos = r.read(MAX_BYTES + 1)
    return datos


@st.cache_data(max_entries=20, show_spinner='Analizando la foto...')
def analizar(datos):
    """Recorta la cara y corre las dos redes. Se guarda en caché: cambiar de pestaña no repite el cálculo."""
    original = abrir_imagen(datos)
    with candado():
        recorte, caja, estado = recortar_rostro(original, cargar_detector(), CONFIG['tam_img'],
                                                CONFIG['umbral_cara'], CONFIG['escala_recorte'])
        x = recorte[None].astype('float32')
        probs = {clave: {salida: np.asarray(p[salida][0], dtype='float32') for salida in ['actor', 'emocion']}
                 for clave, p in ((c, m.predict(x, verbose=0)) for c, m in cargar_modelos()[0].items())}
    vista = Image.fromarray(original)                         # copia chica solo para mostrar
    del original
    if caja:                                                  # la cara usada, marcada como un recuadro de enfoque
        x0, y0, lado = caja
        dib, g = ImageDraw.Draw(vista), max(3, min(vista.size) // 120)
        brazo = max(12, lado // 4)
        for (px, py), (dx, dy) in [((x0, y0), (1, 1)), ((x0 + lado, y0), (-1, 1)),
                                   ((x0, y0 + lado), (1, -1)), ((x0 + lado, y0 + lado), (-1, -1))]:
            dib.line([(px, py), (px + dx * brazo, py)], fill=ENFOQUE, width=g)
            dib.line([(px, py), (px, py + dy * brazo)], fill=ENFOQUE, width=g)
    vista.thumbnail((1100, 1100))
    return {'vista': np.array(vista), 'recorte': recorte, 'caja': caja, 'estado': estado, 'probs': probs}


# ============================== Piezas de la interfaz ==============================
def estilo(grafico):
    """Mismo estilo para todos los gráficos de Altair."""
    return (grafico.configure(font='IBM Plex Sans')
            .configure_axis(labelColor=TINTA_SUAVE, titleColor=TINTA_SUAVE, gridColor='#E8ECF1', domainColor='#C9D1DB',
                            tickColor='#C9D1DB', labelFontSize=12, titleFontSize=12, titleFontWeight=500)
            .configure_view(stroke=None)
            .configure_title(font='Archivo', fontSize=15, fontWeight=700, color=TINTA, anchor='start', offset=10)
            .configure_legend(labelColor=TINTA_SUAVE, titleColor=TINTA_SUAVE, labelFontSize=12, titleFontSize=12)
            .configure_header(labelFont='Archivo', labelFontSize=14, labelFontWeight=700, labelColor=TINTA))


def fila(nombre, valor, color, texto=None, clase=''):
    ancho = max(0.0, min(1.0, float(valor))) * 100
    return (f'<div class="fila"><div class="nom {clase}" title="{html.escape(nombre)}">{html.escape(nombre)}</div>'
            f'<div class="pista"><div class="barra" style="width:{ancho:.1f}%;background:{color}"></div></div>'
            f'<div class="val">{texto if texto is not None else f"{valor:.2f}"}</div></div>')


def con_marco(arr, borde=22, brazo=34, grosor=4):
    """El recorte de la cara con las cuatro esquinas de enfoque alrededor."""
    h, w = arr.shape[:2]
    lienzo = Image.new('RGB', (w + 2 * borde, h + 2 * borde), 'white')
    lienzo.paste(Image.fromarray(arr), (borde, borde))
    dib = ImageDraw.Draw(lienzo)
    W, H = lienzo.size
    for (px, py), (dx, dy) in [((0, 0), (1, 1)), ((W - 1, 0), (-1, 1)), ((0, H - 1), (1, -1)), ((W - 1, H - 1), (-1, -1))]:
        dib.line([(px, py), (px + dx * brazo, py)], fill=ENFOQUE, width=grosor)
        dib.line([(px, py), (px, py + dy * brazo)], fill=ENFOQUE, width=grosor)
    return np.array(lienzo)


def miles(n):
    return f'{int(n):,}'.replace(',', '.')


def pct(p):
    return f'{p * 100:.0f} %'


EMO_LINDA = {'alegria': 'Alegría', 'enojo': 'Enojo', 'neutral': 'Neutral', 'sorpresa': 'Sorpresa', 'tristeza': 'Tristeza'}


def nombre_clase(n):
    return EMO_LINDA.get(n, lindo(n))


# ============================== Pestañas ==============================
tabs = st.tabs(['Inicio', 'Probar con una foto', 'Dataset', 'Arquitectura', 'Resultados', 'Limitaciones'])

# ------------------------------ Inicio ------------------------------
with tabs[0]:
    st.markdown("""<div class="lectura">
<p>El dataset lo armaron todos los grupos del curso con cuadros de videos: entrevistas, películas y programas.
Cada cara tiene dos etiquetas, quién es y qué emoción muestra (alegría, enojo, neutral, sorpresa o tristeza).</p>
<p>Se comparan dos redes. La <b>CNN propia</b> se entrenó desde cero, con pesos al azar, y es la que se compara
entre grupos. <b>EfficientNetB0</b> ya venía entrenada con 1,2 millones de fotos de ImageNet y se ajustó a este
problema: sirve de referencia de lo que se puede lograr con mucho más entrenamiento previo.</p></div>""",
                unsafe_allow_html=True)
    if METRICAS is not None:
        st.subheader('Cómo les fue en el test')
        m = METRICAS.set_index('clave')
        n_test = CONFIG.get('n_test') if CONFIG else None
        bloques = []
        for tarea, titulo, detalle in [('actor', 'Actor', '20 personas'), ('emocion', 'Emoción', '5 emociones')]:
            filas = ''.join(fila(etq, m.loc[clave, f'f1_{tarea}'], COLOR[clave], clase='ref' if clave == 'trivial' else '')
                            for clave, etq in [('cnn_propia', 'CNN propia'), ('efficientnetb0', 'EfficientNetB0'),
                                               ('trivial', 'Clase más común')] if clave in m.index)
            bloques.append(f'<div><div class="tarea-titulo">{titulo}<span>{detalle}</span></div>{filas}</div>')
        st.markdown(f'<div class="marcador">{"".join(bloques)}</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="chico">F1 macro sobre {n_test or "las"} imágenes que ninguna red vio al entrenar ni '
                    'al elegir el modelo. Va de 0 a 1 y le da el mismo peso a cada clase. La última fila es lo que saca '
                    'un clasificador que responde siempre la clase más común: el piso a superar.</p>',
                    unsafe_allow_html=True)
    st.markdown("""<div class="dos-col">
<div><h4>Por qué el actor sale bien</h4><p>Depende de rasgos que no cambian: la forma de la cara, los ojos, la nariz.
Con pocas fotos por persona la red ya los distingue.</p></div>
<div><h4>Por qué la emoción cuesta más</h4><p>Depende del gesto, que es sutil y cambia de un cuadro a otro. Además
dos personas pueden etiquetar distinto la misma cara: en casi la mitad de los cuadros seguidos de un video la
etiqueta cambia.</p></div></div>""", unsafe_allow_html=True)

# ------------------------------ Probar con una foto ------------------------------
with tabs[1]:
    st.header('Probar con una foto')
    if CONFIG is None:
        st.error('Falta datos/config.json: subir la carpeta datos/ junto con app.py.')
    else:
        modelos, problemas = cargar_modelos()
        det = cargar_detector()
        for clave, problema in problemas.items():
            st.warning(f'{NOMBRE[clave]}: no se pudo cargar. {problema}')
        if not modelos:
            st.error('No hay ninguna red cargada. Ver el README: hay que pegar en datos/config.json los links de '
                     'Google Drive de las dos redes, o subirlas a la carpeta modelos/.')
        elif det is None:
            st.error('No se pudo cargar el detector de caras de OpenCV.')
        else:
            st.markdown('<div class="lectura"><p>Subí una foto de alguno de los 20 actores. La app busca la cara, la '
                        'recorta igual que en el entrenamiento y muestra qué responde cada red.</p></div>',
                        unsafe_allow_html=True)
            origen = st.segmented_control('De dónde sale la foto', ['Subir un archivo', 'Pegar el link de una imagen',
                                                                     'Sacar una foto con la cámara'],
                                          default='Subir un archivo', label_visibility='collapsed')
            datos = None
            if origen in (None, 'Subir un archivo'):
                archivo = st.file_uploader('Foto (JPG, PNG o WEBP, hasta 15 MB)', type=['jpg', 'jpeg', 'png', 'webp'])
                if archivo is not None:
                    datos = archivo.getvalue()
            elif origen == 'Pegar el link de una imagen':
                url = st.text_input('Link de la imagen', placeholder='https://...',
                                    help='En Google Imágenes: abrir la imagen, clic derecho y "Copiar dirección de la imagen".')
                if url:
                    try:
                        datos = bajar_imagen(url)
                    except Exception as e:
                        st.error(f'No se pudo bajar la imagen de ese link ({type(e).__name__}). '
                                 'Probá guardarla en la compu y subirla como archivo.')
            else:
                foto = st.camera_input('Sacar una foto')
                if foto is not None:
                    datos = foto.getvalue()
            with st.expander('Qué actores conoce la red'):
                st.write(', '.join(lindo(a) for a in CONFIG['actores']) + '.')

            if datos:
                try:
                    r = analizar(datos)
                except ImagenInvalida as e:
                    st.error(str(e)); r = None
                if r is not None:
                    textos = {'sin_cara': 'No se detectó una cara de frente: se usó el centro de la imagen. '
                                          'Probá con una foto más frontal, o recortá la foto alrededor de la cara '
                                          'y volvé a subirla.',
                              'ambigua': 'Hay más de una cara de tamaño parecido en la foto: la red puede estar '
                                         'mirando a otra persona. Si pasa, recortá la foto alrededor de la cara.'}
                    if r['estado'] in textos:
                        st.warning(textos[r['estado']])
                    c1, c2 = st.columns([1.5, 1], gap='large')
                    c1.image(r['vista'], width='stretch',
                             caption='El recuadro marca la cara que se usa.' if r['caja'] else 'Foto subida.')
                    c2.image(con_marco(r['recorte']), width=300,
                             caption='Lo que ve la red: la cara recortada a 224 × 224 píxeles.')

                    gana = {c: CONFIG['actores'][int(p['actor'].argmax())] for c, p in r['probs'].items()}
                    if len(gana) == 2:
                        a, b = gana.values()
                        veredicto = (f'Las dos redes dicen que es <b>{lindo(a)}</b>.' if a == b else
                                     f'Las redes no coinciden: la CNN propia dice <b>{lindo(gana["cnn_propia"])}</b> y '
                                     f'EfficientNetB0 dice <b>{lindo(gana["efficientnetb0"])}</b>.')
                        st.markdown(f'<div class="veredicto">{veredicto}</div>', unsafe_allow_html=True)
                    columnas = st.columns(len(r['probs']), gap='large')
                    for col, (clave, p) in zip(columnas, r['probs'].items()):
                        pa, pe = p['actor'], p['emocion']
                        ia, ie = int(pa.argmax()), int(pe.argmax())
                        top_a = pa.argsort()[::-1][:3]
                        top_e = pe.argsort()[::-1][:3]
                        filas_a = ''.join(fila(lindo(CONFIG['actores'][k]), pa[k], COLOR[clave], pct(pa[k])) for k in top_a)
                        filas_e = ''.join(fila(nombre_clase(CONFIG['emociones'][k]), pe[k], COLOR[clave], pct(pe[k]))
                                          for k in top_e)
                        col.markdown(f"""<div class="respuesta" style="--c:{COLOR[clave]}">
<div class="red">{NOMBRE[clave]}</div>
<div class="actor">{lindo(CONFIG['actores'][ia])}</div>
<div class="conf">{pct(pa[ia])} de probabilidad</div>
<div class="emo" style="margin-top:.5rem">{nombre_clase(CONFIG['emociones'][ie])} <span class="conf">({pct(pe[ie])})</span></div>
<div class="grupo"><div class="grupo-tit">Actores más probables</div>{filas_a}</div>
<div class="grupo"><div class="grupo-tit">Emociones más probables</div>{filas_e}</div>
</div>""", unsafe_allow_html=True)
                    st.markdown('<p class="chico" style="margin-top:1.4rem">La red solo conoce a los 20 actores del '
                                'dataset: con cualquier otra persona igual elige a uno de ellos. La foto no se guarda.</p>',
                                unsafe_allow_html=True)

# ------------------------------ Dataset ------------------------------
with tabs[2]:
    st.header('El dataset')
    st.markdown('<div class="lectura"><p>Por consigna del TP el dataset no se publica fuera del curso: acá solo hay '
                'cantidades, ninguna imagen.</p></div>', unsafe_allow_html=True)
    if RESUMEN:
        cols = st.columns(4)
        cols[0].metric('Imágenes', f"{RESUMEN.get('imagenes', 0):,}".replace(',', '.'))
        cols[1].metric('Actores', RESUMEN.get('actores', '-'))
        cols[2].metric('Emociones', RESUMEN.get('emociones', '-'))
        cols[3].metric('Videos o bloques', RESUMEN.get('grupos_video', '-'))
    conteos = leer_csv('conteos_actor_emocion.csv', index_col=0)
    if conteos is not None:
        st.subheader('Imágenes por actor y emoción')
        largo = conteos.reset_index().melt(id_vars=conteos.index.name or 'actor', var_name='emocion', value_name='imagenes')
        largo.columns = ['actor', 'emocion', 'imagenes']
        largo['actor'] = largo.actor.map(lindo)
        largo['emocion'] = largo.emocion.map(nombre_clase)
        orden = largo.groupby('actor').imagenes.sum().sort_values(ascending=False).index.tolist()
        st.altair_chart(estilo(alt.Chart(largo).mark_bar().encode(
            y=alt.Y('actor:N', sort=orden, title=None, axis=alt.Axis(labelLimit=200)),
            x=alt.X('imagenes:Q', title='imágenes'),
            color=alt.Color('emocion:N', title='Emoción',
                            scale=alt.Scale(domain=[nombre_clase(e) for e in COLOR_EMO], range=list(COLOR_EMO.values()))),
            order=alt.Order('emocion:N'),
            tooltip=['actor', 'emocion', 'imagenes']).properties(height=520)), width='stretch', theme=None)
        tot = conteos.sum().sort_values(ascending=False)
        st.markdown('<p class="chico">Total por emoción: ' + ', '.join(f'{nombre_clase(e).lower()} {n}' for e, n in tot.items()) +
                    '. Hay desbalance: algunos actores casi no tienen fotos de ciertas emociones.</p>', unsafe_allow_html=True)
    part = leer_csv('particion_por_actor.csv', index_col=0)
    if part is not None:
        st.subheader('Partición en train, validación y test')
        tot = part.sum()
        st.markdown(f"""<div class="lectura"><p>Se reparte <b>por video</b>, no por cuadro suelto: dos cuadros seguidos
de un mismo video son casi iguales, y si uno cae en train y otro en test la red reconoce el plano y no la cara. Los
videos largos y las imágenes sin video identificado se cortan en bloques de 10 cuadros seguidos, y cada bloque va
entero a un solo conjunto.</p><p>Quedaron {tot['train']} imágenes de train, {tot['val']} de validación y {tot['test']}
de test ({tot['train'] / tot.sum():.0%}, {tot['val'] / tot.sum():.0%} y {tot['test'] / tot.sum():.0%}).</p></div>""",
                    unsafe_allow_html=True)
        p2 = part.copy(); p2.index = p2.index.map(lindo); p2.index.name = 'Actor'
        p2.columns = ['Train', 'Validación', 'Test']
        st.dataframe(p2, width='stretch')
    rec = leer_csv('recorte_estados.csv')
    if rec is not None:
        st.subheader('Recorte de la cara')
        st.markdown('<div class="lectura"><p>La misma función recorta la cara al entrenar y al predecir. Si la cara '
                    'ocupa poco de la foto (plano abierto), se recorta un cuadrado de 1,6 veces su ancho.</p></div>',
                    unsafe_allow_html=True)
        explicacion = {'ya_recortada': 'Ya venía recortada', 'recortada': 'Plano abierto: se recortó la cara',
                       'sin_cara': 'El detector no encontró cara: se deja igual',
                       'ambigua': 'Dos caras parecidas: se deja igual', 'cara_dudosa': 'Detección dudosa: se deja igual'}
        tabla_rec = pd.DataFrame({'Qué pasó': rec.estado.map(explicacion).fillna(rec.estado), 'Imágenes': rec.imagenes})
        st.dataframe(tabla_rec, hide_index=True, width='content')
    if 'cambio_etiqueta_pct' in RESUMEN:
        st.subheader('Calidad de las etiquetas de emoción')
        st.markdown(f"""<div class="lectura"><p>En el <b>{str(RESUMEN['cambio_etiqueta_pct']).replace('.', ',')} %</b> de
los cuadros, la emoción cambia respecto del cuadro anterior del mismo video. Un gesto dura más que un cuadro, así que
buena parte es desacuerdo al etiquetar. Hay {RESUMEN['videos_sospechosos']} videos o grupos de imágenes
({RESUMEN['imagenes_sospechosas']} imágenes) con 30 % o más de cambios.</p></div>""", unsafe_allow_html=True)

# ------------------------------ Arquitectura ------------------------------
with tabs[3]:
    st.header('Las dos redes')
    gap = CONFIG is not None and 'gap' in CONFIG.get('mejor_cnn', '')
    cierre = ('GlobalAveragePooling\\n7×7×256 → 256' if gap else
              'MaxPooling 2×2 → Flatten → Dropout 0.3\\n7×7×256 → 3×3×256 → 2.304')
    c1, c2 = st.columns([1, 1.15], gap='large')
    with c1:
        st.subheader('CNN propia, desde cero')
        st.graphviz_chart(f"""
digraph G {{
  rankdir=TB; nodesep=0.25; ranksep=0.28; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#C9D1DB", fontname="IBM Plex Sans",
        fontsize=11, fontcolor="{TINTA}", margin="0.18,0.07"];
  edge [color="#9AA5B8", arrowsize=0.6];
  foto [label="Foto de la cara\\n224×224×3", color="{ENFOQUE}", penwidth=2];
  aum [label="Aumentación (solo al entrenar)\\nespejo, rotación, zoom, corrimiento, brillo, contraste"];
  esc [label="Rescaling ÷ 255"];
  b1 [label="Bloque 1: 32 filtros, 112×112"]; b2 [label="Bloque 2: 64 filtros, 56×56"];
  b3 [label="Bloque 3: 128 filtros, 28×28"]; b4 [label="Bloque 4: 192 filtros, 14×14"];
  b5 [label="Bloque 5: 256 filtros, 7×7"];
  cierre [label="{cierre}", fillcolor="#EEF1FF", color="{COLOR['cnn_propia']}", penwidth=2];
  ha [label="Cabeza actor\\nDense 128 → Dropout → softmax 20"];
  he [label="Cabeza emoción\\nDense 128 → Dropout → softmax 5"];
  foto -> aum -> esc -> b1 -> b2 -> b3 -> b4 -> b5 -> cierre; cierre -> ha; cierre -> he;
}}""", width='stretch')
        if CONFIG and CONFIG.get('parametros_cnn'):
            st.markdown(f"<p class='chico'>{miles(CONFIG['parametros_cnn'])} parámetros, todos entrenados desde "
                        "cero.</p>", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="lectura" style="margin-top:3.2rem">
<p><b>Cada bloque</b> tiene dos convoluciones 3×3 con BatchNormalization y ReLU, un atajo residual (se suma la entrada
del bloque, como en ResNet), MaxPooling 2×2 que achica la imagen a la mitad y SpatialDropout. Los filtros crecen
mientras la imagen se achica: hay menos posiciones, pero más tipos de rasgo.</p>
<p><b>El cierre</b> es el que sugirió el profe, MaxPooling seguido de Flatten. Se comparó contra GlobalAveragePooling
con dos semillas cada uno y se eligió con validación. Flatten conserva dónde está cada rasgo: una curva cerca de la
boca no significa lo mismo que cerca de los ojos. GlobalAveragePooling solo dice qué rasgos aparecieron.</p>
<p><b>Dos cabezas.</b> Una sola base convolucional ve la cara y dos salidas deciden cada tarea. La época se eligió con
el promedio del F1 de las dos tareas en validación.</p></div>""", unsafe_allow_html=True)
    st.subheader('Transferencia de aprendizaje con EfficientNetB0')
    st.markdown("""<div class="lectura">
<p><b>Base:</b> EfficientNetB0 entrenada con ImageNet, sin su capa final. Ya sabe detectar bordes, texturas y formas.</p>
<p><b>Fase 1:</b> la base queda congelada y solo se entrenan las dos cabezas nuevas, con tasa de aprendizaje 1e-3.</p>
<p><b>Fase 2, fine-tuning:</b> se descongelan las últimas 60 capas, con las BatchNormalization fijas, y se reentrena con
una tasa 10 veces menor (1e-4) para afinar sin romper lo aprendido.</p></div>""", unsafe_allow_html=True)
    if CONFIG and CONFIG.get('parametros_tl'):
        st.markdown(f"<p class='chico'>{miles(CONFIG['parametros_tl'])} parámetros en total. En la fase 1 se entrenan "
                    f"{miles(331161)} y en la fase 2, {miles(2980441)}.</p>", unsafe_allow_html=True)

# ------------------------------ Resultados ------------------------------
with tabs[4]:
    st.header('Resultados')
    if METRICAS is None:
        st.info('Falta datos/metricas_test.csv: subir la carpeta datos/ junto con app.py.')
    else:
        st.markdown('<div class="lectura"><p>Todo lo de esta sección es sobre el <b>test interno</b>: imágenes que no se '
                    'usaron ni para entrenar ni para elegir el modelo. La métrica principal es el <b>F1 macro</b>, que le '
                    'da el mismo peso a cada clase para que los actores con muchas fotos no tapen a los que tienen pocas.'
                    '</p></div>', unsafe_allow_html=True)
        t = METRICAS.copy()
        t['modelo'] = t.clave.map({'cnn_propia': 'CNN propia', 'efficientnetb0': 'EfficientNetB0',
                                   'trivial': 'Siempre la clase más común'}).fillna(t.modelo)
        tabla = t[['modelo', 'f1_actor', 'acc_actor', 'f1_emocion', 'acc_emocion']].rename(columns={
            'modelo': 'Modelo', 'f1_actor': 'F1 macro actor', 'acc_actor': 'Accuracy actor',
            'f1_emocion': 'F1 macro emoción', 'acc_emocion': 'Accuracy emoción'})
        st.dataframe(tabla.style.format({c: '{:.3f}' for c in tabla.columns[1:]}), hide_index=True, width='stretch')

        st.subheader('Matriz de confusión')
        st.markdown('<div class="lectura"><p>Cada fila es la clase real y cada columna lo que respondió la red; el número '
                    'es el porcentaje de la fila. La diagonal son los aciertos, y lo que queda afuera muestra con quién se '
                    'confunde cada clase.</p></div>', unsafe_allow_html=True)
        disponibles = [c for c in ['cnn_propia', 'efficientnetb0']
                       if all(os.path.exists(os.path.join(DATOS, f'cm_{c}_{t_}.csv')) for t_ in ['actor', 'emocion'])]
        if disponibles:
            c1, c2 = st.columns(2)
            clave = c1.selectbox('Modelo', disponibles, format_func=lambda c: NOMBRE[c])
            tarea = c2.selectbox('Tarea', ['emocion', 'actor'], format_func=lambda x: 'Emoción' if x == 'emocion' else 'Actor')
            cm = leer_csv(f'cm_{clave}_{tarea}.csv', index_col=0)
            n_fila = cm.sum(axis=1)
            pct_cm = cm.div(n_fila.clip(lower=1), axis=0)
            filas_cm = [f'{nombre_clase(r)} (n={int(n_fila[r])})' for r in cm.index]
            largo_cm = pct_cm.set_axis(filas_cm, axis=0).set_axis([nombre_clase(c) for c in cm.columns], axis=1) \
                             .reset_index(names='real').melt(id_vars='real', var_name='predicho', value_name='pct')
            alto = 34 * len(cm) + 20
            oscuro = '#1B2FB8' if clave == 'cnn_propia' else '#0B6F61'
            base = alt.Chart(largo_cm).encode(
                x=alt.X('predicho:N', sort=None, title='predicho', axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
                y=alt.Y('real:N', sort=None, title='real', axis=alt.Axis(labelLimit=260, labelOverlap=False)))
            mapa = base.mark_rect(stroke='#FFFFFF', strokeWidth=1.5).transform_filter('datum.pct > 0').encode(
                color=alt.Color('pct:Q', scale=alt.Scale(domain=[0, 1], range=['#F4F6FA', oscuro]),
                                legend=alt.Legend(format='%', title=None)),
                tooltip=['real', 'predicho', alt.Tooltip('pct:Q', format='.0%')])
            texto = base.mark_text(fontSize=10 if len(cm) > 8 else 13).transform_filter('datum.pct >= 0.01').encode(
                text=alt.Text('pct:Q', format='.0%'),
                color=alt.condition(alt.datum.pct > 0.5, alt.value('white'), alt.value(TINTA)))
            fondo = base.mark_rect(fill='#FFFFFF', stroke='#EEF1F5', strokeWidth=1)
            st.altair_chart(estilo((fondo + mapa + texto).properties(height=alto)), width='stretch',
                            height=alto + 90, theme=None)
            rep = leer_csv(f'reporte_{clave}_{tarea}.csv')
            if rep is not None:
                rep['clase'] = rep.clase.map(nombre_clase)
                rep = rep.rename(columns={'clase': 'Clase', 'precision': 'Precisión', 'recall': 'Recall', 'f1': 'F1',
                                          'imagenes': 'Imágenes'})
                st.dataframe(rep.sort_values('F1', ascending=False).style.format(
                    {'Precisión': '{:.2f}', 'Recall': '{:.2f}', 'F1': '{:.2f}'}), hide_index=True, width='stretch')
                st.markdown('<p class="chico">Precisión: de las veces que dijo esa clase, cuántas acertó. Recall: de '
                            'todas las fotos de esa clase, cuántas encontró. F1: combina las dos.</p>',
                            unsafe_allow_html=True)

        res61 = leer_csv('experimentos_cnn_resumen.csv')
        if res61 is not None:
            st.subheader('Cómo cerrar la CNN propia')
            st.markdown('<div class="lectura"><p>Cada variante se entrenó con 2 semillas; se muestra la media ± el desvío, '
                        'medidos en validación. Una diferencia más chica que el desvío es un empate.</p></div>',
                        unsafe_allow_html=True)
            r = res61.copy()
            nombres_var = {'maxpool_flatten': 'MaxPooling + Flatten (elegida)', 'gap': 'GlobalAveragePooling',
                           'maxpool_flatten_peso_emo': 'MaxPooling + Flatten, emoción con peso 1,5'}
            out = pd.DataFrame({'Variante': r.iloc[:, 0].map(lambda v: nombres_var.get(v, v))})
            for tarea_, etq in [('f1_actor', 'F1 actor'), ('f1_emocion', 'F1 emoción'), ('f1_promedio', 'F1 promedio')]:
                if f'{tarea_}_mean' in r:
                    out[etq] = [f'{a:.3f} ± {b:.3f}' if pd.notna(b) else f'{a:.3f}'
                                for a, b in zip(r[f'{tarea_}_mean'], r.get(f'{tarea_}_std', pd.Series([np.nan] * len(r))))]
            st.dataframe(out, hide_index=True, width='stretch')

        ex62 = leer_csv('experimento_epocas_emocion.csv')
        if ex62 is not None:
            st.subheader('¿A la emoción le faltaban épocas?')
            st.markdown('<div class="lectura"><p>Se siguió entrenando la red elegida hasta 40 épocas más, mirando solo la '
                        'emoción. Es un experimento: no reemplaza al modelo final. Una diferencia menor a 0,03 se toma '
                        'como ruido.</p></div>', unsafe_allow_html=True)
            e = ex62.rename(columns={'experimento': 'Experimento', 'epocas_extra': 'Épocas extra',
                                     'mejor_epoca': 'Mejor época', 'f1_emocion_val': 'F1 emoción val',
                                     'f1_emocion_train': 'F1 emoción train', 'f1_actor_val': 'F1 actor val',
                                     'vs_6_1': 'Diferencia con la red final'})
            e = e.drop(columns=[c for c in ['minutos'] if c in e])
            for c in e.columns[1:]:
                fmt = '{:.0f}' if c in ('Épocas extra', 'Mejor época') else '{:.3f}'
                e[c] = [('-' if pd.isna(v) else fmt.format(v)) for v in e[c]]
            st.dataframe(e, hide_index=True, width='stretch')

        for archivo, titulo in [('curvas_cnn.csv', 'Entrenamiento de la CNN propia'),
                                ('curvas_tl.csv', 'Entrenamiento de EfficientNetB0')]:
            cur = leer_csv(archivo)
            if cur is None:
                continue
            st.subheader(titulo)
            if 'fase' in cur:
                cur = cur.copy()
                cur['epoca'] = np.arange(1, len(cur) + 1)
            partes = []
            for tarea_, etq in [('actor_f1', 'Actor'), ('emocion_f1', 'Emoción')]:
                if tarea_ in cur and f'val_{tarea_}' in cur:
                    partes.append(cur[['epoca', tarea_, f'val_{tarea_}']].rename(
                        columns={tarea_: 'train', f'val_{tarea_}': 'validación'}).melt('epoca', var_name='conjunto',
                                                                                        value_name='F1').assign(tarea=etq))
            if partes:
                d = pd.concat(partes)
                color_val = COLOR['efficientnetb0'] if 'fase' in cur else COLOR['cnn_propia']
                inicio_f2 = int(cur.loc[cur.fase == 2, 'epoca'].min()) if 'fase' in cur and (cur.fase == 2).any() else None
                for col, tarea_ in zip(st.columns(2, gap='large'), d.tarea.unique()):
                    lineas = alt.Chart(d[d.tarea == tarea_]).mark_line(strokeWidth=2).encode(
                        x=alt.X('epoca:Q', title='época'),
                        y=alt.Y('F1:Q', title='F1 macro', scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color('conjunto:N', title=None,
                                        scale=alt.Scale(domain=['train', 'validación'], range=['#B4BDCC', color_val]),
                                        legend=alt.Legend(orient='bottom')),
                        tooltip=['epoca', 'conjunto', alt.Tooltip('F1:Q', format='.3f')])
                    if inicio_f2:
                        lineas = lineas + alt.Chart(pd.DataFrame({'epoca': [inicio_f2 - 0.5]})).mark_rule(
                            strokeDash=[4, 4], color=TINTA_SUAVE).encode(x='epoca:Q')
                    col.altair_chart(estilo(lineas.properties(title=tarea_, height=230)), width='stretch',
                                     height=290, theme=None)
                st.markdown('<p class="chico">El F1 de train se mide con aumentación y dropout activos, por eso puede '
                            'quedar por debajo del de validación.' +
                            (' La línea punteada marca el comienzo del fine-tuning (fase 2).' if inicio_f2 else '') +
                            '</p>', unsafe_allow_html=True)

# ------------------------------ Limitaciones ------------------------------
with tabs[5]:
    st.header('Limitaciones')
    pct_cambio = RESUMEN.get('cambio_etiqueta_pct')
    n_test = CONFIG.get('n_test') if CONFIG else None
    frac_test = (n_test / (CONFIG['n_train'] + CONFIG['n_val'] + n_test)) if n_test and CONFIG.get('n_train') else None
    items = [
        ('Ruido en las etiquetas de emoción',
         (f"La emoción cambia en el {str(pct_cambio).replace('.', ',')} % de los cuadros seguidos de un mismo video. "
          if pct_cambio else '') +
         'Es probablemente el principal techo de la tarea de emoción. El acuerdo entre las personas que etiquetaron no '
         'se midió todavía; ese número sería la cota realista de cualquier modelo.'),
        ('Desbalance', 'Hay actores con mucho más material que otros, y algunos casi no tienen fotos de ciertas emociones. '
         'La red podría aprender el atajo "este actor nunca sonríe". Se compensa en parte con pesos por clase.'),
        ('Test chico', (f'Son {n_test} imágenes, ' if n_test else '') +
         'así que algunos actores tienen pocas fotos de test y una foto más o menos mueve mucho su resultado.'),
        ('Posible fuga entre bloques', 'Los videos largos y las imágenes sin video identificado se cortan en bloques de '
         '10 cuadros, y dos bloques vecinos del mismo video pueden quedar uno en train y otro en test. Eso puede inflar '
         'un poco el resultado de test. Es el precio de un test chico' +
         (f': acá es el {frac_test:.0%} de las imágenes.' if frac_test else '.')),
        ('Un cuadro solo', 'La emoción se decide con una foto suelta, sin el movimiento del gesto. Con varios cuadros '
         'seguidos (una CNN por cuadro y una LSTM encima, como en la Unidad 4) se podría leer cómo cambia la cara.'),
        ('Detector de caras', 'El detector Haar no encuentra caras de perfil o muy de costado; en ese caso se usa el '
         'centro de la imagen.'),
        ('Solo conoce 20 personas', 'Con la foto de cualquier otra persona igual elige a uno de los 20 actores.'),
    ]
    st.markdown('<div class="lectura">' + ''.join(f'<p><b>{t}.</b> {d}</p>' for t, d in items) + '</div>',
                unsafe_allow_html=True)
    if CONFIG:
        st.markdown(f"<p class='chico' style='margin-top:1.6rem'>Resultados de {CONFIG.get('notebook', 'el notebook')}. "
                    f"CNN elegida: {CONFIG.get('mejor_cnn', '-')}. Keras {CONFIG.get('keras', '-')}, TensorFlow "
                    f"{CONFIG.get('tensorflow', '-')}.</p>", unsafe_allow_html=True)
