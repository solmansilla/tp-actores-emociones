"""App del TP de IA y Aprendizaje Automático II (UCA Rosario).
Reconoce qué actor aparece en una foto y qué emoción muestra, con la CNN propia y con EfficientNetB0.
Resultados (datos/*.csv, datos/*.json): los del notebook TP_CNN_actor_emocion_v2_FINAL.ipynb.
Redes: las mismas que guardó ese notebook. La app las busca en modelos/ y, si no están, las baja una sola vez
de Google Drive con los links de datos/config.json ("links_modelos").
"""
import base64, io, json, os, re, shutil, threading, urllib.request
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
MODELOS, DATOS = os.path.join(BASE, 'modelos'), os.path.join(BASE, 'datos')
INTEGRANTES = ['Juan Ignacio Flores', 'Sol Mansilla', 'María Florencia Romero']
COLOR = {'cnn_propia': '#E8603C', 'efficientnetb0': '#2B6CB0', 'trivial': '#A0AEC0'}
MAX_BYTES = 15 * 1024 * 1024          # foto más grande que esto no se procesa
ALTO_TOP = 150                         # alto de los gráficos de top 3
MAX_PIXELES = 40_000_000               # 40 megapíxeles: más que eso puede tirar la app por memoria

st.set_page_config(page_title='Actores y Emociones', layout='wide')
st.markdown("""
<style>
.titulo {font-family: Georgia, 'Times New Roman', serif; font-size: 4.2rem; font-weight: 700;
         color: #E8603C; line-height: 1.05; margin: 0.5rem 0 0.2rem 0;}
.subtitulo {font-size: 1.9rem; font-weight: 700; color: #2D3748; margin: 0.4rem 0 1.2rem 0;}
.tarjeta {background: #EBF4FF; border-radius: 8px; padding: 0.9rem 1.1rem; height: 100%;}
.tarjeta h4 {margin: 0 0 0.4rem 0; color: #1A365D;}
.nota {color: #4A5568; font-size: 0.92rem;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="titulo">Actores y Emociones</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Redes convolucionales para reconocer quién es y qué siente</div>',
            unsafe_allow_html=True)


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
    if caja:
        x0, y0, lado = caja
        ImageDraw.Draw(vista).rectangle([x0, y0, x0 + lado, y0 + lado], outline='#E8603C',
                                        width=max(3, vista.width // 150))
    vista.thumbnail((1000, 1000))
    return {'vista': np.array(vista), 'recorte': recorte, 'caja': caja, 'estado': estado, 'probs': probs}


def grafico_top(nombres, probs, color, titulo):
    d = pd.DataFrame({'clase': [lindo(n) for n in nombres], 'probabilidad': probs})
    base = alt.Chart(d).encode(
        y=alt.Y('clase:N', sort='-x', title=None, axis=alt.Axis(labelLimit=200, labelFontSize=12)),
        x=alt.X('probabilidad:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%'), title=None))
    barras = base.mark_bar(color=color, cornerRadiusEnd=3)
    texto = base.mark_text(align='left', dx=4).encode(text=alt.Text('probabilidad:Q', format='.0%'))
    return (barras + texto).properties(title=titulo, height=ALTO_TOP)


# ============================== Pestañas ==============================
tabs = st.tabs(['Inicio', 'Probar con una foto', 'Dataset', 'Arquitectura', 'Resultados', 'Limitaciones'])

# ------------------------------ Inicio ------------------------------
with tabs[0]:
    st.header('Quién es y qué siente')
    st.write(
        'Este trabajo entrena redes neuronales para mirar una foto de una cara y responder dos preguntas: '
        '**qué actor es** (entre 20 actores y figuras argentinas) y **qué emoción muestra** '
        '(alegría, enojo, neutral, sorpresa o tristeza). El dataset lo armaron todos los grupos del curso '
        'a partir de cuadros de videos, y cada cara tiene dos etiquetas: identidad y emoción.')
    st.write(
        'Se comparan dos redes convolucionales. La **CNN propia** se entrenó desde cero, con pesos al azar, '
        'y es la que se compara entre grupos. La red de **transferencia de aprendizaje** parte de EfficientNetB0, '
        'ya entrenada con 1,2 millones de fotos de ImageNet, y sirve de referencia.')
    if METRICAS is not None:
        st.subheader('Resultados en el test interno')
        m = METRICAS.set_index('clave')
        cols = st.columns(4)
        for col, (clave, tarea, etiqueta) in zip(cols, [('cnn_propia', 'actor', 'CNN propia · actor'),
                                                        ('cnn_propia', 'emocion', 'CNN propia · emoción'),
                                                        ('efficientnetb0', 'actor', 'EfficientNetB0 · actor'),
                                                        ('efficientnetb0', 'emocion', 'EfficientNetB0 · emoción')]):
            if clave in m.index:
                col.metric(f'F1 macro {etiqueta}', f"{m.loc[clave, f'f1_{tarea}']:.2f}",
                           help=f"Accuracy {m.loc[clave, f'acc_{tarea}']:.0%}")
        if 'trivial' in m.index:
            st.caption(f"Como referencia, un clasificador que siempre responde la clase más común saca "
                       f"{m.loc['trivial', 'f1_actor']:.3f} en actor y {m.loc['trivial', 'f1_emocion']:.3f} en emoción.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="tarjeta"><h4>Identidad</h4>Depende de rasgos que no cambian: la forma de la cara, '
                    'los ojos, la nariz. Es la tarea más fácil y la red la aprende bien.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="tarjeta"><h4>Emoción</h4>Depende del gesto: boca, cejas y ojos. Es sutil, cambia '
                    'de un cuadro a otro y dos personas pueden etiquetar distinto la misma cara.</div>',
                    unsafe_allow_html=True)
    st.subheader('Equipo')
    st.write(' · '.join(INTEGRANTES))
    st.caption('Inteligencia Artificial y Aprendizaje Automático II · Licenciatura en Ciencia de Datos · UCA Rosario · 2026')

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
            st.write('Subí una foto de alguno de los actores del dataset. La app busca la cara, la recorta igual que '
                     'en el entrenamiento y muestra qué responde cada red.')
            with st.expander('Actores que la red conoce'):
                st.write(', '.join(lindo(a) for a in CONFIG['actores']))
            origen = st.radio('De dónde sale la foto', ['Subir un archivo', 'Pegar el link de una imagen',
                                                         'Sacar una foto con la cámara'], horizontal=True)
            datos = None
            if origen == 'Subir un archivo':
                archivo = st.file_uploader('Foto (JPG, PNG o WEBP)', type=['jpg', 'jpeg', 'png', 'webp'])
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

            if datos:
                try:
                    r = analizar(datos)
                except ImagenInvalida as e:
                    st.error(str(e)); r = None
                if r is not None:
                    c1, c2 = st.columns([1.4, 1])
                    ancho = r['vista'].shape[1]
                    c1.image(r['vista'], caption='Foto original' + (' (recuadro: la cara que se usa)' if r['caja'] else ''),
                             width=max(360, min(ancho, 560)))
                    c2.image(r['recorte'], caption='Lo que ve la red (224 × 224)', width=260)
                    textos = {'sin_cara': 'No se detectó una cara de frente: se usó el centro de la imagen. '
                                          'Probá con una foto más frontal, o recortá la foto alrededor de la cara '
                                          'y volvé a subirla.',
                              'ambigua': 'Hay más de una cara de tamaño parecido en la foto: la red puede estar '
                                         'mirando a otra persona. Si pasa, recortá la foto alrededor de la cara.'}
                    if r['estado'] in textos:
                        st.warning(textos[r['estado']])
                    columnas = st.columns(len(r['probs']))
                    for col, (clave, p) in zip(columnas, r['probs'].items()):
                        with col:
                            st.subheader(NOMBRE[clave])
                            for salida, nombres in [('actor', CONFIG['actores']), ('emocion', CONFIG['emociones'])]:
                                probs = p[salida]
                                orden = probs.argsort()[::-1][:3]
                                st.altair_chart(grafico_top([nombres[k] for k in orden], probs[orden], COLOR[clave],
                                                            'Actor: top 3' if salida == 'actor' else 'Emoción: top 3'),
                                                width='stretch', height=ALTO_TOP + 40)
                    st.caption('La red solo conoce a los 20 actores del dataset: con cualquier otra persona igual va a '
                               'elegir a uno de ellos. La foto no se guarda.')

# ------------------------------ Dataset ------------------------------
with tabs[2]:
    st.header('El dataset')
    st.write('Lo armaron todos los grupos del curso a partir de videos (entrevistas, películas, programas). '
             'Por consigna del TP el dataset no se publica fuera del curso, así que acá solo se muestran cantidades, '
             'sin ninguna imagen.')
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
        orden = largo.groupby('actor').imagenes.sum().sort_values(ascending=False).index.tolist()
        st.altair_chart(alt.Chart(largo).mark_bar().encode(
            y=alt.Y('actor:N', sort=orden, title=None),
            x=alt.X('imagenes:Q', title='imágenes'),
            color=alt.Color('emocion:N', title='Emoción', scale=alt.Scale(scheme='tableau10')),
            tooltip=['actor', 'emocion', 'imagenes']).properties(height=520), width='stretch')
        tot = conteos.sum().sort_values(ascending=False)
        st.caption('Total por emoción: ' + ' · '.join(f'{lindo(e)} {n}' for e, n in tot.items()) +
                   '. Hay desbalance: algunos actores casi no tienen fotos de ciertas emociones.')
    part = leer_csv('particion_por_actor.csv', index_col=0)
    if part is not None:
        st.subheader('Partición train / validación / test')
        tot = part.sum()
        st.write(f"Se reparte **por video**, no por cuadro suelto: dos cuadros seguidos de un mismo video son casi "
                 f"iguales, y si uno cae en train y otro en test la red reconoce el plano y no la cara. Los videos "
                 f"largos y las imágenes sin video identificado se cortan en bloques de 10 cuadros seguidos, y cada "
                 f"bloque va entero a un solo conjunto. Quedó train {tot['train']}, validación {tot['val']} y test {tot['test']} imágenes "
                 f"({tot['train'] / tot.sum():.0%} / {tot['val'] / tot.sum():.0%} / {tot['test'] / tot.sum():.0%}).")
        p2 = part.copy(); p2.index = p2.index.map(lindo); p2.columns = ['train', 'validación', 'test']
        st.dataframe(p2, width='stretch')
    rec = leer_csv('recorte_estados.csv')
    if rec is not None:
        st.subheader('Recorte de la cara')
        st.write('La misma función recorta la cara al entrenar y al predecir. Si la cara ocupa poco de la foto '
                 '(plano abierto), se recorta un cuadrado de 1,6 veces su ancho.')
        explicacion = {'ya_recortada': 'ya venía recortada', 'recortada': 'plano abierto, se recortó',
                       'sin_cara': 'el detector no encontró cara (se deja igual)',
                       'ambigua': 'dos caras parecidas (se deja igual)', 'cara_dudosa': 'detección dudosa (se deja igual)'}
        rec['qué significa'] = rec.estado.map(explicacion).fillna('')
        st.dataframe(rec, hide_index=True, width='content')
    if 'cambio_etiqueta_pct' in RESUMEN:
        st.subheader('Calidad de las etiquetas de emoción')
        st.write(f"En el **{RESUMEN['cambio_etiqueta_pct']}%** de los cuadros, la emoción cambia respecto del cuadro "
                 f"anterior del mismo video. Un gesto dura más que un cuadro, así que buena parte es desacuerdo al "
                 f"etiquetar. Hay {RESUMEN['videos_sospechosos']} videos o grupos de imágenes "
                 f"({RESUMEN['imagenes_sospechosas']} imágenes) con 30 % o más de cambios.")

# ------------------------------ Arquitectura ------------------------------
with tabs[3]:
    st.header('Las dos redes')
    gap = CONFIG is not None and 'gap' in CONFIG.get('mejor_cnn', '')
    cierre = ('GlobalAveragePooling\\n7×7×256 → 256' if gap else
              'MaxPooling 2×2 → Flatten → Dropout 0.3\\n7×7×256 → 3×3×256 → 2.304')
    st.subheader('CNN propia, desde cero')
    st.graphviz_chart(f"""
digraph G {{
  rankdir=TB; nodesep=0.25; ranksep=0.3;
  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#A0AEC0", fontname="Helvetica", fontsize=11];
  edge [color="#718096", arrowsize=0.7];
  foto [label="Foto de la cara\\n224×224×3"];
  aum [label="Aumentación (solo al entrenar)\\nespejo, rotación, zoom, corrimiento, brillo, contraste"];
  esc [label="Rescaling ÷ 255"];
  b1 [label="Bloque 1 · 32 filtros\\n112×112"]; b2 [label="Bloque 2 · 64 filtros\\n56×56"];
  b3 [label="Bloque 3 · 128 filtros\\n28×28"]; b4 [label="Bloque 4 · 192 filtros\\n14×14"];
  b5 [label="Bloque 5 · 256 filtros\\n7×7"];
  cierre [label="{cierre}", fillcolor="#FDEBE5", color="#E8603C", penwidth=2];
  ha [label="Cabeza actor\\nDense 128 → Dropout → softmax 20"];
  he [label="Cabeza emoción\\nDense 128 → Dropout → softmax 5"];
  {{rank=same; b1; b2; b3; b4; b5;}}
  foto -> aum -> esc -> b1; b1 -> b2 -> b3 -> b4 -> b5; b5 -> cierre; cierre -> ha; cierre -> he;
}}""")
    st.markdown('**Cada bloque:** dos convoluciones 3×3 con BatchNormalization y ReLU, un **atajo residual** '
                '(se suma la entrada del bloque, como en ResNet), MaxPooling 2×2 que achica la imagen a la mitad y '
                'SpatialDropout. Los filtros crecen mientras la imagen se achica: hay menos posiciones pero más tipos de rasgo.')
    st.markdown('**El cierre:** el profe sugirió MaxPooling seguido de Flatten. Se comparó contra GlobalAveragePooling '
                'con dos semillas cada uno y se eligió con validación. Flatten conserva **dónde** está cada rasgo '
                '(una curva cerca de la boca no significa lo mismo que cerca de los ojos); GlobalAveragePooling '
                'solo dice qué rasgos aparecieron. ' +
                ('Ganó GlobalAveragePooling.' if gap else 'Ganó MaxPooling + Flatten.'))
    st.markdown('**Dos cabezas:** una sola base convolucional ve la cara y dos salidas deciden cada tarea. La época '
                'se eligió con el promedio del F1 de las dos tareas en validación.')
    if CONFIG and CONFIG.get('parametros_cnn'):
        st.caption(f"Parámetros de la CNN propia: {CONFIG['parametros_cnn']:,}".replace(',', '.'))
    st.subheader('Transferencia de aprendizaje: EfficientNetB0')
    st.markdown(
        '- **Base:** EfficientNetB0 entrenada con ImageNet, sin su capa final. Ya sabe detectar bordes, texturas y formas.\n'
        '- **Fase 1:** la base queda congelada y solo se entrenan las dos cabezas nuevas (tasa 1e-3).\n'
        '- **Fase 2, fine-tuning:** se descongelan las últimas 60 capas, con las BatchNormalization fijas, y se '
        'reentrena con una tasa 10 veces menor (1e-4) para afinar sin romper lo aprendido.')
    if CONFIG and CONFIG.get('parametros_tl'):
        st.caption(f"Parámetros de EfficientNetB0 con las dos cabezas: {CONFIG['parametros_tl']:,}".replace(',', '.') +
                   ' (en la fase 1 se entrenan 331.161; en la fase 2, 2.980.441).')

# ------------------------------ Resultados ------------------------------
with tabs[4]:
    st.header('Resultados')
    if METRICAS is None:
        st.info('Falta datos/metricas_test.csv: subir la carpeta datos/ junto con app.py.')
    else:
        st.write('Todo lo de esta sección es sobre el **test interno**: imágenes que no se usaron ni para entrenar ni '
                 'para elegir el modelo (ver Limitaciones). La métrica principal es el **F1 macro**, que le da el mismo peso a cada clase, para que los '
                 'actores con muchas fotos no tapen a los que tienen pocas.')
        t = METRICAS.copy()
        tabla = t[['modelo', 'f1_actor', 'acc_actor', 'f1_emocion', 'acc_emocion']].rename(columns={
            'modelo': 'Modelo', 'f1_actor': 'F1 macro actor', 'acc_actor': 'Accuracy actor',
            'f1_emocion': 'F1 macro emoción', 'acc_emocion': 'Accuracy emoción'})
        st.dataframe(tabla.style.format({c: '{:.3f}' for c in tabla.columns[1:]}), hide_index=True, width='stretch')
        largo = t.melt(id_vars=['clave', 'modelo'], value_vars=['f1_actor', 'f1_emocion'], var_name='tarea', value_name='F1')
        largo['tarea'] = largo.tarea.map({'f1_actor': 'Actor', 'f1_emocion': 'Emoción'})
        largo['modelo'] = largo.clave.map({'cnn_propia': 'CNN propia', 'efficientnetb0': 'EfficientNetB0',
                                           'trivial': 'Trivial'}).fillna(largo.modelo)
        st.altair_chart(alt.Chart(largo).mark_bar().encode(
            x=alt.X('modelo:N', title=None, sort=['CNN propia', 'EfficientNetB0', 'Trivial'], axis=alt.Axis(labelAngle=0)),
            y=alt.Y('F1:Q', scale=alt.Scale(domain=[0, 1]), title='F1 macro'),
            color=alt.Color('clave:N', scale=alt.Scale(domain=list(COLOR), range=list(COLOR.values())), legend=None),
            column=alt.Column('tarea:N', title=None),
            tooltip=['modelo', 'tarea', alt.Tooltip('F1:Q', format='.3f')]).properties(width=260, height=260))

        st.subheader('Matriz de confusión')
        st.write('Cada fila es la clase real y cada columna lo que respondió la red; el número es el porcentaje de la fila. '
                 'La diagonal son los aciertos y lo que queda afuera muestra con quién se confunde cada clase.')
        disponibles = [c for c in ['cnn_propia', 'efficientnetb0']
                       if all(os.path.exists(os.path.join(DATOS, f'cm_{c}_{t}.csv')) for t in ['actor', 'emocion'])]
        if disponibles:
            c1, c2 = st.columns(2)
            clave = c1.selectbox('Modelo', disponibles, format_func=lambda c: NOMBRE[c])
            tarea = c2.selectbox('Tarea', ['emocion', 'actor'], format_func=lambda x: 'Emoción' if x == 'emocion' else 'Actor')
            cm = leer_csv(f'cm_{clave}_{tarea}.csv', index_col=0)
            n_fila = cm.sum(axis=1)
            pct = cm.div(n_fila.clip(lower=1), axis=0)
            filas = [f'{lindo(r)} (n={int(n_fila[r])})' for r in cm.index]
            largo_cm = pct.set_axis(filas, axis=0).set_axis([lindo(c) for c in cm.columns], axis=1) \
                          .reset_index(names='real').melt(id_vars='real', var_name='predicho', value_name='pct')
            alto = 34 * len(cm) + 20
            base = alt.Chart(largo_cm).encode(
                x=alt.X('predicho:N', sort=None, title='predicho', axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
                y=alt.Y('real:N', sort=None, title='real', axis=alt.Axis(labelLimit=260, labelOverlap=False)))
            mapa = base.mark_rect(stroke='#EDF2F7', strokeWidth=1).transform_filter('datum.pct > 0').encode(
                color=alt.Color('pct:Q', scale=alt.Scale(domain=[0, 1], range=['#FFFFFF', '#08519C']),
                                                           legend=alt.Legend(format='%', title=None)),
                                           tooltip=['real', 'predicho', alt.Tooltip('pct:Q', format='.0%')])
            texto = base.mark_text(fontSize=10 if len(cm) > 8 else 13).transform_filter('datum.pct >= 0.01').encode(
                text=alt.Text('pct:Q', format='.0%'),
                color=alt.condition(alt.datum.pct > 0.55, alt.value('white'), alt.value('#1A202C')))
            fondo = base.mark_rect(fill='white', stroke='#EDF2F7', strokeWidth=1)
            st.altair_chart((fondo + mapa + texto).properties(height=alto), width='stretch', height=alto + 90, theme=None)
            rep = leer_csv(f'reporte_{clave}_{tarea}.csv')
            if rep is not None:
                rep['clase'] = rep.clase.map(lindo)
                st.dataframe(rep.sort_values('f1', ascending=False).style.format(
                    {'precision': '{:.2f}', 'recall': '{:.2f}', 'f1': '{:.2f}'}), hide_index=True, width='stretch')
                st.caption('Precisión: de las veces que dijo esa clase, cuántas acertó. Recall: de todas las fotos de '
                           'esa clase, cuántas encontró. F1: combina las dos.')

        res61 = leer_csv('experimentos_cnn_resumen.csv')
        if res61 is not None:
            st.subheader('Experimento: cómo cerrar la CNN propia (validación)')
            st.write('Cada variante se entrenó con 2 semillas; se muestra la media ± el desvío. Una diferencia más chica '
                     'que el desvío es un empate. La elección se hizo con validación, no con test.')
            r = res61.copy()
            r.columns = [r.columns[0]] + list(r.columns[1:])
            nombres_var = {'maxpool_flatten': 'MaxPooling + Flatten', 'gap': 'GlobalAveragePooling',
                           'maxpool_flatten_peso_emo': 'MaxPooling + Flatten, emoción con peso 1,5'}
            out = pd.DataFrame({'Variante': r.iloc[:, 0].map(lambda v: nombres_var.get(v, v))})
            for tarea, etq in [('f1_actor', 'F1 actor'), ('f1_emocion', 'F1 emoción'), ('f1_promedio', 'F1 promedio')]:
                if f'{tarea}_mean' in r:
                    out[etq] = [f'{a:.3f} ± {b:.3f}' if pd.notna(b) else f'{a:.3f}'
                                for a, b in zip(r[f'{tarea}_mean'], r.get(f'{tarea}_std', pd.Series([np.nan] * len(r))))]
            st.dataframe(out, hide_index=True, width='stretch')

        ex62 = leer_csv('experimento_epocas_emocion.csv')
        if ex62 is not None:
            st.subheader('Experimento: ¿a la emoción le faltaban épocas? (validación)')
            st.write('Se siguió entrenando la red elegida hasta 40 épocas más, mirando solo la emoción. Es un experimento: '
                     'no reemplaza al modelo final. Una diferencia menor a 0,03 se toma como ruido.')
            e = ex62.rename(columns={'experimento': 'Experimento', 'epocas_extra': 'Épocas extra',
                                     'mejor_epoca': 'Mejor época', 'f1_emocion_val': 'F1 emoción val',
                                     'f1_emocion_train': 'F1 emoción train', 'f1_actor_val': 'F1 actor val',
                                     'vs_6_1': 'Diferencia con la red final'})
            e = e.drop(columns=[c for c in ['minutos'] if c in e])
            for c in e.columns[1:]:
                fmt = '{:.0f}' if c in ('Épocas extra', 'Mejor época') else '{:.3f}'
                e[c] = [('-' if pd.isna(v) else fmt.format(v)) for v in e[c]]
            st.dataframe(e, hide_index=True, width='stretch')

        if not os.path.exists(os.path.join(DATOS, 'curvas_cnn.csv')):
            st.caption('Las curvas época por época de la CNN propia no se muestran (falta datos/curvas_cnn.csv).')
        for archivo, titulo in [('curvas_cnn.csv', 'Curvas de entrenamiento de la CNN propia'),
                                ('curvas_tl.csv', 'Curvas de entrenamiento de EfficientNetB0')]:
            cur = leer_csv(archivo)
            if cur is None:
                continue
            st.subheader(titulo)
            if 'fase' in cur:
                cur = cur.copy()
                cur['epoca'] = np.arange(1, len(cur) + 1)
            partes = []
            for tarea, etq in [('actor_f1', 'Actor'), ('emocion_f1', 'Emoción')]:
                if tarea in cur and f'val_{tarea}' in cur:
                    partes.append(cur[['epoca', tarea, f'val_{tarea}']].rename(
                        columns={tarea: 'train', f'val_{tarea}': 'validación'}).melt('epoca', var_name='conjunto',
                                                                                      value_name='F1').assign(tarea=etq))
            if partes:
                d = pd.concat(partes)
                inicio_f2 = int(cur.loc[cur.fase == 2, 'epoca'].min()) if 'fase' in cur and (cur.fase == 2).any() else None
                for col, tarea in zip(st.columns(2), d.tarea.unique()):
                    lineas = alt.Chart(d[d.tarea == tarea]).mark_line().encode(
                        x=alt.X('epoca:Q', title='época'),
                        y=alt.Y('F1:Q', title='F1 macro', scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color('conjunto:N', title=None),
                        tooltip=['epoca', 'conjunto', alt.Tooltip('F1:Q', format='.3f')])
                    if inicio_f2:
                        lineas = lineas + alt.Chart(pd.DataFrame({'epoca': [inicio_f2 - 0.5]})).mark_rule(
                            strokeDash=[4, 4], color='#718096').encode(x='epoca:Q')
                    col.altair_chart(lineas.properties(title=tarea, height=230), width='stretch', height=270)
                st.caption('El F1 de train se mide con aumentación y dropout activos.' +
                           (' La línea punteada marca el comienzo del fine-tuning (fase 2).' if inicio_f2 else ''))

# ------------------------------ Limitaciones ------------------------------
with tabs[5]:
    st.header('Limitaciones')
    st.write('Lo que hay que tener en cuenta para leer los resultados:')
    pct = RESUMEN.get('cambio_etiqueta_pct')
    n_test = CONFIG.get('n_test') if CONFIG else None
    items = [
        f"**Ruido en las etiquetas de emoción.** " + (f"La emoción cambia en el {pct}% de los cuadros seguidos de un mismo "
                                                       f"video. " if pct else '') +
        "Es probablemente el principal techo de la tarea de emoción. El acuerdo entre las personas que etiquetaron "
        "no se midió todavía; ese número sería la cota realista de cualquier modelo.",
        "**Desbalance.** Hay actores con mucho más material que otros, y algunos casi no tienen fotos de ciertas emociones. "
        "La red podría aprender el atajo \"este actor nunca sonríe\". Se compensa en parte con pesos por clase.",
        f"**Test chico.** " + (f"Son {n_test} imágenes, " if n_test else '') +
        "así que algunos actores tienen pocas fotos de test y una foto más o menos mueve mucho su resultado.",
        "**Posible fuga entre bloques.** Los videos largos y las imágenes sin video identificado se cortan en bloques "
        "de 10 cuadros, y dos bloques vecinos del mismo video pueden quedar uno en train y otro en test. Eso puede "
        "inflar un poco el resultado de test. Es el precio de un test chico." +
        (f" Acá el test es el {n_test / (CONFIG['n_train'] + CONFIG['n_val'] + n_test):.0%} de las imágenes."
         if n_test and CONFIG.get('n_train') else ''),
        "**Un cuadro solo.** La emoción se decide con una foto suelta, sin el movimiento del gesto. Con varios cuadros "
        "seguidos (una CNN por cuadro y una LSTM encima, como en la Unidad 4) se podría leer cómo cambia la cara.",
        "**Detector de caras.** El detector Haar no encuentra caras de perfil o muy de costado; en ese caso se usa el "
        "centro de la imagen.",
        "**Solo conoce 20 personas.** Con la foto de cualquier otra persona igual elige a uno de los 20 actores.",
    ]
    for it in items:
        st.markdown(f'- {it}')
    if CONFIG:
        st.caption(f"Resultados de {CONFIG.get('notebook', 'el notebook')} · CNN elegida: {CONFIG.get('mejor_cnn', '-')} · "
                   f"Keras {CONFIG.get('keras', '-')} · TensorFlow {CONFIG.get('tensorflow', '-')}")
