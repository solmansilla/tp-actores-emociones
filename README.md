# App del TP: Actores y Emociones

App de Streamlit que muestra el TP de IA y Aprendizaje Automático II (UCA Rosario): subís una foto, recorta la cara
igual que el notebook y muestra qué actor y qué emoción responden la CNN propia y EfficientNetB0. Tiene además
pestañas de dataset (solo cantidades, sin imágenes), arquitectura, resultados y limitaciones.

Todos los números y gráficos son los del notebook final, `TP_CNN_actor_emocion_v2_FINAL.ipynb`
(test: CNN propia 0.873 / 0.393, EfficientNetB0 0.926 / 0.467 de F1 macro en actor / emoción).

## Qué hay en la carpeta

```
app.py
requirements.txt
.streamlit/config.toml
datos/                 <- resultados del notebook final (métricas, matrices, curvas, dataset). Ya están listos.
modelos/haarcascade_frontalface_default.xml
```

Las dos redes NO van en la carpeta: pesan 31 y 42 MB y GitHub no deja subir desde el navegador archivos de más
de 25 MB. La app las baja de Google Drive la primera vez que alguien la abre, con los links de `datos/config.json`.
Los links ya están cargados: son `CNN_PROPIA_FINAL.keras` y `TL_EFFICIENTNETB0_FINAL.keras` de la carpeta
`IA2_resultados_v2` de Juan Ignacio, las redes que el notebook final evaluó en test. Tienen que seguir compartidas como
"Cualquier persona con el enlace" (alcanza con permiso de lector), y no hay que borrarlas ni moverlas de esa carpeta.

## Pasos

1. **Subir a GitHub.** Crear un repositorio y subir todo el contenido de la carpeta ("Add file > Upload files",
   arrastrar). La carpeta `.streamlit` empieza con punto: si no aparece al arrastrar, crearla a mano en GitHub
   ("Add file > Create new file", nombre `.streamlit/config.toml`, y pegar el contenido).
2. **Publicar.** Entrar a https://share.streamlit.io con la cuenta de GitHub, "Create app", elegir el repositorio,
   la rama `main` y el archivo `app.py`. En "Advanced settings" elegir Python 3.12. Deploy.
   La primera vez tarda unos minutos porque instala TensorFlow, y la primera foto tarda un minuto más
   porque baja las redes de Drive.

Si una red no se puede bajar, la pestaña "Probar con una foto" lo dice (por ejemplo, si el archivo no está
compartido) y sigue andando con la otra. Las pestañas de resultados no dependen de las redes.

Otra opción, sin Drive: subir los dos `.keras` a la carpeta `modelos/` con git (no desde el navegador).
Si están ahí, la app los usa y no mira los links.

## Privacidad

El enunciado dice que el dataset no se publica fuera del curso. La app no incluye ninguna imagen del dataset:
solo las redes y tablas de cantidades y métricas. Si prefieren que no sea pública: con el repositorio privado,
la app queda privada y se invita por email a quien la tenga que ver (App settings > Sharing).

## Probarla en la compu (opcional, Windows o Linux)

```
pip install -r requirements.txt
streamlit run app.py
```

En Mac no hay `tensorflow-cpu`: cambiar esa línea de requirements.txt por `tensorflow==2.21.0` solo para probar
en la compu (para Streamlit Cloud dejarla como está).
