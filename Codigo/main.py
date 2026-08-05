from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

modelo = load_model("keras_model.h5", compile=False)

with open("labels.txt", "r") as f:
    etiquetas = [line.strip() for line in f.readlines()]


def predecir(ruta):

    imagen = Image.open(ruta).convert("RGB")
    imagen = imagen.resize((224, 224))

    datos = np.asarray(imagen)
    datos = (datos.astype(np.float32) / 127.5) - 1

    datos = np.expand_dims(datos, axis=0)

    prediccion = modelo.predict(datos)

    indice = np.argmax(prediccion)

    return etiquetas[indice], float(prediccion[0][indice])


@app.route("/", methods=["GET", "POST"])
def inicio():

    resultado = None

    if request.method == "POST":

        archivo = request.files["imagen"]

        ruta = "static/" + archivo.filename

        archivo.save(ruta)

        clase, confianza = predecir(ruta)

        resultado = {
            "imagen": ruta,
            "clase": clase,
            "confianza": round(confianza * 100, 2)
        }

    return render_template("index.html", resultado=resultado)

if __name__ == "__main__":
    app.run(debug=True)
