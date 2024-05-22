from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
from prometheus_client import Summary, start_http_server, Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
import time
import psutil
from PIL import Image
import io

app = FastAPI()


@app.get("/")
def home():
    return "Hello World"


REQUEST_DURATION = Summary("api_timing", "Request duration in seconds")
counter = Counter(
    "api_usage_counter", "Total number of API requests", ["endpoint", "client"]
)
gauge_runtime = Gauge(
    "api_runtime_secs", "runtime of the method in seconds", ["endpoint", "client"]
)
gauge_length = Gauge(
    "input_text_length", "length of input text", ["endpoint", "client"]
)
gauge_ptpc = Gauge(
    "ptpc", "Processing time per character (PTPC)", ["endpoint", "client"]
)
# Additional metrics
gauge_memory_utilization = Gauge(
    "api_memory_utilization", "API memory utilization", ["endpoint", "client"]
)
gauge_cpu_utilization = Gauge(
    "api_cpu_utilization", "API CPU utilization rate", ["endpoint", "client"]
)
gauge_network_io_bytes = Gauge(
    "api_network_io_bytes", "API network I/O bytes sent", ["endpoint", "client"]
)
gauge_network_io_rate = Gauge(
    "api_network_io_rate", "API network I/O rate sent", ["endpoint", "client"]
)

app = FastAPI()
Instrumentator().instrument(app).expose(app)


# Load the model using a h5 file given by the path
# def load_module(path):
#     model = load_model(path)
#     return model


# Format the image to a 28x28 grayscale image
def format_image(image):
    image = image.resize((28, 28))
    image = image.convert("L")
    return image


# Predict the digit given an array of shape (1, 784) and a keras model
# def predict_digit(image_array, model):
#     probs = model.predict(image_array, verbose=True)
#     print("Predicted Digit:", np.argmax(probs))
#     return str(np.argmax(probs))


# Endpoint to predict the digit given an image file (png, jpg, jpeg)
@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
):
    initial_network_io = psutil.net_io_counters()
    start = time.time()
    # load model from h5 file given by the path
    # model = load_module("model.h5")
    contents = await file.read()
    # Convert image into a PIL image
    image = Image.open(io.BytesIO(contents))
    image = format_image(image)
    image_array = np.array(image)
    # Flatten the 28x28 image into a 1x784 array
    image_array = image_array.flatten().reshape(1, 784)
    # Normalize the image array
    image_array = image_array / 255.0
    digit = np.random(10)
    time_taken = time.time() - start  # Time taken
    # Capture final network I/O
    final_network_io = psutil.net_io_counters()
    memory_info = psutil.virtual_memory()
    memory_util = memory_info.percent
    cpu_util = psutil.cpu_percent(interval=1)
    io_bytes_sent = final_network_io.bytes_sent - initial_network_io.bytes_sent
    io_rate_sent = io_bytes_sent / time_taken
    input_text_length = 784
    process_time_per_char = 0
    if input_text_length > 0:
        process_time_per_char = time_taken * 1000000.0 / input_text_length  #

    gauge_ptpc.labels(endpoint="/np", client=request.client.host).set(
        process_time_per_char
    )
    gauge_runtime.labels(endpoint="/np", client=request.client.host).set(
        time_taken
    )  # Gauge time
    gauge_memory_utilization.labels(endpoint="/np", client=request.client.host).set(
        memory_util
    )  # Gauge memory utilization
    gauge_cpu_utilization.labels(endpoint="/np", client=request.client.host).set(
        cpu_util
    )  # Gauge CPU utilization
    gauge_network_io_bytes.labels(endpoint="/np", client=request.client.host).set(
        io_bytes_sent
    )  # Gauge API network I/O bytes sent
    gauge_network_io_rate.labels(endpoint="/np", client=request.client.host).set(
        io_rate_sent
    )  # Gauge API network I/O rate sent

    return {"digit": digit}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


Instrumentator().instrument(app).expose(app)
