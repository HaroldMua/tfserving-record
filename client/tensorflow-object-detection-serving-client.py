import PIL.Image
import numpy
import requests
from pprint import pprint
import time

image = PIL.Image.open("penguin.jpg")  # Change penguin.jpg with your image
image_np = numpy.array(image)

REST_URL = "http://192.168.1.24:8080/v1/models/default:predict"

payload = {"instances": [image_np.tolist()]}

start = time.perf_counter()
res = requests.post(REST_URL, json=payload)
print(f"Took {time.perf_counter()-start:.2f}s")
end = time.perf_counter()
pprint(res.json())
