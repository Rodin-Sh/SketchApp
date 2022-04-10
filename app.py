from PIL import Image, ImageOps
from noise import pnoise2
import cv2 as cv
import numpy as np
import random
import streamlit as st

# Using the pnoise2 function from noise PIP, noise is added to each pixel.
def background(dimensions, octaves):
  texture = np.empty([dimensions[1], dimensions[0]])
  freq = 10.0 * octaves # Default: 16.0

  for i in range(dimensions[1]):
      for j in range(dimensions[0]):
          v = int(pnoise2(i / (freq*2), j / (freq*2), octaves) * 127.0 + 128.0) # Default: * 127.0 + 128.0
          texture[i][j] = v 

  return Image.fromarray(texture).convert("RGB")

# Using Canny and GaussianBlur functions from cv2 PIP, we are able to detect edges and blur them, respectively.
def edges(image):
  grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  kernel = np.ones((3, 3), np.uint8) #Default: (3,3)
  edges = cv.Canny(grey, 300, 300) #Default: (200, 300) #Good: (120, 120)
  blr = cv.GaussianBlur(edges, (3,3), 0) #Default: (3, 3)
  dil = cv.dilate(edges, kernel, iterations = 1)
  image_gray = 255-dil
  image_rgba = cv.cvtColor(image_gray, cv.COLOR_GRAY2RGBA)
  white = np.all(image_rgba == [255,255,255,255], axis=-1)
  image_rgba[white, -1] = 0
  cv.imwrite("edges.png", image_gray)
  return image_gray

# Using background and edges function, the customizations are performed on the greyscale version of the original image.
def sketch(image):
  grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  inv = 255 - grey
  blur = cv.GaussianBlur(inv, (13,13), 0) #Default: 13
  return cv.divide(grey, 255-blur, scale=256)

st.title('Sketch Book')
st.write('An application that modifies images into sketches. Credit: John Fish.')
uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
  # Convert the file to an opencv image.
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  img = cv.imdecode(file_bytes, 1)

  col1, col2 = st.columns(2)
  col1.image(uploaded_file, caption="Original Image")

  if st.button("Sketch"):
    sketch = sketch(img)
    cv.imwrite("sketch.png", sketch)
    bg = background(dimensions=img.shape, octaves=6)
    edges = edges(img)
    # sketchTrans = cv.cvtColor(sketch, cv.COLOR_GRAY2RGBA)

    mask = edges[3]
    sketch = cv.bitwise_and(sketch, edges, edges)
    (thresh, sketch) = cv.threshold(sketch, 240, 255, cv.THRESH_BINARY) # Default: 240, 255
    # sketch = cv.multiply(sketch, np.array(bg), scale=(1./128))

    h, w = sketch.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # mask[1:h+1, 1:w+1] = sketch
    sketchColor = cv.cvtColor(sketch, cv.COLOR_GRAY2RGBA)
    # Makes the white pixels in the image transparent.
    # white = np.all(sketchColor == [255,255,255,255], axis=-1)
    # sketchColor[white, -1] = 0

    cv.imwrite("final.png", sketchColor)
    final = Image.fromarray(sketchColor)
    col2.image(final, caption="Sketched Image")
