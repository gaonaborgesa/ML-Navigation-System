# classifier_test.py

import os
from PIL import Image
import numpy as np
from joblib import dump, load

PATH_TO_MODEL = "./models/htr_new_model.joblib"


def main():
  """
  Load first preprocessed image, run inference, and print the output.
  """
  clf = load(PATH_TO_MODEL)
  img = Image.open(os.path.join("output", "08_bordered_frame_0.jpg"))

  # Processing for image to match what model expects
  data = np.asarray(img).reshape(1, -1)  
  data = data.astype(float) / 255.0

  pred = clf.predict(data)[0]
  print("Detected", pred)


if __name__ == "__main__":
  main()