from ultralytics import FastSAM
from PIL import Image
import matplotlib.pyplot as plt

model = FastSAM('FastSAM-s.pt')
# Create a dummy image or use an image if we have one (we don't have the user's dog image, but we can just write the test logic we will use in app.py)
