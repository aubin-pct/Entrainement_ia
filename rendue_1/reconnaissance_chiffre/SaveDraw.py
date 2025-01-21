from PIL import Image
import numpy as np   

class SaveDraw:

    def __init__(self, root, draw):
        self.root = root
        self.draw = draw

    def save_as_matrix(self):
        # recuperation de l'image
        self.draw.canvas.postscript(file="img/drawing_temp.ps")
        img = Image.open("img/drawing_temp.ps")
        # image traitable en png
        img.save("img/drawing_temp.png")
        # redimentionnement de l'image 
        img = img.resize((80,80), Image.BICUBIC)
        # noir et blanc
        img_gray = img.convert("1")
        img_gray.save("img/drawing_temp.png")
        np_img = np.array(img_gray)
        self.root.destroy()
