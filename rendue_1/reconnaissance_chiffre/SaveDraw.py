from PIL import Image
import numpy as np   

class SaveDraw:

    def __init__(self, root, draw):
        self.root = root
        self.draw = draw

    def save_as_matrix(self):
        self.draw.canvas.postscript(file="img/drawing_temp.ps")
        # image
        img = Image.open("img/drawing_temp.ps")
        # image traitable en png
        img.save("img/drawing_temp.png")
        # noir et blanc (dispensable) 
        img_gray = img.convert("1")

        np_img = np.array(img_gray)
        print(np_img)
        self.root.destroy()
