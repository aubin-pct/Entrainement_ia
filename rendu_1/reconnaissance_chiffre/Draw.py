import tkinter as tk

class Draw:
    global last_x, last_y
    def __init__(self, root, width=600, height=400):
        self.canvas = tk.Canvas(root, width=width, height=height, bg='white')
        self.canvas.pack()
        self.canvas.bind('<ButtonPress-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        self.last_x, self.last_y = None, None


    # Commence à dessiner une ligne à partir de la position de la souris.
    def start_drawing(self, event):
        self.last_x, self.last_y = event.x, event.y

    # Dessine une ligne de la dernière position à la position actuelle de la souris.
    def draw(self, event):
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=10, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
        self.last_x, self.last_y = event.x, event.y

    # Arrête le dessin.
    def stop_drawing(self, event):
        self.last_x, self.last_y = None, None


   

