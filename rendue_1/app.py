import tkinter as tk

"Commence à dessiner une ligne à partir de la position de la souris."
def start_drawing(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

"Dessine une ligne de la dernière position à la position actuelle de la souris."
def draw(event):
    global last_x, last_y
    canvas.create_line(last_x, last_y, event.x, event.y, width=4, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
    last_x, last_y = event.x, event.y

"Arrête le dessin."
def stop_drawing(event):
    global last_x, last_y
    last_x, last_y = None, None

root = tk.Tk()
root.title("Dessin avec la souris")

canvas = tk.Canvas(root, width=600, height=400, bg='white')
canvas.pack()

canvas.bind('<ButtonPress-1>', start_drawing)
canvas.bind('<B1-Motion>', draw)
canvas.bind('<ButtonRelease-1>', stop_drawing)



root.mainloop()
