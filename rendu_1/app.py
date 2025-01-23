from reconnaissance_chiffre import *
import tkinter as tk


root = tk.Tk()
root.title("Dessin avec la souris")
draw = Draw.Draw(root=root)
save_draw = SaveDraw.SaveDraw(root, draw)
root.protocol("WM_DELETE_WINDOW", save_draw.save_as_matrix)

root.mainloop()

