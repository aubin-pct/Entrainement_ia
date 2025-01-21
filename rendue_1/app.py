from reconnaissance_chiffre import Draw
import tkinter as tk


root = tk.Tk()
root.title("Dessin avec la souris")

draw = Draw.Draw(root=root)


root.mainloop()