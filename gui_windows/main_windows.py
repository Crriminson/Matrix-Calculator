from pathlib import Path
from tkinter import Tk, Canvas, Entry, Button, PhotoImage

# Set up paths
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / "assets"  # Use a relative path for the assets folder

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Initialize the main window
window = Tk()
window.geometry("634x395")
window.configure(bg="#FFFFFF")

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=395,
    width=634,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)

# Create main background rectangle
canvas.create_rectangle(
    0.0,
    0.0,
    634.0,
    395.0,
    fill="#DACDCD",
    outline=""
)

# Entry box background
entry_image_1 = PhotoImage(file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    317.0,
    219.5,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_1.place(
    x=220.0,
    y=197.0,
    width=194.0,
    height=43.0
)

# Header rectangle
canvas.create_rectangle(
    0.0,
    56.0,
    634.0,
    114.0,
    fill="#FFFFFF",
    outline=""
)

# Title text
canvas.create_text(
    184.0,
    47.0,
    anchor="nw",
    text="Matrix Calculator",
    fill="#000000",
    font=("InknutAntiqua Medium", 32)
)

# Button
try:
    button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: print("button_1 clicked"),
        relief="flat"
    )
    button_1.place(
        x=276.0,
        y=259.0,
        width=83.0,
        height=41.0
    )
except Exception as e:
    print("Error loading button image:", e)

window.resizable(False, False)
window.mainloop()