from pathlib import Path
from tkinter import Tk, Canvas, OptionMenu, Button, PhotoImage, StringVar

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

# List of matrix operations
matrix_operations = [
    "Transpose",
    "Conjugate",
    "Addition",
    "Multiplication",
    "Determinant",
    "Inverse",
    "Orthogonal Checker",
    "Unitary Checker",
    "Echelon Form",
    "Rank Checker",
    "Normal Form",
    "PAQ Form",
    "Coding Decoding",
    "Test for Consistency (Linear Equations)",
    "Test for Consistency (Homogeneous Equations)",
    "Linear Dependence/Independence",
    "Gauss-Jacobi",
    "Gauss-Seidel",
    "Unknown Reductions for Echelon"
]

# Create a dropdown menu for selecting matrix operations
selected_operation = StringVar(window)
selected_operation.set("Select Operation")  # Default text

# Dropdown (OptionMenu) styling
operation_dropdown = OptionMenu(window, selected_operation, *matrix_operations)
operation_dropdown.config(
    width=25,
    font=("Helvetica", 12),
    bg="#D9D9D9",
    fg="#000716",
    relief="flat",
    highlightthickness=0
)
operation_dropdown.place(
    x=220.0,
    y=197.0,
    width=194.0,
    height=43.0
)

# Button to confirm selection
try:
    button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: print(f"Selected operation: {selected_operation.get()}"),
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
