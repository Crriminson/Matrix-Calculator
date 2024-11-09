from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, StringVar, OptionMenu, messagebox
import numpy as np
from scipy.linalg import lu

# Paths
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\adity\Matrix-Calculator\gui_windows\assets\frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Matrix operation functions with steps
# (Existing functions for echelon_form, is_unitary, etc.)

def multiply_matrices(matrix1, matrix2):
    steps = ["Matrix Multiplication:"]
    try:
        result = np.dot(matrix1, matrix2)
        steps.append(f"Result:\n{result}")
    except ValueError:
        steps.append("Matrix multiplication not possible due to incompatible dimensions.")
        result = None
    return result, "\n".join(steps)

def add_matrices(matrix1, matrix2):
    steps = ["Matrix Addition:"]
    try:
        result = matrix1 + matrix2
        steps.append(f"Result:\n{result}")
    except ValueError:
        steps.append("Matrix addition not possible due to incompatible dimensions.")
        result = None
    return result, "\n".join(steps)

# Encoding function remains the same
# Decoding function remains the same

# Perform matrix operation
def perform_operation():
    operation = selected_operation.get()
    try:
        # Get first matrix values from entry fields
        matrix1 = np.array([[float(entry.get()) for entry in row] for row in entries_matrix])

        # Get second matrix if needed
        matrix2 = None
        if operation in ["Matrix Multiplication", "Matrix Addition", "Encode", "Decode"]:
            matrix2 = np.array([[float(entry.get()) for entry in row] for row in secondary_entries_matrix])

        result, steps = None, ""
        if operation == "Echelon Form":
            result, steps = echelon_form(matrix1)
        elif operation == "Is Unitary":
            result, steps = is_unitary(matrix1)
        elif operation == "Normal Form":
            result, steps = normal_form(matrix1)
        elif operation == "PAQ Form":
            result, steps = paq_form(matrix1)
        elif operation == "Transpose":
            result, steps = transpose(matrix1)
        elif operation == "Is Orthogonal":
            result, steps = is_orthogonal(matrix1)
        elif operation == "Inverse":
            result, steps = inverse(matrix1)
        elif operation == "Determinant":
            result, steps = determinant(matrix1)
        elif operation == "Conjugate":
            result, steps = conjugate(matrix1)
        elif operation == "Encode":
            result, steps = encode(matrix1, matrix2)
        elif operation == "Decode":
            result, steps = decode(matrix1, matrix2)
        elif operation == "Matrix Multiplication":
            result, steps = multiply_matrices(matrix1, matrix2)
        elif operation == "Matrix Addition":
            result, steps = add_matrices(matrix1, matrix2)
        else:
            raise NotImplementedError(f"{operation} is not implemented.")

        steps_text.delete("1.0", "end")
        steps_text.insert("1.0", f"Result:\n{result}\n\nSteps:\n{steps}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Update entry fields based on matrix size
def update_matrix_size(*args):
    global entries_matrix, secondary_entries_matrix, encryption_entries_matrix
    for row in entries_matrix:
        for entry in row:
            entry.place_forget()
    for row in secondary_entries_matrix:
        for entry in row:
            entry.place_forget()
    for row in encryption_entries_matrix:
        for entry in row:
            entry.place_forget()

    size = int(selected_size.get()[0])
    entries_matrix = []
    secondary_entries_matrix = []
    encryption_entries_matrix = []

    # Main matrix entries
    for i in range(size):
        row_entries = []
        for j in range(size):
            entry = Entry(window, bd=0, bg="#D9D9D9", fg="#000716", highlightthickness=0, justify='center')
            entry.place(x=150 + j * 60, y=120 + i * 40, width=50, height=30)
            row_entries.append(entry)
        entries_matrix.append(row_entries)

    # Secondary matrix entries
    for i in range(size):
        row_entries = []
        for j in range(size):
            entry = Entry(window, bd=0, bg="#E0E0E0", fg="#000716", highlightthickness=0, justify='center')
            entry.place(x=450 + j * 60, y=120 + i * 40, width=50, height=30)
            row_entries.append(entry)
        secondary_entries_matrix.append(row_entries)

    # Encryption matrix entries (only visible if Encode or Decode selected)
    if selected_operation.get() in ["Encode", "Decode"]:
        for i in range(size):
            row_entries = []
            for j in range(size):
                entry = Entry(window, bd=0, bg="#FFFACD", fg="#000716", highlightthickness=0, justify='center')
                entry.place(x=450 + j * 60, y=120 + i * 40, width=50, height=30)
                row_entries.append(entry)
            encryption_entries_matrix.append(row_entries)

# Toggle secondary and encryption matrix visibility based on operation
def toggle_secondary_matrix(*args):
    if selected_operation.get() in ["Matrix Multiplication", "Matrix Addition", "Encode", "Decode"]:
        for row in secondary_entries_matrix:
            for entry in row:
                entry.place(x=450 + row.index(entry) * 60, y=120 + secondary_entries_matrix.index(row) * 40, width=50, height=30)
    else:
        for row in secondary_entries_matrix:
            for entry in row:
                entry.place_forget()

    toggle_encryption_entries()

def toggle_encryption_entries(*args):
    if selected_operation.get() in ["Encode", "Decode"]:
        for row in encryption_entries_matrix:
            for entry in row:
                entry.place(x=450 + row.index(entry) * 60, y=120 + encryption_entries_matrix.index(row) * 40, width=50, height=30)
    else:
        for row in encryption_entries_matrix:
            for entry in row:
                entry.place_forget()

# Initialize main window
window = Tk()
window.geometry("800x500")
window.configure(bg="#DACECE")
canvas = Canvas(window, bg="#DACECE", height=500, width=800, bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)

# Add "Matrix Calculator" heading
canvas.create_text(
    317, 30,
    text="Matrix Calculator",
    font=("Arial", 24, "bold"),
    fill="#000000"
)

# Matrix size dropdown
selected_size = StringVar(window)
selected_size.set("3x3")
matrix_sizes = ["2x2", "3x3", "4x4"]
size_dropdown = OptionMenu(window, selected_size, *matrix_sizes, command=update_matrix_size)
size_dropdown.place(x=150, y=70)

# Operation dropdown
selected_operation = StringVar(window)
selected_operation.set("Select Operation")
matrix_operations = [
    "Echelon Form", "Is Unitary", "Normal Form", "PAQ Form", "Transpose",
    "Is Orthogonal", "Inverse", "Determinant", "Conjugate", "Encode",
    "Decode", "Matrix Multiplication", "Matrix Addition"
]
operation_dropdown = OptionMenu(window, selected_operation, *matrix_operations)
operation_dropdown.place(x=350, y=70)
selected_operation.trace("w", toggle_secondary_matrix)

# Matrix entry fields
entries_matrix = []
secondary_entries_matrix = []
encryption_entries_matrix = []
update_matrix_size()

# Calculate button
button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=perform_operation,
    relief="flat"
)
button_1.place(x=247.0, y=260.0, width=139.0, height=62.0)

# Text area to show result and calculation steps
steps_text = Text(window, height=10, width=60, wrap="word", bg="#FFFFFF", fg="#000000", font=("Arial", 10))
steps_text.place(x=50, y=340)

window.resizable(False, False)
window.mainloop()
