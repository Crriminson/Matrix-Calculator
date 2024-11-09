from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, StringVar, OptionMenu, messagebox, simpledialog
import numpy as np
from scipy.linalg import lu

# Paths
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\adity\Matrix-Calculator\gui_windows\assets\frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Matrix operation functions with steps
def echelon_form(matrix):
    matrix = matrix.astype(float)
    rows, cols = matrix.shape
    steps = ["Starting Echelon Form Calculation:"]
    for i in range(min(rows, cols)):
        max_row = np.argmax(abs(matrix[i:, i])) + i
        if i != max_row:
            steps.append(f"Swapping row {i + 1} with row {max_row + 1}")
        matrix[[i, max_row]] = matrix[[max_row, i]]
        matrix[i] = matrix[i] / matrix[i, i]
        steps.append(f"Dividing row {i + 1} by pivot {matrix[i, i]}: {matrix[i]}")
        for j in range(i + 1, rows):
            factor = matrix[j, i]
            matrix[j] = matrix[j] - matrix[i] * factor
            steps.append(f"Row {j + 1} - ({factor}) * Row {i + 1}: {matrix[j]}")
    return matrix, "\n".join(steps)

def is_unitary(matrix):
    steps = ["Checking if matrix is unitary:"]
    check = np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]))
    steps.append(f"Computed matrix * conjugate transpose: {matrix @ matrix.conj().T}")
    steps.append(f"Expected identity matrix: {np.eye(matrix.shape[0])}")
    steps.append("Matrix is unitary." if check else "Matrix is not unitary.")
    return check, "\n".join(steps)

def normal_form(matrix):
    eig_vals, _ = np.linalg.eig(matrix)
    steps = [
        f"Eigenvalues: {eig_vals}",
        f"Normal Form (Diagonal Matrix of Eigenvalues): {np.diag(eig_vals)}"
    ]
    return np.diag(eig_vals), "\n".join(steps)

def paq_form(matrix):
    P, _, U = lu(matrix)
    steps = [
        f"P matrix (Permutation):\n{P}",
        f"Original matrix:\n{matrix}",
        f"U matrix (Upper triangular from LU decomposition):\n{U}",
        "PAQ Form is P * Original Matrix * U"
    ]
    result = P @ matrix @ U
    return result, "\n".join(steps)

# Other operations like Transpose, Determinant, etc. would also go here...

# Perform matrix operation
def perform_operation():
    operation = selected_operation.get()
    try:
        # Get matrix values from entry fields
        matrix = np.array([[float(entry.get()) for entry in entry_row] for entry_row in entries_matrix])

        result, steps = None, ""
        if operation == "Echelon Form":
            result, steps = echelon_form(matrix)
        elif operation == "Is Unitary":
            result, steps = is_unitary(matrix)
        elif operation == "Normal Form":
            result, steps = normal_form(matrix)
        elif operation == "PAQ Form":
            result, steps = paq_form(matrix)
        else:
            raise NotImplementedError(f"{operation} is not implemented.")

        # Display the result with steps
        steps_text.delete("1.0", "end")
        steps_text.insert("1.0", f"Result:\n{result}\n\nSteps:\n{steps}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Initialize main window
window = Tk()
window.geometry("634x395")
window.configure(bg="#DACECE")
canvas = Canvas(window, bg="#DACECE", height=395, width=634, bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)

# Operation dropdown
selected_operation = StringVar(window)
selected_operation.set("Select Operation")
matrix_operations = ["Echelon Form", "Is Unitary", "Normal Form", "PAQ Form"]  # Add other operations as needed
operation_dropdown = OptionMenu(window, selected_operation, *matrix_operations)
operation_dropdown.place(x=350, y=50)

# Matrix entry fields
entries_matrix = []
for i in range(3):  # assuming 3x3 matrix
    row_entries = []
    for j in range(3):
        entry = Entry(window, bd=0, bg="#D9D9D9", fg="#000716", highlightthickness=0)
        entry.place(x=150 + j * 40, y=100 + i * 30, width=30, height=20)
        row_entries.append(entry)
    entries_matrix.append(row_entries)

# Calculate button
button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
button_1 = Button(image=button_image_1, borderwidth=0, highlightthickness=0, command=perform_operation, relief="flat")
button_1.place(x=247.0, y=225.0, width=139.0, height=62.0)

# Text area to show result and calculation steps
steps_text = Text(window, height=10, width=50)
steps_text.place(x=50, y=280)

window.resizable(False, False)
window.mainloop()
