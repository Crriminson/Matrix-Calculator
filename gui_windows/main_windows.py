from pathlib import Path
from tkinter import Tk, Canvas, OptionMenu, Button, StringVar, Label, Text, END, messagebox, simpledialog
import numpy as np
from scipy.linalg import lu

# Set up paths
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / "assets"  # Use a relative path for the assets folder

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Function to calculate echelon form with steps
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

# Function to check if a matrix is unitary with steps
def is_unitary(matrix):
    steps = ["Checking if matrix is unitary:"]
    check = np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]))
    steps.append(f"Computed matrix * conjugate transpose: {matrix @ matrix.conj().T}")
    steps.append(f"Expected identity matrix: {np.eye(matrix.shape[0])}")
    steps.append("Matrix is unitary." if check else "Matrix is not unitary.")
    return check, "\n".join(steps)

# Function to convert to normal form with steps
def normal_form(matrix):
    eig_vals, eig_vecs = np.linalg.eig(matrix)
    steps = [
        f"Eigenvalues: {eig_vals}",
        f"Normal Form (Diagonal Matrix of Eigenvalues): {np.diag(eig_vals)}"
    ]
    return np.diag(eig_vals), "\n".join(steps)

# Function to perform PAQ form with steps
def paq_form(matrix):
    P, L, U = lu(matrix)
    steps = [
        f"P matrix (Permutation):\n{P}",
        f"Original matrix:\n{matrix}",
        f"U matrix (Upper triangular from LU decomposition):\n{U}",
        "PAQ Form is P * Original Matrix * U"
    ]
    result = P @ matrix @ U
    return result, "\n".join(steps)

# Function to encode a matrix with steps
def encode_matrix(matrix, scalar=1):
    steps = [f"Encoding matrix by adding scalar {scalar} to each element."]
    encoded_matrix = matrix + scalar
    steps.append(f"Encoded Matrix:\n{encoded_matrix}")
    return encoded_matrix, "\n".join(steps)

# Function to decode a matrix with steps
def decode_matrix(matrix, scalar=1):
    steps = [f"Decoding matrix by subtracting scalar {scalar} from each element."]
    decoded_matrix = matrix - scalar
    steps.append(f"Decoded Matrix:\n{decoded_matrix}")
    return decoded_matrix, "\n".join(steps)

# Perform matrix operation
def perform_operation():
    operation = selected_operation.get()
    try:
        # Get matrix values from text area
        matrix_input = text_area.get("1.0", END).strip()
        if not matrix_input:
            raise ValueError("Matrix input cannot be empty.")

        # Convert input to a numpy array
        matrix = np.array([list(map(float, row.split())) for row in matrix_input.splitlines()])

        # Perform the selected operation
        result, steps = None, ""
        if operation == "Transpose":
            result = matrix.T
            steps = f"Transposed matrix:\n{result}"
        elif operation == "Determinant":
            result = np.linalg.det(matrix)
            steps = f"Calculated determinant: {result}"
        elif operation == "Inverse":
            result = np.linalg.inv(matrix)
            steps = f"Inverse matrix:\n{result}"
        elif operation == "Addition":
            second_matrix_input = simpledialog.askstring("Input", "Enter second matrix (row by row):")
            second_matrix = np.array([list(map(float, row.split())) for row in second_matrix_input.splitlines()])
            result = matrix + second_matrix
            steps = f"Added matrices:\n{result}"
        elif operation == "Multiplication":
            second_matrix_input = simpledialog.askstring("Input", "Enter second matrix (row by row):")
            second_matrix = np.array([list(map(float, row.split())) for row in second_matrix_input.splitlines()])
            result = np.dot(matrix, second_matrix)
            steps = f"Multiplied matrices:\n{result}"
        elif operation == "Echelon Form":
            result, steps = echelon_form(matrix)
        elif operation == "Is Unitary":
            result, steps = is_unitary(matrix)
        elif operation == "Normal Form":
            result, steps = normal_form(matrix)
        elif operation == "PAQ Form":
            result, steps = paq_form(matrix)
        elif operation == "Encode":
            scalar = float(simpledialog.askstring("Input", "Enter scalar for encoding:"))
            result, steps = encode_matrix(matrix, scalar)
        elif operation == "Decode":
            scalar = float(simpledialog.askstring("Input", "Enter scalar used in encoding:"))
            result, steps = decode_matrix(matrix, scalar)
        else:
            raise NotImplementedError(f"{operation} is not implemented.")

        # Display the result with steps
        messagebox.showinfo("Result", f"Result:\n{result}\n\nSteps:\n{steps}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Initialize the main window
window = Tk()
window.geometry("634x500")
window.configure(bg="#FFFFFF")

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=500,
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
    500.0,
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
    "Determinant",
    "Inverse",
    "Addition",
    "Multiplication",
    "Echelon Form",
    "Is Unitary",
    "Normal Form",
    "PAQ Form",
    "Encode",
    "Decode"
]

# Create a dropdown menu for selecting matrix operations
selected_operation = StringVar(window)
selected_operation.set("Select Operation")

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

# Text area for matrix input
text_area = Text(window, height=10, width=50)
text_area.place(x=50, y=250)

# Label for instructions
instructions_label = Label(window, text="Enter matrix (row by row):", bg="#FFFFFF")
instructions_label.place(x=50, y=220)

# Button to confirm selection
button_1 = Button(
    text="Calculate",
    command=perform_operation,
    relief="flat"
)
button_1.place(
    x=276.0,
    y=460.0,
    width=83.0,
    height=41.0
)

window.resizable(False, False)

window.mainloop()
