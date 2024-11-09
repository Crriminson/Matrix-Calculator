from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, StringVar, OptionMenu, messagebox
import numpy as np
from scipy.linalg import lu
from sympy import symbols, Eq, solve

# Paths
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets/frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Matrix operation functions with steps
def check_consistency(matrix, constants=None):
    """Check if the system of equations is consistent."""
    steps = ["Checking Consistency of System of Equations:"]
    augmented_matrix = np.hstack((matrix, constants.reshape(-1, 1))) if constants is not None else matrix
    rank_coeff = np.linalg.matrix_rank(matrix)
    rank_augmented = np.linalg.matrix_rank(augmented_matrix)
    steps.append(f"Rank of coefficient matrix: {rank_coeff}")
    steps.append(f"Rank of augmented matrix: {rank_augmented}")

    if rank_coeff == rank_augmented:
        if rank_coeff == matrix.shape[1]:  # Unique solution case
            steps.append("System is consistent with a unique solution.")
            result = "Consistent with unique solution"
        else:  # Infinite solutions
            steps.append("System is consistent with infinitely many solutions.")
            result = "Consistent with infinitely many solutions"
    else:
        steps.append("System is inconsistent (no solutions).")
        result = "Inconsistent"
    return result, "\n".join(steps)

def check_homogeneous_consistency(matrix):
    """Check if the homogeneous system has only the trivial solution or non-trivial solutions."""
    steps = ["Checking Consistency of Homogeneous Equations:"]
    rank = np.linalg.matrix_rank(matrix)
    num_columns = matrix.shape[1]
    steps.append(f"Rank of matrix: {rank}")

    if rank == num_columns:
        steps.append("Homogeneous system has only the trivial solution.")
        result = "Only trivial solution"
    else:
        steps.append("Homogeneous system has non-trivial solutions.")
        result = "Non-trivial solutions exist"
    return result, "\n".join(steps)

def linear_dependence_independence(matrix):
    """Check if the columns of the matrix are linearly dependent or independent."""
    steps = ["Checking Linear Dependence/Independence of Columns:"]
    rank = np.linalg.matrix_rank(matrix)
    num_columns = matrix.shape[1]
    steps.append(f"Rank of matrix: {rank}")

    if rank == num_columns:
        steps.append("Columns are linearly independent.")
        result = "Linearly Independent"
    else:
        steps.append("Columns are linearly dependent.")
        result = "Linearly Dependent"
    return result, "\n".join(steps)

def echelon_form(matrix):
    """Compute the echelon form of the matrix."""
    steps = ["Computing Echelon Form:"]
    echelon_matrix = np.linalg.matrix_rank(matrix)  # Placeholder for actual echelon form computation
    rank = np.linalg.matrix_rank(matrix)
    steps.append(f"Rank of matrix: {rank}")
    steps.append(f"Echelon form of the matrix is:\n{echelon_matrix}")
    return echelon_matrix, "\n".join(steps)

def is_unitary(matrix):
    """Check if the matrix is unitary."""
    steps = ["Checking if the matrix is Unitary:"]
    unitary = np.allclose(np.eye(matrix.shape[0]), matrix @ matrix.T.conj())
    result = "Unitary" if unitary else "Not Unitary"
    steps.append(result)
    return result, "\n".join(steps)

def normal_form(matrix):
    """Compute the normal form of the matrix."""
    steps = ["Computing Normal Form:"]
    normal_matrix = matrix  # Placeholder for actual normal form computation
    steps.append(f"Normal form of the matrix is:\n{normal_matrix}")
    return normal_matrix, "\n".join(steps)

def paq_form(matrix):
    """Compute the PAQ form of the matrix."""
    steps = ["Computing PAQ Form:"]
    P, L, U = lu(matrix)
    PAQ_matrix = P @ matrix @ U  # Placeholder for actual PAQ form computation
    rank = np.linalg.matrix_rank(matrix)
    steps.append(f"Rank of matrix: {rank}")
    steps.append(f"PAQ form of the matrix is:\n{PAQ_matrix}")
    return PAQ_matrix, "\n".join(steps)

def transpose(matrix):
    """Compute the transpose of the matrix."""
    steps = ["Computing Transpose:"]
    transposed_matrix = matrix.T
    steps.append(f"Transpose of the matrix is:\n{transposed_matrix}")
    return transposed_matrix, "\n".join(steps)

def is_orthogonal(matrix):
    """Check if the matrix is orthogonal and solve for unknowns if any."""
    steps = ["Checking if the matrix is Orthogonal:"]
    unknowns = symbols('a b c')  # Add more symbols if needed
    orthogonal = np.allclose(np.eye(matrix.shape[0]), matrix @ matrix.T)
    
    if orthogonal:
        result = "Orthogonal"
        steps.append(result)
    else:
        # Attempt to solve for unknowns
        equations = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i == j:
                    equations.append(Eq(matrix[i, j] * matrix[i, j], 1))
                else:
                    equations.append(Eq(matrix[i, j] * matrix[j, i], 0))
        solutions = solve(equations, unknowns)
        if solutions:
            result = f"Orthogonal with solutions: {solutions}"
            steps.append(result)
        else:
            result = "Not Orthogonal"
            steps.append(result)
    return result, "\n".join(steps)

def inverse(matrix):
    """Compute the inverse of the matrix."""
    steps = ["Computing Inverse:"]
    inverse_matrix = np.linalg.inv(matrix)
    steps.append(f"Inverse of the matrix is:\n{inverse_matrix}")
    return inverse_matrix, "\n".join(steps)

def determinant(matrix):
    """Compute the determinant of the matrix."""
    steps = ["Computing Determinant:"]
    det = np.linalg.det(matrix)
    steps.append(f"Determinant of the matrix is:\n{det}")
    return det, "\n".join(steps)

def conjugate(matrix):
    """Compute the conjugate of the matrix."""
    steps = ["Computing Conjugate:"]
    conjugate_matrix = np.conjugate(matrix)
    steps.append(f"Conjugate of the matrix is:\n{conjugate_matrix}")
    return conjugate_matrix, "\n".join(steps)

def encode(matrix1, matrix2):
    """Encode the matrix using another matrix."""
    steps = ["Encoding Matrix:"]
    encoded_matrix = matrix1 @ matrix2  # Placeholder for actual encoding logic
    steps.append(f"Encoded matrix is:\n{encoded_matrix}")
    return encoded_matrix, "\n".join(steps)

def decode(matrix1, matrix2):
    """Decode the matrix using another matrix."""
    steps = ["Decoding Matrix:"]
    decoded_matrix = matrix1 @ np.linalg.inv(matrix2)  # Placeholder for actual decoding logic
    steps.append(f"Decoded matrix is:\n{decoded_matrix}")
    return decoded_matrix, "\n".join(steps)

def multiply_matrices(matrix1, matrix2):
    """Multiply two matrices."""
    steps = ["Multiplying Matrices:"]
    product_matrix = matrix1 @ matrix2
    steps.append(f"Product of the matrices is:\n{product_matrix}")
    return product_matrix, "\n".join(steps)

def add_matrices(matrix1, matrix2):
    """Add two matrices."""
    steps = ["Adding Matrices:"]
    sum_matrix = matrix1 + matrix2
    steps.append(f"Sum of the matrices is:\n{sum_matrix}")
    return sum_matrix, "\n".join(steps)

# Perform matrix operation
def perform_operation():
    operation = selected_operation.get()
    try:
        # Get main matrix values from entry fields
        matrix1 = np.array([[float(entry.get()) for entry in row] for row in entries_matrix])

        result, steps = None, ""
        if operation == "Test Consistency of Equations":
            # Retrieve the constants column if present for non-homogeneous equations
            constants = np.array([float(entry.get()) for entry in constants_entries])
            result, steps = check_consistency(matrix1, constants)
        elif operation == "Test Consistency of Homogeneous Equations":
            result, steps = check_homogeneous_consistency(matrix1)
        elif operation == "Linear Dependence/Independence":
            result, steps = linear_dependence_independence(matrix1)
        # Other operations
        elif operation == "Echelon Form":
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

        # Display the result with steps
        steps_text.delete("1.0", "end")
        steps_text.insert("1.0", f"Result:\n{result}\n\nSteps:\n{steps}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Update entry fields based on matrix size
def update_matrix_size(*args):
    global entries_matrix, secondary_entries_matrix, constants_entries, encryption_entries_matrix
    for row in entries_matrix:
        for entry in row:
            entry.place_forget()
    for row in secondary_entries_matrix:
        for entry in row:
            entry.place_forget()
    for entry in constants_entries:
        entry.place_forget()
    for row in encryption_entries_matrix:
        for entry in row:
            entry.place_forget()

    size = int(selected_size.get()[0])
    entries_matrix = []
    secondary_entries_matrix = []
    constants_entries = []
    encryption_entries_matrix = []

    # Main matrix entries
    for i in range(size):
        row_entries = []
        for j in range(size):
            entry = Entry(window, bd=0, bg="#D9D9D9", fg="#000716", highlightthickness=0, justify='center')
            entry.place(x=150 + j * 60, y=120 + i * 40, width=50, height=30)
            row_entries.append(entry)
        entries_matrix.append(row_entries)

    # Constants column entries for consistency check
    for i in range(size):
        entry = Entry(window, bd=0, bg="#F0D9FF", fg="#000716", highlightthickness=0, justify='center')
        entry.place(x=450, y=120 + i * 40, width=50, height=30)
        constants_entries.append(entry)

    # Secondary matrix entries
    for i in range(size):
        row_entries = []
        for j in range(size):
            entry = Entry(window, bd=0, bg="#E0E0E0", fg="#000716", highlightthickness=0, justify='center')
            entry.place(x=550 + j * 60, y=120 + i * 40, width=50, height=30)
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

# Toggle matrix visibility based on selected operation
def toggle_secondary_matrix(*args):
    if selected_operation.get() in ["Matrix Multiplication", "Matrix Addition", "Encode", "Decode"]:
        for row in secondary_entries_matrix:
            for entry in row:
                entry.place(x=550 + row.index(entry) * 60, y=120 + secondary_entries_matrix.index(row) * 40, width=50, height=30)
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
    "Decode", "Matrix Multiplication", "Matrix Addition", "Linear Dependence/Independence",
    "Test Consistency of Equations", "Test Consistency of Homogeneous Equations"
]
operation_dropdown = OptionMenu(window, selected_operation, *matrix_operations)
operation_dropdown.place(x=350, y=70)
selected_operation.trace("w", toggle_secondary_matrix)

# Matrix entry fields
entries_matrix = []
secondary_entries_matrix = []
constants_entries = []
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
