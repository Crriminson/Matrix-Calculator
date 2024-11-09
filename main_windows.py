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
    """Compute the echelon form of the matrix with detailed steps."""
    steps = ["Computing Echelon Form:"]
    augmented_matrix = matrix.copy()
    rows, cols = augmented_matrix.shape
    for i in range(min(rows, cols)):
        # Find the pivot
        max_row = np.argmax(np.abs(augmented_matrix[i:, i])) + i
        if augmented_matrix[max_row, i] == 0:
            steps.append(f"No pivot in column {i+1}, moving to next column.")
            continue
        # Swap rows
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]
        steps.append(f"Swapped row {i+1} with row {max_row+1}:\n{augmented_matrix}")
        # Make pivot 1
        pivot = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / pivot
        steps.append(f"Normalized row {i+1} by dividing by pivot {pivot}:\n{augmented_matrix}")
        # Eliminate below
        for j in range(i+1, rows):
            factor = augmented_matrix[j, i]
            augmented_matrix[j] = augmented_matrix[j] - factor * augmented_matrix[i]
            steps.append(f"Eliminated element at row {j+1}, column {i+1} by subtracting {factor} * row {i+1}:\n{augmented_matrix}")
    steps.append("Echelon form achieved.")
    return augmented_matrix, "\n".join(steps)

def rref(matrix):
    """Compute the Reduced Row Echelon Form (RREF) with detailed steps."""
    steps = ["Converting to Reduced Row Echelon Form:"]
    rref_matrix = matrix.copy().astype(float)
    rows, cols = rref_matrix.shape
    lead = 0
    for r in range(rows):
        if lead >= cols:
            break
        i = r
        while rref_matrix[i, lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if lead == cols:
                    steps.append("RREF achieved.")
                    return rref_matrix, "\n".join(steps)
        rref_matrix[[i, r]] = rref_matrix[[r, i]]
        steps.append(f"Swapped row {i+1} with row {r+1}:\n{rref_matrix}")
        lv = rref_matrix[r, lead]
        rref_matrix[r] = rref_matrix[r] / lv
        steps.append(f"Normalized row {r+1} by dividing by leading value {lv}:\n{rref_matrix}")
        for i in range(rows):
            if i != r:
                lv = rref_matrix[i, lead]
                rref_matrix[i] = rref_matrix[i] - lv * rref_matrix[r]
                steps.append(f"Eliminated element at row {i+1}, column {lead+1} by subtracting {lv} * row {r+1}:\n{rref_matrix}")
        lead +=1
    steps.append("Reduced Row Echelon Form achieved.")
    return rref_matrix, "\n".join(steps)

def normal_form(matrix):
    """Compute the normal form of the matrix with detailed steps."""
    steps = ["Computing Normal Form using RREF:"]
    rref_matrix, rref_steps = rref(matrix)
    steps.append(rref_steps)
    steps.append(f"Normal form of the matrix is:\n{rref_matrix}")
    return rref_matrix, "\n".join(steps)

def paq_form(matrix):
    """Compute the PAQ form of the matrix with detailed steps."""
    steps = ["Computing PAQ Form:"]
    P, L, U = lu(matrix)
    steps.append(f"Computed LU decomposition:\nP =\n{P}\nL =\n{L}\nU =\n{U}")
    PAQ_matrix = P @ matrix @ U
    steps.append(f"PAQ form of the matrix is:\n{PAQ_matrix}")
    return PAQ_matrix, "\n".join(steps)

def is_unitary(matrix):
    """Check if the matrix is unitary with detailed steps."""
    steps = ["Checking if the matrix is Unitary:"]
    identity = np.eye(matrix.shape[0])
    product = matrix @ matrix.T.conj()
    steps.append(f"Computed matrix * matrix.T.conj():\n{product}")
    unitary = np.allclose(identity, product)
    if unitary:
        result = "Unitary"
        steps.append("The matrix is unitary.")
    else:
        result = "Not Unitary"
        steps.append("The matrix is not unitary.")
    return result, "\n".join(steps)

def is_orthogonal(matrix):
    """Check if the matrix is orthogonal with detailed steps."""
    steps = ["Checking if the matrix is Orthogonal:"]
    identity = np.eye(matrix.shape[0])
    product = matrix @ matrix.T
    steps.append(f"Computed matrix * matrix.T:\n{product}")
    orthogonal = np.allclose(identity, product)
    if orthogonal:
        result = "Orthogonal"
        steps.append("The matrix is orthogonal.")
    else:
        # Attempt to solve for unknowns
        steps.append("The matrix is not orthogonal. Attempting to solve for unknowns:")
        unknowns = symbols('a b c')  # Add more symbols as needed
        equations = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i == j:
                    equations.append(Eq(matrix[i, j] ** 2, 1))
                else:
                    equations.append(Eq(matrix[i, j] * matrix[j, i], 0))
        solutions = solve(equations, unknowns)
        if solutions:
            result = f"Orthogonal with solutions: {solutions}"
            steps.append(f"Solutions found: {solutions}")
        else:
            result = "Not Orthogonal"
            steps.append("No solutions found. The matrix is not orthogonal.")
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
    # Detailed encoding steps
    steps.append("Step 1: Validating dimensions of Matrix1 and Matrix2.")
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError("Number of columns in Matrix1 must equal number of rows in Matrix2.")
    steps.append(f"Matrix1 dimensions: {matrix1.shape}")
    steps.append(f"Matrix2 dimensions: {matrix2.shape}")
    
    steps.append("Step 2: Performing matrix multiplication (Matrix1 @ Matrix2).")
    encoded_matrix = matrix1 @ matrix2  # Placeholder for actual encoding logic
    steps.append(f"Result of multiplication:\n{encoded_matrix}")
    
    steps.append("Step 3: Encoding completed successfully.")
    return encoded_matrix, "\n".join(steps)

def decode(matrix1, matrix2):
    """Decode the matrix using another matrix."""
    steps = ["Decoding Matrix:"]
    # Detailed decoding steps
    steps.append("Step 1: Validating dimensions of Matrix1 and Matrix2.")
    if matrix2.shape[1] != matrix1.shape[0]:
        raise ValueError("Number of columns in Matrix2 must equal number of rows in Matrix1.")
    steps.append(f"Matrix1 dimensions: {matrix1.shape}")
    steps.append(f"Matrix2 dimensions: {matrix2.shape}")
    
    steps.append("Step 2: Calculating the inverse of Matrix2.")
    try:
        inverse_matrix2 = np.linalg.inv(matrix2)
        steps.append(f"Inverse of Matrix2:\n{inverse_matrix2}")
    except np.linalg.LinAlgError:
        raise ValueError("Matrix2 is singular and cannot be inverted.")
    
    steps.append("Step 3: Performing matrix multiplication (Matrix1 @ Inverse(Matrix2)).")
    decoded_matrix = matrix1 @ inverse_matrix2  # Placeholder for actual decoding logic
    steps.append(f"Result of multiplication:\n{decoded_matrix}")
    
    steps.append("Step 4: Decoding completed successfully.")
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

def transpose(matrix):
    """Compute the transpose of the matrix with detailed steps."""
    steps = ["Computing Transpose of the matrix:"]
    transpose_matrix = matrix.T
    steps.append(f"Transpose of the matrix is:\n{transpose_matrix}")
    return transpose_matrix, "\n".join(steps)

# Perform matrix operation
def perform_operation():
    operation = selected_operation.get()
    try:
        # Get main matrix values from entry fields
        matrix1 = np.array([[float(entry.get()) for entry in row] for row in entries_matrix])
        
        # Define matrix2 if the operation requires a second matrix
        if operation in ["Encode", "Decode", "Matrix Multiplication", "Matrix Addition"]:
            matrix2 = np.array([[float(entry.get()) for entry in row] for row in secondary_entries_matrix])
        
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
