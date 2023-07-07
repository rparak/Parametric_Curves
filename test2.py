import numpy as np

def generate_knot_vector(degree, num_ctrlpts, clamped):
    # Number of repetitions at the start and end of the array
    num_repeat = degree

    # Number of knots in the middle
    num_segments = num_ctrlpts - (degree + 1)

    if not clamped:
        # No repetitions at the start and end
        num_repeat = 0
        # Should conform the rule: m = n + p + 1
        num_segments = degree + num_ctrlpts - 1

    # First knots
    knot_vector = [0.0 for _ in range(0, num_repeat)]
    print(num_repeat)
    # Middle knots
    knot_vector += np.linspace(0.0, 1.0, num_segments + 2)

    # Last knots
    knot_vector += [1.0 for _ in range(0, num_repeat)]

    # Return auto-generated knot vector
    return knot_vector

# Example usage
n = 6  # Number of control points minus 1
degree = 3  # Degree of the B-spline

knot_vector = generate_knot_vector(degree, n, False)
print(knot_vector)
