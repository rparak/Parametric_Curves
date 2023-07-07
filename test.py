import numpy as np

def generate_open_uniform_vector_1(degree, num_control_points):
    num_knots = num_control_points + degree + 1
    num_internal_knots = num_knots - 2 * (degree + 1)

    # Calculate the number of internal knots (excluding the first and last ones)
    internal_knots = [0] * num_internal_knots

    # Calculate the values for internal knots
    for i in range(num_internal_knots):
        internal_knots[i] = (i + 1) / (num_internal_knots + 1)

    # Create the open uniform vector
    knot_vector = [0] * (degree + 1) + internal_knots + [1] * (degree + 1)

    return knot_vector

def generate_open_uniform_vector_2(n, m):
    u = np.zeros((m, 1), dtype=float)
    j = 1
    for i in range(m):
        if i <= n: 
            u[i] = 0.0
        elif i < m - (n+1): 
            u[i] = 1.0 / (m - 2*(n+1) + 1) * j
            j += 1
        else: 
            u[i] = 1.0 
    return u.flatten()

def generate_open_uniform_vector_3(degree, num_control_points):
    num_knots = num_control_points + degree + 1
    knot_vector = []

    # Generate the knot vector
    for i in range(num_knots):
        if i < degree + 1:
            knot_vector.append(0.0)
        elif i >= num_control_points:
            knot_vector.append(1.0)
        else:
            knot_vector.append((i - degree)/(num_control_points - degree))

    return knot_vector

# Example usage
degree = 3
num_control_points = 6
print(generate_open_uniform_vector_1(degree, num_control_points))
print(generate_open_uniform_vector_2(degree, num_control_points + degree + 1))
print(generate_open_uniform_vector_3(degree, num_control_points))

