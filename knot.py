import numpy as np

# If the knot vector is from 0 to 1 is normalized

def Generate_Chord_Length_Knot_Vector(num_control_points, degree, control_points):
    num_knots = num_control_points + degree + 1
    knot_vector = np.zeros(num_knots)

    # Calculate chord lengths
    chord_lengths = np.zeros(num_control_points)
    for i in range(1, num_control_points):
        chord_lengths[i] = np.linalg.norm(control_points[i] - control_points[i - 1])

    # Calculate knot vector
    total_length = np.sum(chord_lengths)
    knot_vector[degree+1] = 0.0
    for i in range(degree + 2, num_control_points):
        knot_vector[i] = knot_vector[i-1] + chord_lengths[i-1] / total_length
    knot_vector[num_control_points+degree] = 1.0

    return knot_vector

def Generate_Uniform_Knot_Vector(degree, num_control_points):
    num_knots = num_control_points + degree + 1
    knot_vector = np.linspace(0.0, 1.0, num_knots)

    return knot_vector

def Generate_Open_Uniform_2_Knot_vector(degree, num_control_points):
    num_knots = num_control_points + degree + 1
    knot_vector = np.zeros(num_knots)

    # Generate open-uniform knot vector
    knot_span = 1.0 / (num_control_points - degree)
    for i in range(num_knots):
        if i <= degree:
            knot_vector[i] = 0.0
        elif i >= num_control_points:
            knot_vector[i] = 1.0
        else:
            knot_vector[i] = (i - degree) * knot_span

    return knot_vector

def Generate_Open_Uniform_Knot_Vector(degree, num_control_points):
    num_knots = num_control_points + degree + 1

    # Generate the knot vector
    knot_vector = []; j = 1
    for i in range(num_knots):
        if i <= degree:
            knot_vector.append(0.0)
        elif i <= num_control_points:
            knot_vector.append(j / (num_control_points - degree + 1))
            j += 1
        else:
            knot_vector.append(1.0)

    return knot_vector

def Generate_Centripetal_Knot_Vector(num_control_points, degree):
    num_knots = num_control_points + degree + 1
    knot_vector = np.zeros(num_knots)

    # Calculate parameter values
    alpha = 0.5  # Centripetal parameter
    t = np.linspace(0, 1, num_control_points)
    t = np.power(t, alpha)

    # Calculate knots
    for i in range(num_knots):
        if i < degree + 1:
            knot_vector[i] = 0.0
        elif i >= num_control_points:
            knot_vector[i] = 1.0
        else:
            knot_vector[i] = np.sum(t[i-degree:i]) / degree

    return knot_vector

degree = 3
num_control_points = 6
print(Generate_Open_Uniform_Knot_Vector(degree, num_control_points))
print(Generate_Uniform_Knot_Vector(degree, num_control_points))
print(Generate_Open_Uniform_2_Knot_vector(degree, num_control_points))
#knot_vector = Generate_Centripetal_Knot_Vector(num_control_points, degree)
#print(knot_vector)
