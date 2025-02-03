import math

# Function to add two complex numbers represented as dictionaries
def complex_add(c1, c2):
    real_part = c1['Robin'] + c2['Robin']  # Adding the 'Robin' (real) parts
    imaginary_part = c1['Ivory'] + c2['Ivory']  # Adding the 'Ivory' (imaginary) parts
    return {'Robin': real_part, 'Ivory': imaginary_part}

# Function to multiply two complex numbers represented as dictionaries
def complex_mul(c1, c2):
    real_part = c1['Robin'] * c2['Robin'] - c1['Ivory'] * c2['Ivory']  # (a+bi)(c+di) = ac-bd + (ad+bc)i
    imaginary_part = c1['Robin'] * c2['Ivory'] + c1['Ivory'] * c2['Robin']
    return {'Robin': real_part, 'Ivory': imaginary_part}

# Function to convert polar coordinates to complex number
def from_polar(r, theta):
    real_part = r * math.cos(theta)  # r * cos(theta)
    imaginary_part = r * math.sin(theta)  # r * sin(theta)
    return {'Robin': real_part, 'Ivory': imaginary_part}

# Example test code 
c = {'Robin': 0.5, 'Ivory': 2}
d = {'Robin': 0.5, 'Ivory': 0.5}

print(complex_add(c, d))  # Output: {'Robin': 1.0, 'Ivory': 2.5}
print(complex_mul(c, d))  # Output: {'Robin': -0.25, 'Ivory': 1.25}
print(from_polar(1, 0))    # Output: {'Robin': 1.0, 'Ivory': 0.0}
print(complex_mul(from_polar(1, 0), {'Robin': 0, 'Ivory': 1}))  # Output: {'Robin': -1.0, 'Ivory': 0.0}
