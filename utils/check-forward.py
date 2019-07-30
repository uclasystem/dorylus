
import numpy as np
from scipy.linalg import fractional_matrix_power


weights = [
    np.array([
        [-1.29132,  -1.17642,  0.960311,  -0.0528824],
        [-0.794382, -1.17226,  -1.10235,  0.748811],
        [0.259365,  0.151377,  0.199394,  0.218145],
        [0.359989,  -0.669245, -0.998963, 1.42596],
        [-0.972823, -0.235563, 0.894912,  -1.21858],
        [0.346023,  -1.3924,   0.973205,  0.658695]
    ]),
    np.array([
        [0.693969,  -0.471074],
        [-0.348153, -1.40479],
        [-0.293149, -0.95399],
        [1.29462,   -0.25147]
    ])
]

input_feats = np.array([
    [0, 1, 1, 1],
    [1, 1, 1, 1],
    [2, 1, 1, 1],
    [3, 1, 1, 1],
    [4, 1, 1, 1],
    [5, 1, 1, 1]
])

adj_mat = np.array([
    [1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [1, 0, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 1]
])

deg_mat = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0],
    [0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 3],
])


def activate(mat):
    return np.tanh(mat)


normed_deg_mat = fractional_matrix_power(deg_mat, -0.5)
S_mat = np.dot(normed_deg_mat, np.dot(adj_mat, normed_deg_mat))

print("S_mat: ")
print(S_mat)


activation_mat1 = activate(np.dot(S_mat, np.dot(input_feats    , weights[0])))
activation_mat2 = activate(np.dot(S_mat, np.dot(activation_mat1, weights[1])))

print("Activation_mats: ")
print(activation_mat1)
print(activation_mat2)
