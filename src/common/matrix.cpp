#include "matrix.hpp"

Matrix::Matrix() {
    rows = 0; cols = 0;
}

Matrix::Matrix(const char* _name, unsigned _rows, unsigned _cols) {
    tensorName = _name;
    rows = _rows; cols = _cols;
}

Matrix::Matrix(const char* _name, unsigned _rows, unsigned _cols, FeatType *_data) {
    tensorName = _name;
    rows = _rows; cols = _cols; data = _data;
}

Matrix::Matrix(unsigned _rows, unsigned _cols) {
    rows = _rows; cols = _cols;
}

Matrix::Matrix(unsigned _rows, unsigned _cols, FeatType *_data) {
    rows = _rows; cols = _cols; data = _data;
}

Matrix::Matrix(unsigned _rows, unsigned _cols, char *_data) {
    rows = _rows; cols = _cols; data = (FeatType *) _data;
}

// Get Matrix data / information
std::string Matrix::name() { return tensorName; }
unsigned Matrix::getRows() { return rows; }
unsigned Matrix::getCols() { return cols; }
unsigned Matrix::getNumElemts() { return rows * cols; }
size_t Matrix::getDataSize() const { return rows * cols * sizeof(FeatType); }
FeatType* Matrix::getData() const { return data; }

// Get a specific element in the matrix
FeatType Matrix::get(unsigned row, unsigned col) { return data[row * cols + col]; }

// Get a full row in the matrix
// Just returns a pointer to the start of the row (no size information etc)
FeatType* Matrix::get(unsigned row) { return data + (row * cols); }

// Setting Matrix info (not sure when this would be used)
void Matrix::setName(const char* _name) { tensorName = _name; }
void Matrix::setRows(unsigned _rows) { rows = _rows; }
void Matrix::setCols(unsigned _cols) { cols = _cols; }
void Matrix::setDims(unsigned _rows, unsigned _cols) { rows = _rows; cols = _cols; }
void Matrix::setData(FeatType *_data) { data = _data; }

void Matrix::free() {
    if (data) {
        delete[] data;
        data = NULL;
    }
}

bool Matrix::empty() { return rows == 0 || cols == 0; }


Matrix Matrix::operator*(float rhs) {
    FeatType* result = new FeatType[getNumElemts()];

    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        result[ui] = data[ui] * rhs;
    }

    return Matrix(rows, cols, result);
}

Matrix operator*(float lhs, Matrix& rhs) {
    FeatType* result = new FeatType[rhs.getNumElemts()];

    FeatType* rhsData = rhs.getData();
    for (unsigned ui = 0; ui < rhs.getNumElemts(); ++ui) {
        result[ui] = rhsData[ui] * lhs;
    }

    return Matrix(rhs.getRows(), rhs.getCols(), result);
}

void Matrix::operator*=(float rhs) {
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        data[ui] *= rhs;
    }
}

Matrix Matrix::operator/(float rhs) {
    FeatType* result = new FeatType[getNumElemts()];

    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        result[ui] = data[ui] / rhs;
    }

    return Matrix(rows, cols, result);

}

void Matrix::operator/=(float rhs) {
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        data[ui] /= rhs;
    }
}

// Addition by float
Matrix Matrix::operator+(float rhs) {
    FeatType* result = new FeatType[getNumElemts()];

    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        result[ui] = data[ui] + rhs;
    }

    return Matrix(rows, cols, result);
}

Matrix operator+(float lhs, Matrix& rhs) {
    FeatType* result = new FeatType[rhs.getNumElemts()];

    FeatType* rhsData = rhs.getData();
    for (unsigned ui = 0; ui < rhs.getNumElemts(); ++ui) {
        result[ui] = rhsData[ui] + lhs;
    }

    return Matrix(rhs.getRows(), rhs.getCols(), result);
}

void Matrix::operator+=(float rhs) {
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        data[ui] += rhs;
    }
}

Matrix Matrix::operator-(float rhs) {
    FeatType* result = new FeatType[getNumElemts()];

    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        result[ui] = data[ui] - rhs;
    }

    return Matrix(rows, cols, result);

}

void Matrix::operator-=(float rhs) {
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        data[ui] -= rhs;
    }
}

Matrix Matrix::operator^(float rhs) {
    FeatType* result = new FeatType[getNumElemts()];

    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        result[ui] = std::pow(data[ui], rhs);
    }

    return Matrix(rows, cols, result);

}

void Matrix::operator^=(float rhs) {
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        data[ui] = std::pow(data[ui], rhs);
    }
}

Matrix Matrix::operator*(Matrix& M) {
    assert(rows == M.getRows());
    assert(cols == M.getCols());

    FeatType* result = new FeatType[rows * cols];

    FeatType* MData = M.getData();
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        result[ui] = data[ui] * MData[ui];
    }

    return Matrix(rows, cols, result);
}

Matrix Matrix::operator/(Matrix& M) {
    assert(rows == M.getRows());
    assert(cols == M.getCols());

    FeatType* result = new FeatType[rows * cols];

    FeatType* MData = M.getData();
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        result[ui] = data[ui] / MData[ui];
    }

    return Matrix(rows, cols, result);
}

Matrix Matrix::operator+(Matrix& M) {
    assert(rows == M.getRows());
    assert(cols == M.getCols());

    FeatType* result = new FeatType[rows * cols];

    FeatType* MData = M.getData();
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        result[ui] = data[ui] + MData[ui];
    }

    return Matrix(rows, cols, result);
}

Matrix Matrix::operator-(Matrix& M) {
    assert(rows == M.getRows());
    assert(cols == M.getCols());

    FeatType* result = new FeatType[rows * cols];

    FeatType* MData = M.getData();
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        result[ui] = data[ui] - MData[ui];
    }

    return Matrix(rows, cols, result);
}

void Matrix::operator*=(Matrix& M) {
    assert(rows == M.getRows());
    assert(cols == M.getCols());

    FeatType* MData = M.getData();
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        data[ui] *= MData[ui];
    }
}

void Matrix::operator/=(Matrix& M) {
    assert(rows == M.getRows());
    assert(cols == M.getCols());

    FeatType* MData = M.getData();
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        data[ui] /= MData[ui];
    }
}

void Matrix::operator+=(Matrix& M) {
    assert(rows == M.getRows());
    assert(cols == M.getCols());

    FeatType* MData = M.getData();
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        data[ui] += MData[ui];
    }
}

void Matrix::operator-=(Matrix& M) {
    assert(rows == M.getRows());
    assert(cols == M.getCols());

    FeatType* MData = M.getData();
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) {
        data[ui] -= MData[ui];
    }
}

Matrix Matrix::dot(Matrix& M, bool transpose1, bool transpose2, float scale) {
    unsigned m = 0, k = 0, n = 0;
    CBLAS_TRANSPOSE cblasTrans1 = CblasNoTrans, cblasTrans2 = CblasNoTrans;
    FeatType* result;

    // Depending on transposed matrices, check dimension alignment and assign
    // correct values
    // NOTE: Annoying I have to make a separate call to cblas per case becuase the
    //  inputs vary slightly... there is probably a better way to do this

    // Case 1 - Neither transposed
    if (!transpose1 && !transpose2) {
        m = getRows(), k = getCols(), n = M.getCols();
        assert(k == M.getRows());

        result = new FeatType[m * n];
        cblas_sgemm(CblasRowMajor, cblasTrans1, cblasTrans2, m, n, k, scale,
                    getData(), k, M.getData(), n, 0.0, result, n);

    // Case 2 - Both transposed
    } else if (transpose1 && transpose2) {
        m = getCols(), k = getRows(), n = M.getRows();
        assert(k == M.getCols());
        cblasTrans1 = CblasTrans;
        cblasTrans2 = CblasTrans;

        result = new FeatType[m * n];
        cblas_sgemm(CblasRowMajor, cblasTrans1, cblasTrans2, m, n, k, scale,
                    getData(), m, M.getData(), k, 0.0, result, n);

    // Case 3 - Left Matrix transposed
    } else if (transpose1) {
        m = getCols(), k = getRows(), n = M.getCols();
        assert(k == M.getRows());
        cblasTrans1 = CblasTrans;

        result = new FeatType[m * n];
        cblas_sgemm(CblasRowMajor, cblasTrans1, cblasTrans2, m, n, k, scale,
                    getData(), m, M.getData(), n, 0.0, result, n);

    // Case 4 - Right Matrix transposed
    } else if (transpose2) {
        m = getRows(), k = getCols(), n = M.getRows();
        assert(k == M.getCols());
        cblasTrans2 = CblasTrans;

        result = new FeatType[m * n];
        cblas_sgemm(CblasRowMajor, cblasTrans1, cblasTrans2, m, n, k, scale,
                    getData(), k, M.getData(), k, 0.0, result, n);
    }

    return Matrix(m, n, result);
}

// Print functions for info / debugging
float Matrix::sum() {
    float sum = 0;
    unsigned numElemts = getNumElemts();
    for (unsigned u = 0; u < numElemts; ++u) {
        sum += std::fabs(data[u]);
    }

    return sum;
}

std::string Matrix::shape() { return "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")"; }

// std::string Matrix::str() {
//     std::stringstream output;
//     output << "Matrix " << tensorName << "\n";
//     output << "Dims: " << shape() << "\n";
//     for (unsigned i = 0; i < rows; ++i) {
//         for (unsigned j = 0; j < cols; ++j) {
//             output << std::fixed << std::setprecision(8) << data[i * cols + j] << " ";
//         }
//         output << "\n";
//     }
//     return output.str();
// }

std::string Matrix::str() {
    std::stringstream output;
    output << "Sum: " << sum() << "\n";
    output << "Dims: " << shape() << "\n";
    unsigned row_part1 = std::min(rows, 3u);
    unsigned row_part2 = std::max(row_part1 + 3, rows) - 3;

    unsigned col_part1 = std::min(cols, 3u);
    unsigned col_part2 = std::max(col_part1 + 3, cols) - 3;

    for (unsigned i = 0; i < row_part1; ++i) {
        for (unsigned j = 0; j < col_part1; ++j) {
            output << std::fixed << std::setprecision(8) << data[i * cols + j] << " ";
        }
        if (col_part2 > col_part1)
            output << "... ";
        for (unsigned j = col_part2; j < cols; ++j) {
            output << std::fixed << std::setprecision(8) << data[i * cols + j] << " ";
        }
        output << "\n";
    }
    if (row_part2 > row_part1) {
        output << "...\n";
    }
    for (unsigned i = row_part2; i < rows; ++i) {
        for (unsigned j = 0; j < col_part1; ++j) {
            output << std::fixed << std::setprecision(8) << data[i * cols + j] << " ";
        }
        if (col_part2 > col_part1)
            output << "... ";
        for (unsigned j = col_part2; j < cols; ++j) {
            output << std::fixed << std::setprecision(8) << data[i * cols + j] << " ";
        }
        output << "\n";
    }
    return output.str();
}

std::string Matrix::signature() {
    std::stringstream output;
    output << "Matrix: " << tensorName << "\n";
    output << "Sum: " << sum() << "\n";

    return output.str();
}


void Matrix::toFile(std::string filename) {
    std::ofstream output(filename, std::ios::out | std::ios::binary);
    output.write((char*)&rows, sizeof(rows));
    output.write((char*)&cols, sizeof(cols));
    output.write((char*)data, getDataSize());
}

void Matrix::fromFile(std::string filename) {
    std::ifstream input(filename, std::ios::in | std::ios::binary);
    input.read((char*)&rows, sizeof(rows));
    input.read((char*)&cols, sizeof(cols));
    data = new FeatType[rows * cols];
    input.read((char*)data, rows * cols * sizeof(FeatType));
}