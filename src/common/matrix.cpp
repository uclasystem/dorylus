#include "matrix.hpp"

Matrix::Matrix() {
    rows = 0; cols = 0;
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
void Matrix::setRows(unsigned _rows) { rows = _rows; }
void Matrix::setCols(unsigned _cols) { cols = _cols; }
void Matrix::setDims(unsigned _rows, unsigned _cols) { rows = _rows; cols = _cols; }
void Matrix::setData(FeatType *_data) { data = _data; }

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
    for (unsigned ui = 0; ui < getNumElemts(); ++ui) { data[ui] -= rhs;
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

// Print functions for info / debugging
std::string Matrix::shape() { return "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")"; }

std::string Matrix::str() {
    std::stringstream output;
    output << "Matrix Dims: " << shape() << "\n";
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
            output << std::fixed << std::setprecision(8) << data[i * cols + j] << " ";
        }
        output << "\n";
    }
    return output.str();
}

