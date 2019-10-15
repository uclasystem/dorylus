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

Matrix Matrix::dotT(Matrix& M) {
    assert(getRows() == M.getRows());

    unsigned colsThis = getCols();
    unsigned colsM = M.getCols();
    FeatType* result = new FeatType[colsThis * colsM];

    // For the number of rows in the matrices
    for (unsigned r = 0; r < getRows(); ++r) {
        // comment these 2 lines to make compiler happy.
        // FeatType* rowThis = get(r);
        // FeatType* rowM = M.get(r);

        // For each number in the first row
        for (unsigned cThis = 0; cThis < colsThis; ++cThis) {
            // For each number in the second row
            for (unsigned cM = 0; cM < colsM; ++cM) {
                // Sum the product of the number at index cThis,cM
                result[cThis * colsM + cM] += get(r, cThis) * M.get(r, cM);
            }
        }
    }

    return Matrix(colsThis, colsM, result);
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
