#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <cassert>
#include <cmath>
#include <memory>

#include "cblas.h"

#include "utils.hpp"

struct EdgeTensor {
    unsigned numLvids;
    unsigned numRvids;
    unsigned featDim;
    unsigned numEdges;

    unsigned* edgeMapping;
    FeatType* chunkData;
};

struct EdgeInfo {
    unsigned numLvids;
    unsigned nChunkEdges;

    unsigned long long* edgePtrs;
};

/**
 *
 * Struct for a matrix.
 *
 */
class Matrix {
public:
    Matrix();
    Matrix(const char* _name, unsigned _rows, unsigned _cols);
    Matrix(const char* _name, unsigned _rows, unsigned _cols, FeatType *_data);
    Matrix(unsigned _rows, unsigned _cols);
    Matrix(unsigned _rows, unsigned _cols, FeatType *_data);
    Matrix(unsigned _rows, unsigned _cols, char *_data);

    std::string name();
    unsigned getRows();
    unsigned getCols();
    unsigned getNumElemts();
    FeatType *getData() const;
    size_t getDataSize() const;

    // Get a specific element in the matrix
    FeatType get(unsigned row, unsigned col);

    // Get a full row in the matrix
    // Just returns a pointer to the start of the row (no size information etc)
    FeatType* get(unsigned row);

    void setName(const char* _name);
    void setRows(unsigned _rows);
    void setCols(unsigned _cols);
    void setDims(unsigned _rows, unsigned _cols);
    void setData(FeatType *_data);
    void free();

    bool empty();

    // Multiply every element by some float
    Matrix operator*(float rhs);
    friend Matrix operator*(float lhs, Matrix& rhs);
    void operator*=(float rhs);

    // Divide every element by some float
    Matrix operator/(float rhs);
    void operator/=(float rhs);

    // Adding some float to every element
    Matrix operator+(float rhs);
    friend Matrix operator+(float lhs, Matrix& rhs);
    void operator+=(float rhs);

    // Adding some float to every element
    Matrix operator-(float rhs);
    void operator-=(float rhs);

    Matrix operator^(float rhs);
    void operator^=(float rhs);

    // Elementwise operations on matrices
    Matrix operator*(Matrix& M);
    Matrix operator/(Matrix& M);
    Matrix operator+(Matrix& M);
    Matrix operator-(Matrix& M);

    void operator*=(Matrix& M);
    void operator/=(Matrix& M);
    void operator+=(Matrix& M);
    void operator-=(Matrix& M);

    // Matrix multiplication
    // If using this make sure to assign it to a new matrix as overwriting the current matrix
    // will cause a dangling pointer
    Matrix dot(Matrix& M, bool transpose1 = false, bool transpose2 = false, float scale = 1.0);

    float sum();
    std::string shape();
    std::string str();
    std::string signature();

    void toFile(std::string filename);
    void fromFile(std::string filename);

private:
    std::string tensorName;
    unsigned rows;
    unsigned cols;
    FeatType *data;
};

typedef std::map<std::string, Matrix> TensorMap;
typedef std::map<std::string, FeatType**> ETensorMap;


#endif
