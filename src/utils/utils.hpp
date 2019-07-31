#ifndef __UTILS_HPP__
#define __UTILS_HPP__


/** Feature type is float, so be consistent. */
typedef float DTYPE;


static const int32_t HEADER_SIZE = sizeof(int32_t) * 5;
enum OP { PUSH, PULL, REQ, RESP, TERM };


/**
 *
 * Serialization utilities.
 * 
 */
template<class T>
void
serialize(char *buf, int32_t offset, T val) {
	std::memcpy(buf + (offset * sizeof(T)), &val, sizeof(T));
}

template<class T>
T
parse(const char *buf, int32_t offset) {
	T val;
	std::memcpy(&val, buf + (offset * sizeof(T)), sizeof(T));
	return val;
}

// ID represents either layer or data partition, depending on server responding.
void populateHeader(char* header, int32_t op, int32_t id, int32_t rows = 0, int32_t cols = 0) {
	serialize<int32_t>(header, 0, op);
	serialize<int32_t>(header, 1, id);
	serialize<int32_t>(header, 2, rows);
	serialize<int32_t>(header, 3, cols);
}


/**
 *
 * Struct for a matrix.
 * 
 */
struct Matrix {
    int32_t rows;
    int32_t cols;
    FeatType *data;

    Matrix() { rows = 0; cols = 0; }
    Matrix(int _rows, int _cols) { rows = _rows; cols = _cols; }
    Matrix(int _rows, int _cols, FeatType *_data) { rows = _rows; cols = _cols; data = _data; }
    Matrix(int _rows, int _cols, char *_data) { rows = _rows; cols = _cols; data = (FeatType *) _data; }

    FeatType *getData() const { return data; }
    size_t getDataSize() const { return rows * cols * sizeof(FeatType); }

    void setRows(int32_t _rows) { rows = _rows; }
    void setCols(int32_t _cols) { cols = _cols; }
    void setDims(int32_t _rows, int32_t _cols) { rows = _rows; cols = _cols; }
    void setData(FeatType *_data) { data = _data; }

    bool empty() { return rows == 0 || cols == 0; }

    std::string shape() { return "(" + std::to_string(rows) + "," + std::to_string(cols) + ")"; }

    std::string str() {
        std::stringstream output;
        output << "Matrix Dims: " << shape() << "\n";
        for (int32_t i = 0; i < rows; ++i) {
            for (int32_t j = 0; j < cols; ++j) {
                output << data[i * cols + j] << " ";
            }
            output << "\n";
        }
        return output.str();
    }
};


/**
 *
 * Struct for a timer.
 * 
 */
struct Timer {
    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::high_resolution_clock::time_point end;

    void start() { begin = std::chrono::high_resolution_clock::now(); }
    void stop() { end = std::chrono::high_resolution_clock::now(); }

    double getTime() {
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);
        return time_span.count();
    }
};


#endif // UTILS_HPP
