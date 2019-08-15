#ifndef __GLOBAL_UTILS_HPP__
#define __GLOBAL_UTILS_HPP__


/** Feature type is float, so be consistent. */
typedef float FeatType;


static const size_t HEADER_SIZE = sizeof(unsigned) * 5;
enum OP { PUSH, PULL, REQ, RESP, TERM };


/**
 *
 * Serialization utilities.
 * 
 */
template<class T>
static inline void
serialize(char *buf, unsigned offset, T val) {
	std::memcpy(buf + (offset * sizeof(T)), &val, sizeof(T));
}

template<class T>
static inline T
parse(const char *buf, unsigned offset) {
	T val;
	std::memcpy(&val, buf + (offset * sizeof(T)), sizeof(T));
	return val;
}

// ID represents either layer or data partition, depending on server responding.
static inline void
populateHeader(char* header, unsigned op, unsigned id = 0, unsigned rows = 0, unsigned cols = 0) {
	serialize<unsigned>(header, 0, op);
	serialize<unsigned>(header, 1, id);
	serialize<unsigned>(header, 2, rows);
	serialize<unsigned>(header, 3, cols);
}


/**
 *
 * Struct for a matrix.
 * 
 */
class Matrix {

public:

    Matrix() { rows = 0; cols = 0; }
    Matrix(unsigned _rows, unsigned _cols) { rows = _rows; cols = _cols; }
    Matrix(unsigned _rows, unsigned _cols, FeatType *_data) { rows = _rows; cols = _cols; data = _data; }
    Matrix(unsigned _rows, unsigned _cols, char *_data) { rows = _rows; cols = _cols; data = (FeatType *) _data; }

    unsigned getRows() { return rows; }
    unsigned getCols() { return cols; }
    FeatType *getData() const { return data; }
    size_t getDataSize() const { return rows * cols * sizeof(FeatType); }

    void setRows(unsigned _rows) { rows = _rows; }
    void setCols(unsigned _cols) { cols = _cols; }
    void setDims(unsigned _rows, unsigned _cols) { rows = _rows; cols = _cols; }
    void setData(FeatType *_data) { data = _data; }

    bool empty() { return rows == 0 || cols == 0; }

    std::string shape() { return "(" + std::to_string(rows) + "," + std::to_string(cols) + ")"; }

    std::string str() {
        std::stringstream output;
        output << "Matrix Dims: " << shape() << "\n";
        for (unsigned i = 0; i < rows; ++i) {
            for (unsigned j = 0; j < cols; ++j) {
                output << data[i * cols + j] << " ";
            }
            output << "\n";
        }
        return output.str();
    }

private:

    unsigned rows;
    unsigned cols;
    FeatType *data;
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


#endif // GLOBAL_UTILS_HPP
