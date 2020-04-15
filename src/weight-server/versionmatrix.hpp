#ifndef __VERSION_MATRIX_HPP__
#define __VERSION_MATRIX_HPP__

#include <vector>
#include <map>
#include "../common/matrix.hpp"
#include "../common/utils.hpp"
#include "AdamOptimizer.hpp"

// Matrix with reference counting
struct RefMat {
    unsigned refCnt = 0;
    Matrix mat;

    RefMat() {};
    RefMat(unsigned _refCnt, Matrix _mat);
};

struct VersionMatrix {
    unsigned currVer = 0;
    // These 2 maps should map to the same RefMat set
    std::map<unsigned, RefMat> ver2Mat;
    std::map<Chunk, unsigned> chunk2Ver;

    VersionMatrix() {}; // shouldn't be called
    VersionMatrix(Matrix &mat);

    Matrix& currMat() { return ver2Mat[currVer].mat; };
    Matrix& updateVersion();
    Matrix& getMat(Chunk &chunk);
    void decRef(Chunk &chunk);
};

typedef std::map<std::string, VersionMatrix> VersionTensorMap;

#endif // __VERSION_MATRIX_HPP__