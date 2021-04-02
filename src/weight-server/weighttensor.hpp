#ifndef __WEIGHT_TENSOR_HPP__
#define __WEIGHT_TENSOR_HPP__

#include <vector>
#include <map>
#include <mutex>
#include "../common/matrix.hpp"
#include "../common/utils.hpp"
#include "AdamOptimizer.hpp"

#define CORRECT_CHECK false

// Matrix with reference counting
struct RefMat {
    unsigned refCnt = 0;
    Matrix mat;

    RefMat() {};
    RefMat(unsigned _refCnt, Matrix _mat);
};

struct WeightTensor {
    unsigned currVer = 0;
    // These 2 maps should map to the same RefMat set
    std::map<unsigned, RefMat> ver2Mat;
    std::map<Chunk, unsigned> chunk2Ver;

    std::mutex *wmtx;

    WeightTensor() {}; // shouldn't be called
    WeightTensor(Matrix &mat, std::mutex *_wmtx, std::mutex *_umtx, bool _sync = false);
    void free();

    Matrix& currMat() { return ver2Mat[currVer].mat; };
    Matrix& updateVersion(bool withLock = true);
    Matrix& getMat(Chunk &chunk);
    void decRef(Chunk &chunk);

    void stopUpdate();

    bool sync;
    bool stop;

    unsigned localUpdTot;
    unsigned ghostUpdTot;

    unsigned localUpdCnt;
    unsigned ghostUpdCnt;

    Matrix localUpdMat;
    Matrix ghostUpdMat;

    std::mutex *umtx;

    unsigned localUpdate(FeatType *updTensor);
    unsigned ghostUpdate(FeatType *updTensor);

    // Sync update needs both localUpdCnt and ghostUpdCnt equal to [local|ghost]UpdTot.
    // tryApply will do nothing if either condition is not satisfied.
    // Async update will directly apply.
    std::string tryApplyUpdate(float lr, FeatType *updTensor = NULL);
    std::string tryApplyUpdate(AdamOptimizer *adamOpt, unsigned layer, FeatType *updTensor = NULL);
    // Simulate weight update
    std::string tryApplyUpdateFake(float lr, FeatType *updTensor = NULL);
    std::string tryApplyUpdateFake(AdamOptimizer *adamOpt, unsigned layer, FeatType *updTensor = NULL);

    void setLocalUpdTot(unsigned lut) {
        std::lock_guard<std::mutex> lg(*umtx);
        localUpdTot = lut;
    }
    void setGhostUpdTot(unsigned gut) {
        std::lock_guard<std::mutex> lg(*umtx);
        ghostUpdTot = gut;
    }
};

typedef std::map<std::string, WeightTensor> WeightTensorMap;
typedef std::map<std::string, std::mutex> MutexMap;

#endif // __WEIGHT_TENSOR_HPP__