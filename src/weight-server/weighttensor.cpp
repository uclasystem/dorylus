#include "weighttensor.hpp"
#include <cstdlib>
#include <iostream>

RefMat::RefMat(unsigned _refCnt, Matrix _mat) :
    refCnt(_refCnt), mat(_mat) {}

WeightTensor::WeightTensor(Matrix &mat, std::mutex *_wmtx, std::mutex *_umtx, bool _sync) :
    sync(_sync), currVer(0), stop(false),
    wmtx(_wmtx), umtx(_umtx),
    localUpdTot(-1u), ghostUpdTot(-1u),
    localUpdCnt(0), ghostUpdCnt(0),
    localUpdMat(mat.getRows(), mat.getCols()),
    ghostUpdMat(mat.getRows(), mat.getCols()) {

    // init local weight store
    ver2Mat[0] = RefMat(0, mat);

    // init update store
    const unsigned numElemts = mat.getNumElemts();
    FeatType *lPtr = new FeatType[numElemts];
    for (unsigned i = 0; i < numElemts; ++i) {
        lPtr[i] = 0.0;
    }
    localUpdMat.setData(lPtr);
    if (sync) { // ghost update matrix is only needed in sync update
        FeatType *gPtr = new FeatType[numElemts];
        for (unsigned i = 0; i < numElemts; ++i) {
            gPtr[i] = 0.0;
        }
        ghostUpdMat.setData(gPtr);
    }
}

void WeightTensor::free() {
    for (auto &kv : ver2Mat) {
        // std::cout << "delete Weight Mat " << kv.second.mat.name() << " " << kv.second.mat.getData() << std::endl;
        kv.second.mat.free();
    }
    localUpdMat.free();
    if (sync) {
        ghostUpdMat.free();
    }
}

Matrix& WeightTensor::updateVersion(bool withLock) {
    if (withLock) {
        wmtx->lock();
    }

    if (stop) {
        return currMat();
    }

    RefMat &rmat = ver2Mat[currVer];
    // Current weight tensor is already done. Directly move it.
    if (rmat.refCnt == 0) {
        currVer++;
        ver2Mat[currVer] = rmat;
        ver2Mat.erase(currVer - 1);
        // std::cerr << "directly remove version " << currVer - 1 << std::endl;
    } else {
    // copy current weight matrix for new version, then apply update to the new one.
        const unsigned numElemts = rmat.mat.getNumElemts();
        FeatType *data = new FeatType[numElemts];
        memcpy(data, rmat.mat.getData(), sizeof(FeatType) * numElemts);

        currVer++;
        ver2Mat[currVer] = RefMat(0, Matrix(rmat.mat.name().c_str(), rmat.mat.getRows(), rmat.mat.getCols(), data));
        // std::cerr << "stash version " << currVer - 1 << std::endl;
    }
    // std::cerr << "Update version to " << currVer << std::endl;
    Matrix &updatedMat = currMat();
    if (withLock) {
        wmtx->unlock();
    }
    return updatedMat;
}

Matrix& WeightTensor::getMat(Chunk &chunk) {
    std::lock_guard<std::mutex> lg(*wmtx);

    // PROP_TYPE dir = chunk.dir;
    chunk.dir = PROP_TYPE::FORWARD; // force dir to forward to eliminate differences between forward and backward
    auto found = chunk2Ver.find(chunk);
    if (found != chunk2Ver.end()) {
        // std::cerr << "GET chunk " << chunk.layer << ":" << chunk.globalId << "-" << chunk.lowBound <<
        //             " dir: " << chunk.dir << " -> ver: " << found->second << "/" << currVer << std::endl;
        return ver2Mat[found->second].mat;
    } else { // Not found. This is a new chunk
        RefMat &rmat = ver2Mat[currVer];
        rmat.refCnt++;
        chunk2Ver[chunk] = currVer;
        // std::cerr << "GET chunk " << chunk.layer << ":" << chunk.globalId << "-" << chunk.lowBound <<
        //             " dir: " << chunk.dir << " -> ver: " << currVer << "/" << currVer << std::endl;
        return rmat.mat;
    }
}

void WeightTensor::decRef(Chunk &chunk) {
    std::lock_guard<std::mutex> lg(*wmtx);

    // PROP_TYPE dir = chunk.dir;
    chunk.dir = PROP_TYPE::FORWARD; // force dir to forward to eliminate differences between forward and backward
    unsigned ver;
    auto found = chunk2Ver.find(chunk);
    if (found != chunk2Ver.end()) {
        ver = chunk2Ver[chunk];
        RefMat &rmat = ver2Mat[ver];
        rmat.refCnt--;
        if (ver < currVer && rmat.refCnt == 0) { // an old stashed weights is done. We can safely delete it.
            rmat.mat.free();
            ver2Mat.erase(ver);
            // std::cerr << "weights " << ver << " done. deleted..." << std::endl;
        }
        chunk2Ver.erase(chunk);
        // std::cerr << "PUT chunk " << chunk.layer << ":" << chunk.globalId << "-" << chunk.lowBound <<
        //             " dir: " << chunk.dir << " -> ver: " << ver << std::endl;
    } else {
        std::cerr << "wrong chunk dec ref! " << chunk.str() << std::endl;
        return;
    }
}

void WeightTensor::stopUpdate() {
    std::lock_guard<std::mutex> lgw(*wmtx);
    std::lock_guard<std::mutex> lgu(*umtx);
    stop = true;
}

unsigned WeightTensor::localUpdate(FeatType *updTensor) {
    std::lock_guard<std::mutex> lg(*umtx);

    if (stop) {
        return localUpdCnt;
    }

    FeatType *lPtr = localUpdMat.getData();
    const unsigned numElemts = localUpdMat.getNumElemts();
    for (unsigned i = 0; i < numElemts; ++i) {
        lPtr[i] += updTensor[i];
    }

    localUpdCnt++;
    return localUpdCnt;
}

unsigned WeightTensor::ghostUpdate(FeatType *updTensor) {
    // Since only receiver thread call this function, it is not thread safe
    if (stop) {
        return ghostUpdCnt;
    }

    FeatType *gPtr = ghostUpdMat.getData();
    const unsigned numElemts = ghostUpdMat.getNumElemts();
    for (unsigned i = 0; i < numElemts; ++i) {
        gPtr[i] += updTensor[i];
    }

    ghostUpdCnt++;
    return ghostUpdCnt;
}

// SGD update with learning_rate
std::string WeightTensor::tryApplyUpdate(float lr, FeatType *updTensor) {
    std::lock_guard<std::mutex> lgu(*umtx);
    std::lock_guard<std::mutex> lgw(*wmtx);
    if (stop || // stop updating
        (sync && (localUpdCnt < localUpdTot || ghostUpdCnt < ghostUpdTot)) || // not ready for sync update
        (!sync && updTensor == NULL && localUpdCnt < localUpdTot)) { // not ready for async local update
        return "";
    }

    // For correct check
    std::string checkInfo;
    FeatType checkSum = 0;
    FeatType maxEle, minEle;

    Matrix &wmat = updateVersion(false);
    FeatType *wPtr = wmat.getData();

    const unsigned numElemts = localUpdMat.getNumElemts();
    if (sync) {
        FeatType *lPtr = localUpdMat.getData();
        FeatType *gPtr = ghostUpdMat.getData();
        for (unsigned u = 0; u < numElemts; ++u) {
            lPtr[u] += gPtr[u];
            if (CORRECT_CHECK) {
                checkSum += std::fabs(lPtr[u]);
            }
            wPtr[u] -= lr * lPtr[u];
        }

        if (CORRECT_CHECK) {
            maxEle = *(std::max_element(lPtr, lPtr + numElemts));
            minEle = *(std::min_element(lPtr, lPtr + numElemts));
        }

        for (unsigned u = 0; u < numElemts; ++u) {
            lPtr[u] = 0.0;
            gPtr[u] = 0.0;
        }
        localUpdCnt = 0;
        ghostUpdCnt = 0;
    } else if (updTensor) { // async && applyGhostUpdate
        for (unsigned u = 0; u < numElemts; ++u) {
            wPtr[u] -= lr * updTensor[u];
            if (CORRECT_CHECK) {
                checkSum += std::fabs(updTensor[u]);
            }
        }

        if (CORRECT_CHECK) {
            maxEle = *(std::max_element(updTensor, updTensor + numElemts));
            minEle = *(std::min_element(updTensor, updTensor + numElemts));
        }
    } else { // async && applyLocalUpdate
        FeatType *lPtr = localUpdMat.getData();
        if (CORRECT_CHECK) {
            for (unsigned u = 0; u < numElemts; ++u) {
                checkSum += std::fabs(lPtr[u]);
            }

            maxEle = *(std::max_element(lPtr, lPtr + numElemts));
            minEle = *(std::min_element(lPtr, lPtr + numElemts));
        }
        for (unsigned u = 0; u < numElemts; ++u) {
            wPtr[u] -= lr * lPtr[u];
            lPtr[u] = 0.0;
        }
        localUpdCnt = 0;
    }

    if (CORRECT_CHECK) {
        char buf[512];
        sprintf(buf, " Weight Grad Agg: %.5e Max element: %.2f Min element: %.2f",
                    checkSum, maxEle, minEle);
        checkInfo = std::string(buf);
    } else {
        checkInfo = "Current version: " + std::to_string(currVer) + ", active weight version cnt: " + std::to_string(ver2Mat.size());
    }
    return checkInfo;
}

// Update with Adam optimizer
std::string WeightTensor::tryApplyUpdate(AdamOptimizer *adamOpt, unsigned layer, FeatType *updTensor) {
    std::lock_guard<std::mutex> lgu(*umtx);
    std::lock_guard<std::mutex> lgw(*wmtx);
    if (stop || // stop updating
        (sync && (localUpdCnt < localUpdTot || ghostUpdCnt < ghostUpdTot)) || // not ready for sync update
        (!sync && updTensor == NULL && localUpdCnt < localUpdTot)) { // not ready for async local update
        return "";
    }

    std::string checkInfo;
    FeatType checkSum = 0;
    FeatType maxEle, minEle;

    Matrix &wmat = updateVersion(false);
    FeatType *wPtr = wmat.getData();

    const unsigned numElemts = localUpdMat.getNumElemts();
    if (sync) {
        FeatType *lPtr = localUpdMat.getData();
        FeatType *gPtr = ghostUpdMat.getData();
        for (unsigned u = 0; u < numElemts; ++u) {
            lPtr[u] += gPtr[u];
            if (CORRECT_CHECK) {
                checkSum += std::fabs(lPtr[u]);
            }
        }
        adamOpt->update(layer, wPtr, lPtr);

        if (CORRECT_CHECK) {
            maxEle = *(std::max_element(lPtr, lPtr + numElemts));
            minEle = *(std::min_element(lPtr, lPtr + numElemts));
        }

        for (unsigned u = 0; u < numElemts; ++u) {
            lPtr[u] = 0.0;
            gPtr[u] = 0.0;
        }
        localUpdCnt = 0;
        ghostUpdCnt = 0;
    } else if (updTensor) { // async && applyGhostUpdate
        if (CORRECT_CHECK) {
            for (unsigned u = 0; u < numElemts; ++u) {
                checkSum += std::fabs(updTensor[u]);
            }

            maxEle = *(std::max_element(updTensor, updTensor + numElemts));
            minEle = *(std::min_element(updTensor, updTensor + numElemts));
        }
        adamOpt->update(layer, wPtr, updTensor);
    } else { // async && applyLocalUpdate
        FeatType *lPtr = localUpdMat.getData();
        if (CORRECT_CHECK) {
            for (unsigned u = 0; u < numElemts; ++u) {
                checkSum += std::fabs(lPtr[u]);
            }

            maxEle = *(std::max_element(lPtr, lPtr + numElemts));
            minEle = *(std::min_element(lPtr, lPtr + numElemts));
        }
        adamOpt->update(layer, wPtr, lPtr);
        for (unsigned u = 0; u < numElemts; ++u) {
            lPtr[u] = 0.0;
        }
        localUpdCnt = 0;
    }

    if (CORRECT_CHECK) {
        char buf[512];
        sprintf(buf, " Weight Grad Agg: %.5f Max element: %.5f Min element: %.5f",
                checkSum, maxEle, minEle);
        checkInfo = std::string(buf);
    } else {
        checkInfo = "Current version: " + std::to_string(currVer) + ", active weight version cnt: " + std::to_string(ver2Mat.size())
                    + " " + std::to_string(localUpdCnt) + ":" + std::to_string(localUpdTot)
                    + " " + std::to_string(ghostUpdCnt) + ":" + std::to_string(ghostUpdTot)
                    + " -> (";
        for (auto &kv : ver2Mat) {
            checkInfo += std::to_string(kv.first) + ":" + std::to_string(kv.second.refCnt) + ", ";
        }
        checkInfo += ")";
    }
    return checkInfo;
}

std::string WeightTensor::tryApplyUpdateFake(float lr, FeatType *updTensor) {
    std::lock_guard<std::mutex> lgu(*umtx);
    std::lock_guard<std::mutex> lgw(*wmtx);
    localUpdCnt = 0;
    ghostUpdCnt = 0;

    return "Fake Apply Update";
}

std::string WeightTensor::tryApplyUpdateFake(AdamOptimizer *adamOpt, unsigned layer, FeatType *updTensor) {
    std::lock_guard<std::mutex> lgu(*umtx);
    std::lock_guard<std::mutex> lgw(*wmtx);
    localUpdCnt = 0;
    ghostUpdCnt = 0;

    return "Fake Apply Update";
}