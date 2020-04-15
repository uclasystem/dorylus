#include "versionmatrix.hpp"
#include <cstdlib>
#include <iostream>

RefMat::RefMat(unsigned _refCnt, Matrix _mat) :
    refCnt(_refCnt), mat(_mat) {}

VersionMatrix::VersionMatrix(Matrix &mat) : currVer(0) {
    ver2Mat[0] = RefMat(0, mat);
}

Matrix& VersionMatrix::updateVersion() {
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
    return currMat();
}

Matrix& VersionMatrix::getMat(Chunk &chunk) {
    // PROP_TYPE dir = chunk.dir;
    chunk.dir = PROP_TYPE::FORWARD; // force dir to forward to eliminate differences between forward and backward
    auto found = chunk2Ver.find(chunk);
    if (found != chunk2Ver.end()) {
        // std::cerr << "GET chunk " << chunk.layer << ":" << chunk.chunkId << "-" << chunk.lowBound <<
        //             " dir: " << dir << " -> ver: " << found->second << "/" << currVer << std::endl;
        return ver2Mat[found->second].mat;
    } else { // Not found. This is a new chunk
        RefMat &rmat = ver2Mat[currVer];
        rmat.refCnt++;
        chunk2Ver[chunk] = currVer;
        // std::cerr << "GET chunk " << chunk.layer << ":" << chunk.chunkId << "-" << chunk.lowBound <<
        //             " dir: " << dir << " -> ver: " << currVer << "/" << currVer << std::endl;
        return rmat.mat;
    }
}

void VersionMatrix::decRef(Chunk &chunk) {
    // PROP_TYPE dir = chunk.dir;
    chunk.dir = PROP_TYPE::FORWARD; // force dir to forward to eliminate differences between forward and backward
    unsigned ver = chunk2Ver[chunk];
    RefMat &rmat = ver2Mat[ver];
    rmat.refCnt--;
    if (ver < currVer && rmat.refCnt == 0) { // an old stashed weights is done. We can safely delete it.
        rmat.mat.free();
        ver2Mat.erase(ver);
        // std::cerr << "weights " << ver << " done. deleted..." << std::endl;
    }
    chunk2Ver.erase(chunk);
    // std::cerr << "PUT chunk " << chunk.layer << ":" << chunk.chunkId << "-" << chunk.lowBound <<
    //             " dir: " << dir << " -> ver: " << ver << std::endl;
}