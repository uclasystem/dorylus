//Example 2. Application Using C and CUBLAS: 0-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"


#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <cmath>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <zmq.hpp>
#include "../../src/utils/utils.hpp"
#include <GPU_server.hpp>


using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;
using namespace aws::lambda_runtime;
using namespace std::chrono;

#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))


/**
 *
 * Request the input matrix data from dataserver.
 * 
 */
Matrix
requestMatrix(zmq::socket_t& socket, int32_t id) {
    
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PULL, id);
    socket.send(header);

    // Listen on respond.
    zmq::message_t respHeader;
    socket.recv(&respHeader);

    // Parse the respond.
    int32_t layerResp = parse<int32_t>((char *) respHeader.data(), 1);
    if (layerResp == -1) {      // Failed.
        std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
        return Matrix();
    } else {                    // Get matrix data.
        int32_t rows = parse<int32_t>((char *) respHeader.data(), 2);
        int32_t cols = parse<int32_t>((char *) respHeader.data(), 3);
        zmq::message_t matxData(rows * cols * sizeof(FeatType));
        socket.recv(&matxData);

        char *matxBuffer = new char[matxData.size()];
        std::memcpy(matxBuffer, matxData.data(), matxData.size());

        Matrix m(rows, cols, matxBuffer);
        return m;
    }
}


/**
 *
 * Send multiplied matrix result back to dataserver.
 * 
 */
void
sendMatrices(Matrix& zResult, Matrix& actResult, zmq::socket_t& socket, int32_t id) {
    
    // Send push header.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PUSH, id, zResult.rows, zResult.cols);
    socket.send(header, ZMQ_SNDMORE);

    // Send zData and actData.
    zmq::message_t zData(zResult.getDataSize());
    std::memcpy(zData.data(), zResult.getData(), zResult.getDataSize());
    zmq::message_t actData(actResult.getDataSize());
    std::memcpy(actData.data(), actResult.getData(), actResult.getDataSize());
    socket.send(zData, ZMQ_SNDMORE);
    socket.send(actData);

    // Wait for data settled reply.
    zmq::message_t confirm;
    socket.recv(&confirm);
}


/**
 *
 * Matrix multiplication function.
 * 
 */
Matrix
dot(Matrix& features, Matrix& weights) {
    int m = features.rows, k = features.cols, n = weights.cols;
    Matrix result(m, n);

    auto resultData = new FeatType[m * n];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                features.getData(), k, weights.getData(), n, 0.0, resultData, n);

    result.setData(resultData);

    return result;
}


/**
 *
 * Apply activation function on a matrix.
 * 
 */
Matrix
activate(Matrix& mat) {
    FeatType *activationData = new FeatType[mat.rows * mat.cols];
    FeatType *zData = mat.getData();
    
    for (int i = 0; i < mat.rows * mat.cols; ++i)
        activationData[i] = std::tanh(zData[i]);

    return Matrix(mat.rows, mat.cols, activationData);
}


/** Handler that hooks with lambda API. */
static invocation_response
my_handler(invocation_request const& request) {
    ptree pt;
    std::istringstream is(request.payload);
    read_json(is, pt);

    std::string dataserver = pt.get<std::string>("dataserver");
    std::string weightserver = pt.get<std::string>("weightserver");
    std::string dport = pt.get<std::string>("dport");
    std::string wport = pt.get<std::string>("wport");
    int32_t layer = pt.get<int32_t>("layer");
    int32_t chunkId = pt.get<int32_t>("id");

    std::cout << "Thread " << chunkId << " is requested from " << dataserver << ":" << dport
              << ", layer " << layer << "." << std::endl;

    return matmul(dataserver, weightserver, dport, wport, chunkId, layer);
}


int
main(int argc, char *argv[]) {
    run_handler(my_handler);
    
    return 0;
}





static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int
n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-q, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}
int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    float* devPtrA;
    float* a = 0;
    a = (float *)malloc (M * N * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            a[IDX2C(i,j,M)] = (float)(i * M + j + 1);
        }
    }
    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    modify (handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);
    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaFree (devPtrA);
    cublasDestroy(handle);
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", a[IDX2C(i,j,M)]);
        }   
        printf ("\n");
    }
    free(a);
    return EXIT_SUCCESS;
}