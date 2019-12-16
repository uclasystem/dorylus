#include "cu_matrix.cuh"
#include "comp_unit.cuh"
#include <iostream>
#include <cudnn.h>
#include <time.h>
#include <cusparse.h>
#include <set>
#include <tuple>
typedef std::tuple<int, int, EdgeType> triplet;
using namespace std;
cublasHandle_t handle;
cusparseHandle_t  spHandle;
ComputingUnit cu = ComputingUnit::getInstance();

const int ROW = 5000;
const int COL = 41;
float alpha = 1.0, beta = 0.0;

void test0() {
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Matrix matA(2, 3, a);
    Matrix matB(3, 4, b);
    CuMatrix cuA(matA, handle);
    CuMatrix cuB(matB, handle);
    CuMatrix cuD = cuA.dot(cuB, false, false);
    cuD.updateMatrixFromGPU();
    cout << cuD.str();
    cout << matA.dot(matB).str();
}
void test1() {
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Matrix matA(3, 2, a);
    Matrix matB(3, 4, b);
    CuMatrix cuA(matA, handle);
    CuMatrix cuB(matB, handle);
    CuMatrix cuD = cuA.dot(cuB, true, false);
    cuD.updateMatrixFromGPU();
    cout << cuD.str();
    cout << matA.dot(matB, true, false).str();
}
void test2() {
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Matrix matA(2, 3, a);
    Matrix matB(4, 3, b);
    CuMatrix cuA(matA, handle);
    CuMatrix cuB(matB, handle);
    CuMatrix cuD = cuA.dot(cuB, false, true);
    cuD.updateMatrixFromGPU();
    cout << cuD.str();
    cout << matA.dot(matB, false, true).str();
}
void test3() {
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Matrix matA(3, 2, a);
    Matrix matB(4, 3, b);
    CuMatrix cuA(matA, handle);
    CuMatrix cuB(matB, handle);
    CuMatrix cuD = cuA.dot(cuB, true, true);
    cuD.updateMatrixFromGPU();
    cout << cuD.str();
    cout << matA.dot(matB, true, true).str();

}
void test4() {
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Matrix matA(3, 2, a);
    Matrix matB(4, 3, b);
    CuMatrix cuA(matA, handle);
    CuMatrix cuB(matB, handle);
    CuMatrix cuD = cuA.dot(cuB, true, true);
    cuD.updateMatrixFromGPU();
    cout << cuD.str();
    cout << matA.dot(matB, true, true).str();

}

Matrix softmax(Matrix &mat) {
    FeatType *result = new FeatType[mat.getNumElemts()];

    for (unsigned r = 0; r < mat.getRows(); ++r) {
        unsigned length = mat.getCols();
        FeatType *vecSrc = mat.getData() + r * length;
        FeatType *vecDst = result + r * length;

        FeatType denom = 0.0;
        for (unsigned c = 0; c < length; ++c) {
            vecDst[c] = std::exp(vecSrc[c]);
            denom += vecDst[c];
        }

        for (unsigned c = 0; c < length; ++c) {
            vecDst[c] /= denom;
        }
    }

    return Matrix(mat.getRows(), mat.getCols(), result);
}
void test_softmax() {
    float alpha = 1.0, beta = 0.0;
    float b[ROW * COL];
    Matrix matB(ROW, COL, b);
    for (int i = 0; i < ROW * COL; ++i) {
        matB.getData()[i] = i / ROW * COL;
    }
    CuMatrix cuB(matB, handle);

    clock_t t = clock();
    CuMatrix cuC = cu.softmaxRows(cuB);
    t = clock() - t;
    printf ("GPU %f\n", ((float)t) / CLOCKS_PER_SEC);

    t = clock();
    Matrix C = softmax(matB);
    t = clock() - t;

    printf ("CPU %f\n", ((float)t) / CLOCKS_PER_SEC);
    // cout<<C.str();
    cuC.updateMatrixFromGPU();
    // cout<<cuC.str();

    printf("CHECKING\n");
    for (int i = 0; i < ROW * COL; ++i) {
        if(C.getData()[i] - cuC.getData()[i] > 0.01) {
            printf("ERROR\n");
            return;
        }
    }
}

void test_tanh() {
    float alpha = 1.0, beta = 0.0;
    float a[ROW * COL];
    float b[ROW * COL];

    Matrix matA(ROW, COL, a);
    Matrix matB(ROW, COL, b);
    for (int i = 0; i < ROW * COL; ++i) {
        matA.getData()[i] = i / ROW * COL;
        matB.getData()[i] = i / ROW * COL;
    }
    CuMatrix cuB(matB, handle);

    clock_t t = clock();
    for (unsigned i = 0; i < matA.getNumElemts(); ++i)
        matA.getData()[i] = std::tanh(matA.getData()[i]);
    t = clock() - t;
    printf ("CPU %f\n", ((float)t) / CLOCKS_PER_SEC);

    t = clock();
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               ROW, 1, 1, COL);

    cudnnActivationDescriptor_t actDesc;
    cudnnCreateActivationDescriptor(&actDesc);
    cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 1.0);
    cudnnActivationForward(cu.cudnnHandle, actDesc,
                           &alpha, srcTensorDesc, cuB.devPtr, &beta, srcTensorDesc, cuB.devPtr);
    t = clock() - t;

    printf ("CUDNN %f\n", ((float)t) / CLOCKS_PER_SEC);
    cuB.updateMatrixFromGPU();

    printf("CHECKING\n");
    for (int i = 0; i < ROW * COL; ++i) {
        if(matA.getData()[i] - cuB.getData()[i] > 0.01) {
            printf("ERROR\n");
            return;
        }
    }
}

static Matrix
activateDerivate(Matrix &mat) {
    FeatType *res = new FeatType[mat.getNumElemts()];
    FeatType *zData = mat.getData();

    for (unsigned i = 0; i < mat.getNumElemts(); ++i)
        res[i] = 1 - std::pow(std::tanh(zData[i]), 2);

    return Matrix(mat.getRows(), mat.getCols(), res);
}

void test_aggregation() {
    std::set<triplet> links;
    links.insert(triplet(0, 2, 1));
    links.insert(triplet(0, 0, .1));
    links.insert(triplet(1, 1, .01));
    links.insert(triplet(1, 2, 1));
    links.insert(triplet(2, 2, .001));
    unsigned count = links.size();
    EdgeType *norms  = new EdgeType[count];
    int *rowInd = new int[count];
    int *colInd = new int[count];
    unsigned i = 0;
    for(auto link : links) {
        rowInd[i] = std::get<0>(link);
        colInd[i] = std::get<1>(link);
        norms[i] = std::get<2>(link);
        i++;
    }
    for(int i = 0; i < count; ++i)
        cout << rowInd[i] << " ";
    cout << endl;
    for(int i = 0; i < count; ++i)
        cout << colInd[i] << " ";
    cout << endl;
    for(int i = 0; i < count; ++i)
        cout << norms[i] << " ";
    cout << endl;

    EdgeType *csrVal;
    int *csrColInd;
    int *cooRowInd;
    auto
    //copy val
    cudaStat = cudaMalloc ((void **)&csrVal, count * sizeof(EdgeType));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMemcpy(csrVal, norms, sizeof(EdgeType) * count, cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);

    //copy row indices
    cudaStat = cudaMalloc ((void **)&cooRowInd, count * sizeof(int));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMemcpy(cooRowInd, rowInd, sizeof(int) * count, cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);

    //copy col
    cudaStat = cudaMalloc ((void **)&csrColInd, count * sizeof(int));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMemcpy(csrColInd, colInd, sizeof(int) * count, cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);


    float feats[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    CuMatrix f(Matrix(3, 2, feats), handle);
    cout << f.str();
    CuMatrix A;
    A.nnz = count;
    A.cooRowInd = cooRowInd;
    A.csrVal = csrVal;
    A.csrColInd = csrColInd;
    A.setRows(3);
    A.setCols(3);
    A.handle = handle;

    CuMatrix out = cu.aggregate(A, f);
    cout << out.getMatrix().str();
    auto z = out.transpose();
    // float* z=new float[out.getNumElemts()];
    // cudaMemcpy(z, out.devPtr, sizeof(EdgeType) * out.getNumElemts(), cudaMemcpyDeviceToHost);
    // for(int j=0;j<out.getNumElemts();++j)
    //     cout<<z[j]<<" ";
    // cout<<"\n";
    cout << z.getMatrix().str();
}

void testElementSub() {
    float y[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    float x[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.7};
    CuMatrix cuX(Matrix(6, 1, x), handle);
    CuMatrix cuY(Matrix(6, 1, y), handle);
    auto z = cu.hadamardSub(cuX, cuY);
    cout << z.getMatrix().str();

}


int main() {
    handle = cu.handle;
    spHandle = cu.spHandle;


    return 0;
}
