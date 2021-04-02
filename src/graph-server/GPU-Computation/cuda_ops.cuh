struct activateDerivative_functor {
    activateDerivative_functor() {}
    __host__ __device__ float operator()(const float &x) const {
        return 1 - pow(tanh(x), 2);
    }
};

struct leakyRelu_functor {
    leakyRelu_functor(float coef_) : coef(coef_) {}
    __host__ __device__ float operator()(const float &x) const {
        return x > 0 ? x : coef * x;
    }
    float coef;
};

struct leakyReluPrime_functor {
    leakyReluPrime_functor(float coef_) : coef(coef_) {}
    __host__ __device__ float operator()(const float &x) const {
        return x > 0 ? 1 : coef;
    }
    float coef;
};

struct exp_functor {
    exp_functor() {}
    __host__ __device__ float operator()(const float &x) const {
        return expf(x);
    }
};

struct divide_functor {
    divide_functor(float *denoms_) : denoms(denoms_) {}
    __host__ __device__ float operator()(const int &i, const float &x) const {
        return x / denoms[i];
    }
    float *denoms;
};

struct setRow_functor {
    setRow_functor(unsigned col_) : col(col_) {}
    unsigned col;
    __host__ __device__ int operator()(const int &x) const { return x / col; }
};

struct setRowStarts {
    setRowStarts(FeatType *data_ptr_, unsigned col_)
        : col(col_), data_ptr(data_ptr_) {}
    __host__ __device__ FeatType *operator()(const unsigned &i) const {
        return data_ptr + i * col;
    }
    FeatType *data_ptr;
    unsigned col;
};

struct findRowMaximum {
    findRowMaximum(unsigned col_) : col(col_) {}
    __host__ __device__ unsigned operator()(const FeatType *ptr) const {
        unsigned index = 0;
        float max = ptr[0];
        for (unsigned i = 1; i < col; ++i) {
            if (ptr[i] > max) {
                index = i;
                max = ptr[i];
            }
        }
        return index;
    }
    unsigned col;
};

struct isPredictCorrect {
    isPredictCorrect(unsigned col_) : col(col_) {}
    __host__ __device__ unsigned operator()(const unsigned pred,
                                            const FeatType *label) const {
        if (label[pred] == 1) return 1;
        return 0;
    }
    unsigned col;
};

struct findTrueLabel {
    findTrueLabel(unsigned col_) : col(col_) {}
    __host__ __device__ unsigned operator()(const FeatType *ptr) const {
        for (unsigned i = 0; i < col; i++) {
            ;
            if (ptr[i] == 1.0) {
                return i;
            }
        }

        return (unsigned)-1;
    }
    unsigned col;
};

struct isEqual {
    isEqual() {}
    __host__ __device__ unsigned operator()(const unsigned &i,
                                            const unsigned &j) const {
        if (i == j)
            return 1;
        else
            return 0;
    }
};

struct getLoss {
    getLoss(unsigned col_) : col(col_) {}
    __host__ __device__ FeatType operator()(const unsigned &i,
                                            const FeatType *softmax_ptr) const {
        return -logf(softmax_ptr[i]);
    }
    unsigned col;
};