#ifndef __GCN_UTILS_HPP__
#define __GCN_UTILS_HPP__

#include <aws/lambda-runtime/runtime.h>

#include "../../common/matrix.hpp"

invocation_response
constructResp(bool success, unsigned id, std::string msg) {
    JsonValue jsonResponse;
    jsonResponse.WithBool("success", success);
    jsonResponse.WithInteger("id", id);
    jsonResponse.WithString("message", msg);
    auto response = jsonResponse.View().WriteCompact();
    return invocation_response::success(response, "appliation/json");
}

void
deleteMatrix(Matrix &mat) {
    if (!mat.empty()) {
        delete[] mat.getData();
        mat = Matrix();
    }
}

#endif
