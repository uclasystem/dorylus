#include "lambdaworker.hpp"


extern std::mutex count_mutex, eval_mutex;
extern std::condition_variable cv_forward, cv_backward;


/**
 *
 * LambdaWorker constructor & destructor.
 * 
 */
LambdaWorker::LambdaWorker(unsigned nodeId_, zmq::context_t& ctx_,
                           unsigned numLambdasForward_, unsigned numLambdasBackward_,
                           unsigned& countForward_, unsigned& countBackward_,
                           unsigned& numCorrectPredictions_, float& totalLoss_,
                           unsigned& numValidationVertices_,
                           std::vector<bool>& trainPartitions_)
    : nodeId(nodeId_), ctx(ctx_), workersocket(ctx, ZMQ_DEALER),
      numLambdasForward(numLambdasForward_), numLambdasBackward(numLambdasBackward_),
      countForward(countForward_), countBackward(countBackward_),
      numCorrectPredictions(numCorrectPredictions_), totalLoss(totalLoss_),
      numValidationVertices(numValidationVertices_),
      trainPartitions(trainPartitions_), evalLambdas(0) {
    workersocket.setsockopt(ZMQ_LINGER, 0);
    workersocket.connect("inproc://backend");

    // Find the number of evaluation partitions
    // May be better to do in LambdaComm... Somewhat redundant here
    evalPartitions = 0;
    for (bool train : trainPartitions) {
        if (!train)
            ++evalPartitions;
    }
}

LambdaWorker::~LambdaWorker() {
    workersocket.close();
}


/**
 *
 * Lambdaworker is a wrapper over the sender & receiver thread.
 * 
 */
void
LambdaWorker::work() {
    try {
        while (true) {
            zmq::message_t identity;
            zmq::message_t header;

            workersocket.recv(&identity);
            workersocket.recv(&header);

            unsigned op = parse<unsigned>((char *) header.data(), 0);
            unsigned partId = parse<unsigned>((char *) header.data(), 1);

            switch (op) {
                case (OP::PULL_FORWARD):
                    sendAggregatedChunk(identity, partId);
                    break;
                case (OP::PUSH_FORWARD):
                    recvLambdaResults(identity, partId);
                    break;
                case (OP::PULL_BACKWARD):
                    sendBackpropChunks(identity, partId);
                    break;
                case (OP::PULL_EVAL):
                    sendTargetMatrix(identity, partId);
                    break;
                case (OP::PUSH_EVAL):
                    recvValidationResults(identity, header);
                    break;
                case (OP::PUSH_BACKWARD):
                    recvBackpropFinishMsg(identity);
                    break;
                default:
                    break;  /** Not an op that I care about. */
            }
        }
    } catch (std::exception& ex) { /** Context Termintated. */ }
}


/**
 *
 * Reset the member values for the next round of communication.
 * 
 */
void
LambdaWorker::refreshState(Matrix actMatrix_, FeatType *zData_, FeatType *actData_, unsigned numFeatsNext_, bool eval) {           // For forward-prop.
    actMatrix = actMatrix_;
    zData = zData_;
    actData = actData_;
    numFeatsNext = numFeatsNext_;
    evaluate = eval;

    if (evaluate) {
        evalLambdas = evalPartitions;
    }
}

void
LambdaWorker::refreshState(std::vector<Matrix> zMatrices_, std::vector<Matrix> actMatrices_, Matrix targetMatrix_) {    // For backward-prop.
    zMatrices = zMatrices_;
    actMatrices = actMatrices_;
    targetMatrix = targetMatrix_;
}


/**
 *
 * Sending & receiving messages to / from lambda threads.
 * 
 */
void
LambdaWorker::sendAggregatedChunk(zmq::message_t& client_id, unsigned partId) {

    // Reject a send request if the partition id is invalid.
    if (partId >= numLambdasForward) {
        workersocket.send(client_id, ZMQ_SNDMORE);
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
        workersocket.send(header);

        printLog(nodeId, "[ERROR] Got a request for partition %u, but number of lambdas is %u", partId, numLambdasForward);

    // Partition id is valid, so send the matrix segment.
    } else {

        workersocket.send(client_id, ZMQ_SNDMORE);

        // Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
        // If they do, set partition end to the end of the array.
        unsigned partRows = std::ceil((float) actMatrix.getRows() / (float) numLambdasForward);
        unsigned thisPartRows = partRows;
        if ((partId * partRows + partRows) > actMatrix.getRows())
            thisPartRows = partRows - (partId * partRows + partRows) + actMatrix.getRows();
        unsigned bufSize = thisPartRows * actMatrix.getCols() * sizeof(FeatType);
        FeatType *partitionStart = actMatrix.getData() + (partId * partRows * actMatrix.getCols());

        // Tell the lambda whether or not it should run evaluation
        // If the engine has set evaluate to true and this is not a training partition
        unsigned lambdaEval = evaluate && !trainPartitions[partId];
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, actMatrix.getCols(), lambdaEval);
        workersocket.send(header, ZMQ_SNDMORE);

        zmq::message_t partitionData(bufSize);
        std::memcpy(partitionData.data(), partitionStart, bufSize);
        workersocket.send(partitionData);
    }
}

void
LambdaWorker::recvLambdaResults(zmq::message_t& client_id, unsigned partId) {
    unsigned partRows = std::ceil((float) actMatrix.getRows() / (float) numLambdasForward);
    FeatType *partitionZStart = zData + partId * partRows * numFeatsNext;
    FeatType *partitionActStart = actData + partId * partRows * numFeatsNext;

    // Receive the pushed-back results.
    zmq::message_t data;
    workersocket.recv(&data);
    std::memcpy(partitionZStart, data.data(), data.size());
    workersocket.recv(&data);
    std::memcpy(partitionActStart, data.data(), data.size());

    // Send confirm ACK message.
    zmq::message_t confirm;
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(confirm);

    // Check for total number of partitions received. If all partitions received, wake up lambdaComm.
    std::lock_guard<std::mutex> lk(count_mutex);
    ++countForward;
    if (countForward == numLambdasForward)
        cv_forward.notify_one();
}

void
LambdaWorker::sendBackpropChunks(zmq::message_t& client_id, unsigned partId) {

    // Reject a send request if the partition id is invalid.
    if (partId >= numLambdasBackward) {
        workersocket.send(client_id, ZMQ_SNDMORE);
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
        workersocket.send(header);

    // Partition id is valid, so send the matrix segments.
    } else {

        workersocket.send(client_id, ZMQ_SNDMORE);

        // Check to make sure that the bounds of this partition do not exceed the bounds of the data array.
        // If they do, set partition end to the end of the array.
        unsigned partRows = std::ceil((float) targetMatrix.getRows() / (float) numLambdasBackward);
        unsigned thisPartRows = partRows;
        if ((partId * partRows + partRows) > targetMatrix.getRows())
            thisPartRows = partRows - (partId * partRows + partRows) + targetMatrix.getRows();

        // Send z matrices, from layer 1-> last.
        for (Matrix& matrix : zMatrices) {
            unsigned bufSize = thisPartRows * matrix.getCols() * sizeof(FeatType);
            FeatType *partitionStart = matrix.getData() + (partId * partRows * matrix.getCols());

            zmq::message_t header(HEADER_SIZE);
            populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, matrix.getCols());
            workersocket.send(header, ZMQ_SNDMORE);

            zmq::message_t partitionData(bufSize);
            std::memcpy(partitionData.data(), partitionStart, bufSize);
            workersocket.send(partitionData, ZMQ_SNDMORE);
        }

        // Send activation matrices, from layer 0 -> last.
        for (Matrix& matrix : actMatrices) {
            unsigned bufSize = thisPartRows * matrix.getCols() * sizeof(FeatType);
            FeatType *partitionStart = matrix.getData() + (partId * partRows * matrix.getCols());

            zmq::message_t header(HEADER_SIZE);
            populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, matrix.getCols());
            workersocket.send(header, ZMQ_SNDMORE);

            zmq::message_t partitionData(bufSize);
            std::memcpy(partitionData.data(), partitionStart, bufSize);
            workersocket.send(partitionData, ZMQ_SNDMORE);
        }

        // Send target label matrix.
        unsigned bufSize = thisPartRows * targetMatrix.getCols() * sizeof(FeatType);
        FeatType *partitionStart = targetMatrix.getData() + (partId * partRows * targetMatrix.getCols());

        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::RESP, 0, thisPartRows, targetMatrix.getCols());
        workersocket.send(header, ZMQ_SNDMORE);

        zmq::message_t partitionData(bufSize);
        std::memcpy(partitionData.data(), partitionStart, bufSize);
        workersocket.send(partitionData);
    }
}

void
LambdaWorker::recvBackpropFinishMsg(zmq::message_t& client_id) {

    // Send confirm ACK message.
    zmq::message_t confirm;
    workersocket.send(client_id, ZMQ_SNDMORE);
    workersocket.send(confirm);

    // Check for total number of partitions received. If all partitions received, wake up lambdaComm.
    std::lock_guard<std::mutex> lk(count_mutex);
    ++countBackward;
    if (countBackward == numLambdasBackward)
        cv_backward.notify_one();
}

void
LambdaWorker::sendTargetMatrix(zmq::message_t& client_id, unsigned partId) {
    unsigned partRows = std::ceil((float) targetMatrix.getRows() / (float) numLambdasForward);
    unsigned thisPartRows = partRows;
    if ((partId * partRows + partRows) > targetMatrix.getRows())
        thisPartRows = partRows - (partId * partRows + partRows) + targetMatrix.getRows();

    unsigned bufSize = thisPartRows * targetMatrix.getCols() * sizeof(FeatType);
    FeatType* partitionStart = targetMatrix.getData() + (partId * partRows * targetMatrix.getCols());

    zmq::message_t header(HEADER_SIZE);
    populateHeader((char*) header.data(), OP::RESP, 0, thisPartRows, targetMatrix.getCols());
    workersocket.send(header, ZMQ_SNDMORE);

    zmq::message_t partitionData(bufSize);
    std::memcpy(partitionData.data(), partitionStart, bufSize);
    workersocket.send(partitionData);
}

void
LambdaWorker::recvValidationResults(zmq::message_t& client_d, zmq::message_t& header) {
    // Send empty ack message
    zmq::message_t confirm;
    workersocket.send(confirm);

    unsigned totalCorrectThisPartition = parse<unsigned>((char*)header.data(), 2);
    float lossThisPartition = parse<float>((char*)header.data(), 3);

    // atomically sum correct predictions and loss
    std::lock_guard<std::mutex> lk(eval_mutex);
    numCorrectPredictions += totalCorrectThisPartition;
    totalLoss += lossThisPartition;

    --evalLambdas;

    // If we have received all validation results, calculate the actual loss
    // and accuracy
    // NOTE: Will only be acc/loss per node, not global acc/loss
    if (evalLambdas == 0) {
        printLog(nodeId, "Accuracy this epoch: %f\nLoss this epoch %f\n",
                 (float) numCorrectPredictions / numValidationVertices,
                 totalLoss / numValidationVertices);
    }
}
