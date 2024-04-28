#include "scann/scann_ops/cc/scann.h"
#include "scann/scann_ops/cc/scannffi.h"
#include "scann/utils/common.h"

#define INTERFACE_CAST(ptr) reinterpret_cast<research_scann::ScannInterface*>(ptr)
#define SUCCESS 0
#define FAILURE 1


// TODO: Expose error messages.
// TODO: Prevent unnecessary deep copies.

ScannStatus scann_new(Scann** scann_ptr) {
    try {
        auto interface = new research_scann::ScannInterface();
        *scann_ptr = reinterpret_cast<Scann*>(interface);
        return SCANN_SUCCESS;
    } catch (std::bad_alloc& e) {
        return SCANN_ERROR;
    }
}

void scann_free(Scann* scann) {
    auto interface = INTERFACE_CAST(scann);
    delete interface;
}

// XXX: Dataset must be two-dimensional
ScannStatus scann_initialize(Scann* scann, const float* dataset, size_t rows,
        size_t cols, const char* config, int training_threads)
{
    research_scann::ConstSpan<float> span(dataset, rows * cols);
    std::string _config(config);

    auto interface = INTERFACE_CAST(scann);
    auto status { interface->Initialize(span, rows, _config, training_threads) };

    return (status.ok()) ? SCANN_SUCCESS : SCANN_ERROR;
}

void scann_initialize_proto();

void scann_search();

// XXX: Queries must be two-dimensional
ScannStatus scann_search_batched(Scann* scann, const float* queries, size_t rows,
        size_t cols, int final_nn, int pre_reorder_nn, int leaves, bool
        parallel, int batch_size)
{
    auto interface = INTERFACE_CAST(scann);

    size_t size { rows * cols };
    std::vector<float> queries_vec(queries, queries + size);
    auto query_dataset { research_scann::DenseDataset<float>(queries_vec, rows) };
    std::vector<research_scann::NNResultsVector> res(query_dataset.size());

    absl::Status status;

    if (parallel)
        status = interface->SearchBatchedParallel(query_dataset, research_scann::MakeMutableSpan(res), final_nn, pre_reorder_nn, leaves, batch_size);
    else
        status = interface->SearchBatched(query_dataset, research_scann::MakeMutableSpan(res), final_nn, pre_reorder_nn, leaves);

    if (!status.ok()) return SCANN_ERROR;

    for (const auto& nn_res : res)
        final_nn = std::max<int>(final_nn, nn_res.size());

    // TODO

    /* pybind11::array_t<research_scann::DatapointIndex> indices( {static_cast<long>(query_dataset.size()), static_cast<long>(final_nn)} ); */
    /* pybind11::array_t<float> distances(  {static_cast<long>(query_dataset.size()), static_cast<long>(final_nn)}); */

    /* auto idx_ptr = reinterpret_cast<research_scann::DatapointIndex*>(indices.request().ptr); */
    /* auto dis_ptr = reinterpret_cast<float*>(distances.request().ptr); */

    /* interface->ReshapeBatchedNNResult(absl::MakeConstSpan(res), idx_ptr, dis_ptr, final_nn); */

    return SCANN_SUCCESS;
}

void scann_search_batched_parallel();

void scann_serialize();

void scann_upsert();

void scann_delete();

void scann_rebalance();

void scann_size();

void scann_set_num_threads();

void scann_reserve();

void scann_config();

void scann_suggest_autopilot();
