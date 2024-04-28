#include <stddef.h>
#include "scann/utils/common.h"

typedef struct _scann Scann;
typedef research_scann::DatapointIndex DatapointIndex;

typedef enum {
    SCANN_SUCCESS = 0,
    SCANN_ERROR = 1,
} ScannStatus;

extern "C" {
    ScannStatus scann_new(Scann **ptr);

    void scann_free();

    ScannStatus scann_initialize(Scann* scann, const float* dataset, size_t
            rows, size_t cols, const char* config, int training_threads);

    void scann_initialize_proto();

    void scann_search();

    ScannStatus scann_search_batched(Scann* scann, DatapointIndex** indices,
            float** distances, const float* queries, size_t rows, size_t cols, int
            final_nn, int pre_reorder_nn, int leaves, bool parallel, int batch_size);

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
}
