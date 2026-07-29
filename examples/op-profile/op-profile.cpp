// op-profile: per-ggml-op wall-clock time breakdown for a decode step.
//
// Loads a model, runs a prefill pass (uninstrumented), then runs n_predict
// decode steps with a cb_eval callback attached that times every node.
// Setting a callback forces the scheduler to synchronize after each node
// (see ggml_backend_sched_compute_splits in ggml-backend.cpp), so per-op
// timings here are relative/comparative, not representative of steady-state
// pipelined throughput -- cross-check the top few ops with
// GGML_METAL_CAPTURE_COMPUTE for hardware-accurate absolute numbers.

#include "llama.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

using clock_type = std::chrono::steady_clock;

struct op_stat {
    uint64_t calls = 0;
    double   total_ms = 0.0;
};

struct profile_ctx {
    std::map<std::string, op_stat> by_op;
    clock_type::time_point last;
    bool have_last = false;
    uint64_t total_calls = 0;
    double total_ms = 0.0;
    // false during prefill: skip timing/sync overhead until the decode loop begins
    bool active = false;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * pc = (profile_ctx *) user_data;

    if (!pc->active) {
        // don't force a per-node sync during (uninstrumented) prefill
        return false;
    }

    if (ask) {
        // we want timing data for every node
        return true;
    }

    auto now = clock_type::now();
    if (pc->have_last) {
        double ms = std::chrono::duration<double, std::milli>(now - pc->last).count();
        auto & st = pc->by_op[ggml_op_name(t->op)];
        st.calls++;
        st.total_ms += ms;
        pc->total_calls++;
        pc->total_ms += ms;
    }
    pc->last = now;
    pc->have_last = true;

    return true;
}

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "\nusage: %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [-c n_ctx] [prompt]\n\n"
        "Profiles per-ggml-op wall-clock time during decode (prefill excluded).\n",
        argv0);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "Explain how a Raft consensus leader election works, step by step.";
    int ngl = 999;
    int n_predict = 64;
    int n_ctx_arg = 4096;

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
                model_path = argv[++i];
            } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_predict = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
                ngl = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
                n_ctx_arg = atoi(argv[++i]);
            } else {
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argv[0]);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        fprintf(stderr, "error: unable to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), prompt_tokens.data(), (int) prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "error: failed to tokenize prompt\n");
        return 1;
    }

    profile_ctx pc;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = n_prompt + n_predict + n_ctx_arg;
    ctx_params.n_batch = std::max(n_prompt, 512);
    ctx_params.no_perf = false;
    // cb_eval must be set at context creation (no runtime setter). pc.active
    // stays false through prefill so the callback returns false on "ask" and
    // the scheduler doesn't pay per-node sync overhead until we flip it below.
    ctx_params.cb_eval = cb_eval;
    ctx_params.cb_eval_user_data = &pc;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to create context\n");
        return 1;
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // prefill (uninstrumented: pc.active is still false)
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int) prompt_tokens.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "error: prefill decode failed\n");
        return 1;
    }

    fprintf(stderr, "prefill done (%d tokens). starting instrumented decode of %d tokens...\n",
        n_prompt, n_predict);

    pc.active = true;

    llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

    const auto t_decode_start = clock_type::now();
    int n_decode = 0;
    for (int i = 0; i < n_predict; i++) {
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }
        llama_batch b1 = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(ctx, b1)) {
            fprintf(stderr, "error: decode failed at step %d\n", i);
            return 1;
        }
        n_decode++;
        new_token_id = llama_sampler_sample(smpl, ctx, -1);
    }
    const auto t_decode_end = clock_type::now();
    double wall_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

    // report
    std::vector<std::pair<std::string, op_stat>> rows(pc.by_op.begin(), pc.by_op.end());
    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
        return a.second.total_ms > b.second.total_ms;
    });

    printf("\n=== op-profile: %d decode steps, prompt=%d tokens ===\n", n_decode, n_prompt);
    printf("wall clock (decode loop, includes sync overhead from instrumentation): %.1f ms  (%.2f tok/s)\n",
        wall_ms, n_decode > 0 ? (1000.0 * n_decode / wall_ms) : 0.0);
    printf("sum of per-op instrumented time: %.1f ms (%.1fx wall -- sync-per-node overhead)\n\n",
        pc.total_ms, wall_ms > 0 ? pc.total_ms / wall_ms : 0.0);

    printf("%-24s %10s %12s %10s %10s\n", "op", "calls", "total_ms", "avg_us", "% total");
    printf("%-24s %10s %12s %10s %10s\n", "------------------------", "----------", "------------", "----------", "----------");
    for (auto & [name, st] : rows) {
        double pct = pc.total_ms > 0 ? 100.0 * st.total_ms / pc.total_ms : 0.0;
        double avg_us = st.calls > 0 ? (st.total_ms * 1000.0 / st.calls) : 0.0;
        printf("%-24s %10llu %12.2f %10.1f %9.1f%%\n",
            name.c_str(), (unsigned long long) st.calls, st.total_ms, avg_us, pct);
    }
    printf("\n");

    llama_perf_context_print(ctx);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
