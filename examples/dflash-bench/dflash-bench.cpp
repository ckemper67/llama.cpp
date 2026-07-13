// llama-dflash-bench
//
// Throughput benchmarking / tuning harness for DFlash (diffusion-block) speculative
// decoding. Loads the target + draft models ONCE and then runs a fixed generation for
// each requested config, so draft length and sampling can be tweaked per run without
// reloading. Reports decode t/s, draft acceptance, and mean accepted length.
//
// v1: single config (validates the dflash decode loop). Sweep driver added on top.
//
// The DFlash decode loop differs from a classic draft model: the draft is NOT produced
// by re-running a draft model over the same tokens. Instead, the target model's hidden
// features are extracted during its decode and injected into the draft via
// common_speculative_process(). This mirrors the server's generate step; getting this
// wrong is exactly what makes examples/speculative-simple segfault on DFlash.

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

static std::vector<int> parse_int_list(const std::string & s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        std::string tok = s.substr(i, j - i);
        if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
        i = j + 1;
    }
    return out;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct bench_result {
    int    n_max      = 0;
    float  temp       = 0.0f;
    double tps_decode = 0.0;   // generated tokens / decode time
    double accept_pct = 0.0;   // n_accept / n_drafted
    double mean_len   = 0.0;   // generated tokens / verify cycles
    int    n_predict  = 0;
    int    n_drafted  = 0;
    int    n_accept   = 0;
    int    n_cycles   = 0;
};

// Run one generation with the currently-loaded models. Assumes ctx_tgt/ctx_dft/spec are
// set up. Resets KV + speculator state at the start so it can be called repeatedly.
static bench_result run_config(
        common_params  &  params,
        llama_model    *  model_tgt,
        llama_context  *  ctx_tgt,
        llama_context  *  ctx_dft,
        common_speculative * spec,
        const llama_vocab * vocab,
        const std::vector<llama_token> & inp,
        int   cfg_n_max,
        const common_params_sampling & sparams) {
    const llama_seq_id seq_id = 0;

    // fresh KV for both contexts
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, -1, -1);
    llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, -1, -1);

    // fresh sampler for this config's sampling params (init takes a non-const ref)
    common_params_sampling sparams_local = sparams;
    common_sampler_ptr smpl(common_sampler_init(model_tgt, sparams_local));

    const bool need_embd = common_speculative_need_embd(spec);

    // whether the context requires whole-sequence removal (-> checkpoints) or supports
    // partial removal (-> simple trim). Mirrors speculative-simple.
    const bool use_ckpt_tgt = (common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);
    const bool use_ckpt_dft = (common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    // ---- prompt eval on the target; seed draft features via process() ----
    // Note: must be a fully-formed batch (per-token seq_id/n_seq_id), NOT
    // llama_batch_get_one(): common_speculative_process() indexes batch.seq_id[k][0].
    {
        llama_batch pb = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
        for (size_t i = 0; i + 1 < inp.size(); ++i) {
            common_batch_add(pb, inp[i], (llama_pos) i, { seq_id }, true);
        }
        llama_set_embeddings(ctx_tgt, need_embd);
        llama_decode(ctx_tgt, pb);
        common_speculative_process(spec, pb);
        llama_batch_free(pb);
    }

    llama_token id_last = inp.back();

    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = (int) inp.size() - 1;

    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;
    int n_cycles  = 0;
    bool has_eos  = false;

    size_t n_draft = 0;
    llama_tokens draft;
    common_prompt_checkpoint ckpt;

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // ---- produce a draft block ----
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting = */ true,
                /* .n_max    = */ cfg_n_max,   // per-request draft-length override
                /* .n_past   = */ n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ &prompt_tgt,
                /* .result   = */ &draft,
            };
            common_speculative_draft(spec);

            n_draft = draft.size();

            if (!draft.empty() && use_ckpt_tgt) {
                ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
        }

        // ---- build [id_last, draft0..draftN-1] and verify on the target ----
        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, id_last, n_past++, { seq_id }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
        }

        llama_set_embeddings(ctx_tgt, need_embd);
        llama_decode(ctx_tgt, batch_tgt);
        // DFlash: inject target features into the draft (replaces a draft-model decode)
        common_speculative_process(spec, batch_tgt);

        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        GGML_ASSERT(ids.size() > 0);

        n_cycles++;

        // partial acceptance without partial-rm support: restore checkpoint, retry
        if (use_ckpt_tgt && ids.size() - 1 < draft.size()) {
            draft = std::move(ids);

            ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);
            n_past = (int) prompt_tgt.size();
            continue;
        }

        common_speculative_accept(spec, seq_id, ids.size() - 1);

        n_past    += ids.size() - 1;
        n_drafted += n_draft;
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }
        }

        draft.clear();

        // trim any KV past the accepted tokens
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);

        if ((params.n_predict >= 0 && n_predict >= params.n_predict) || has_eos) {
            break;
        }
    }

    const auto t_dec_end = ggml_time_us();
    llama_batch_free(batch_tgt);

    const double dt = (t_dec_end - t_dec_start) / 1e6;

    bench_result r;
    r.n_max      = cfg_n_max;
    r.temp       = sparams.temp;
    r.n_predict  = n_predict;
    r.n_drafted  = n_drafted;
    r.n_accept   = n_accept;
    r.n_cycles   = n_cycles;
    r.tps_decode = dt > 0 ? n_predict / dt : 0.0;
    r.accept_pct = n_drafted > 0 ? 100.0 * n_accept / n_drafted : 0.0;
    r.mean_len   = n_cycles  > 0 ? (double) n_predict / n_cycles : 0.0;
    return r;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    // Extract our custom sweep args before common_params_parse (which rejects unknowns).
    std::string sweep_n_max_str = "1,2,4,6,8,12,15";
    int         repeats         = 3;
    std::vector<char *> fwd_argv;
    fwd_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--sweep-n-max") == 0 && i + 1 < argc) {
            sweep_n_max_str = argv[++i];
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        } else {
            fwd_argv.push_back(argv[i]);
        }
    }

    if (!common_params_parse((int) fwd_argv.size(), fwd_argv.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < 0) {
        params.n_predict = 256;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // ---- load target (once) ----
    auto llama_init_tgt = common_init_from_params(params);
    llama_model   * model_tgt = llama_init_tgt->model();
    llama_context * ctx_tgt   = llama_init_tgt->context();
    if (model_tgt == nullptr) {
        LOG_ERR("%s: failed to load target model\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // ---- set up the draft/speculative context via the same path the server uses ----
    common_params params_dft = common_base_params_to_speculative(params);
    auto spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
    llama_context * ctx_dft = spec_init->context();
    if (ctx_dft == nullptr) {
        LOG_ERR("%s: failed to initialize draft/speculative context\n", __func__);
        return 1;
    }

    params.speculative.draft.ctx_tgt = ctx_tgt;
    params.speculative.draft.ctx_dft = ctx_dft;

    // launch the speculator at the largest n_max in the sweep (clamped internally to
    // block_size-1); each config then caps it DOWN via the per-request override.
    std::vector<int> sweep_n_max = parse_int_list(sweep_n_max_str);
    if (sweep_n_max.empty()) sweep_n_max = { params.speculative.draft.n_max };
    params.speculative.draft.n_max = *std::max_element(sweep_n_max.begin(), sweep_n_max.end());

    common_speculative * spec = common_speculative_init(params.speculative, 1);
    if (spec == nullptr) {
        LOG_ERR("%s: failed to initialize speculator\n", __func__);
        return 1;
    }

    // ---- tokenize the prompt (once) ----
    std::vector<llama_token> inp = common_tokenize(ctx_tgt, params.prompt, true, true);
    if (inp.empty()) {
        LOG_ERR("%s: empty prompt\n", __func__);
        return 1;
    }
    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: prompt (%d) exceeds context (%d)\n", __func__, (int) inp.size(), (int) llama_n_ctx(ctx_tgt));
        return 1;
    }

    // ---- Tier-B scenarios (sampling presets held fixed while we tune Tier-A) ----
    // Speculative decoding is exact, so n_max only affects speed, never output;
    // sampling defines the output distribution and is treated as an exogenous scenario.
    struct scenario { std::string name; common_params_sampling sp; };
    std::vector<scenario> scenarios;
    {
        common_params_sampling greedy = params.sampling;
        greedy.temp = 0.0f; greedy.top_k = 1;
        greedy.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
        scenarios.push_back({ "greedy", greedy });

        common_params_sampling mid = params.sampling;
        mid.temp = 0.7f; mid.top_k = 20; mid.top_p = 0.95f; mid.min_p = 0.0f;
        mid.penalty_present = 0.0f;
        scenarios.push_back({ "mid(t0.7)", mid });
    }

    LOG_INF("\n=== dflash-bench: n_predict=%d repeats=%d n_max_sweep=[%s] ===\n",
            params.n_predict, repeats, sweep_n_max_str.c_str());

    // warm-up (graph alloc / first-run cost), discarded
    run_config(params, model_tgt, ctx_tgt, ctx_dft, spec, vocab, inp,
               sweep_n_max.front(), scenarios.front().sp);

    std::vector<bench_result> best_per_scenario;

    for (const auto & sc : scenarios) {
        LOG_INF("\n--- scenario: %s ---\n", sc.name.c_str());
        LOG_INF("%-6s %-10s %-9s %-9s %-8s\n", "n_max", "t/s", "accept%", "mean_len", "gen");

        bench_result best; best.tps_decode = -1.0;
        for (int nm : sweep_n_max) {
            std::vector<double> tps, acc, mlen;
            bench_result last;
            for (int rep = 0; rep < repeats; ++rep) {
                last = run_config(params, model_tgt, ctx_tgt, ctx_dft, spec, vocab, inp, nm, sc.sp);
                tps.push_back(last.tps_decode);
                acc.push_back(last.accept_pct);
                mlen.push_back(last.mean_len);
            }
            bench_result r = last;
            r.tps_decode = median(tps);
            r.accept_pct = median(acc);
            r.mean_len   = median(mlen);
            LOG_INF("%-6d %-10.2f %-9.2f %-9.2f %-8d\n",
                    r.n_max, r.tps_decode, r.accept_pct, r.mean_len, r.n_predict);
            if (r.tps_decode > best.tps_decode) best = r;
        }
        LOG_INF("  best: n_max=%d -> %.2f t/s (accept %.1f%%, mean_len %.2f)\n",
                best.n_max, best.tps_decode, best.accept_pct, best.mean_len);
        best_per_scenario.push_back(best);
    }

    LOG_INF("\n=== recommended settings ===\n");
    for (size_t i = 0; i < scenarios.size(); ++i) {
        const auto & b = best_per_scenario[i];
        LOG_INF("scenario %-10s : spec-draft-n-max = %d   (%.1f t/s, accept %.1f%%, mean_len %.2f)\n",
                scenarios[i].name.c_str(), b.n_max, b.tps_decode, b.accept_pct, b.mean_len);
    }

    common_speculative_free(spec);
    llama_backend_free();
    return 0;
}
