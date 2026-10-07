// llama-roofline: where the time of a decode (or N-token verify) step goes, and which limit each op hits
//
// 1. harvest: run one real step of the model and record every dispatched op through the scheduler's
//    eval callback (ask phase only, so no extra synchronization), in graph order
// 2. deduplicate the ops into signatures (op, op params, types, shapes, strides) with a count per step
// 3. cost each signature analytically: bytes moved and FLOPs
// 4. probe the device ceilings: read bandwidth, matmul throughput, cost of one dependent dispatch
// 5. replay the heavy ops of the step as one sequence in their real order, reading the model's own
//    weights, and time each signature by leaving it out of the sequence. a kernel's time then includes
//    what it pays for starting after different work, which a replay of one kernel back to back hides
// 6. replay each heavy signature alone: R independent copies with their own buffers, so the working
//    set exceeds the device caches, each copy its own node. a throughput figure, for comparison
// 7. report: heavy signatures ranked by time per step with their limit, and the residual of the
//    measured end-to-end step time that the heavy ops do not explain (small ops, launch overhead)
//
// the tool is backend-neutral: it only uses llama.cpp and ggml-backend APIs

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

//
// options
//

struct options {
    std::string model;
    std::string json_out;
    std::string device;              // replay device name (default: first GPU)
    int         n_gpu_layers = 99;
    int         depth        = 0;    // KV depth before the measured step
    int         n_tokens     = 1;    // tokens per step (1 = decode, >1 = verify / batched)
    bool        flash_attn   = true;
    int         n_steps      = 32;   // end-to-end steps to time
    int         reps         = 5;    // replay repetitions (median)
    double      cache_bust   = 512.0;  // MiB per replayed signature, to defeat device caches
    double      mem_cap      = 4096.0; // MiB cap per replayed signature
    bool        serial       = true;   // ask backends not to overlap the independent replay copies
};

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m model.gguf [options]\n"
        "  -m, --model PATH      model to analyze\n"
        "  -d, --depth N         KV depth before the measured step (default: 0)\n"
        "  -n, --n-tokens N      tokens per step: 1 = decode, >1 = verify (default: 1)\n"
        "  -ngl N                layers to offload (default: 99)\n"
        "  -fa on|off            flash attention (default: on)\n"
        "  --steps N             end-to-end steps to time (default: 32)\n"
        "  --reps N              replay repetitions, median (default: 5)\n"
        "  --cache-bust MIB      working set per replayed signature (default: 512)\n"
        "  --device NAME         replay device (default: first GPU)\n"
        "  -o, --json PATH       also write the report as JSON\n", argv0);
}

static bool parse_args(int argc, char ** argv, options & o) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: missing value for %s\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") {
            o.model = next();
        } else if (a == "-d" || a == "--depth") {
            o.depth = atoi(next());
        } else if (a == "-n" || a == "--n-tokens") {
            o.n_tokens = atoi(next());
        } else if (a == "-ngl") {
            o.n_gpu_layers = atoi(next());
        } else if (a == "-fa") {
            o.flash_attn = std::string(next()) != "off";
        } else if (a == "--steps") {
            o.n_steps = atoi(next());
        } else if (a == "--reps") {
            o.reps = atoi(next());
        } else if (a == "--cache-bust") {
            o.cache_bust = atof(next());
        } else if (a == "--serial") {
            o.serial = std::string(next()) != "off";
        } else if (a == "--device") {
            o.device = next();
        } else if (a == "-o" || a == "--json") {
            o.json_out = next();
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument %s\n", a.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (o.model.empty() || o.n_tokens < 1 || o.depth < 0 || o.n_steps < 1 || o.reps < 1) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

//
// harvested op descriptions
//

struct tensor_desc {
    bool      present = false;
    ggml_type type    = GGML_TYPE_F32;
    int64_t   ne[4]   = { 0, 0, 0, 0 };
    size_t    nb[4]   = { 0, 0, 0, 0 };
    bool      weight  = false; // lives in a weights buffer (read from DRAM every step)
};

struct op_desc {
    ggml_op     op = GGML_OP_NONE;
    int32_t     op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)] = {};
    tensor_desc dst;
    tensor_desc src[GGML_MAX_SRC];
    std::string buft;  // buffer type of the destination (which backend ran it)
    std::string name;  // first tensor name seen for this signature
};

static tensor_desc describe(const ggml_tensor * t) {
    tensor_desc d;
    if (!t) {
        return d;
    }
    d.present = true;
    d.type    = t->type;
    for (int i = 0; i < 4; i++) {
        d.ne[i] = t->ne[i];
        d.nb[i] = t->nb[i];
    }
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    d.weight = base->buffer && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    return d;
}

static std::string key_of(const op_desc & d) {
    std::string k = ggml_op_name(d.op);
    char buf[256];
    for (int i = 0; i < (int) (GGML_MAX_OP_PARAMS / sizeof(int32_t)); i++) {
        snprintf(buf, sizeof(buf), ":%d", d.op_params[i]);
        k += buf;
    }
    auto add = [&](const tensor_desc & t) {
        if (!t.present) {
            k += "|-";
            return;
        }
        snprintf(buf, sizeof(buf), "|%s[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "][%zu,%zu,%zu,%zu]%c",
                 ggml_type_name(t.type), t.ne[0], t.ne[1], t.ne[2], t.ne[3], t.nb[0], t.nb[1], t.nb[2], t.nb[3],
                 t.weight ? 'w' : 'a');
        k += buf;
    };
    add(d.dst);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add(d.src[i]);
    }
    return k + "|" + d.buft;
}

// one dispatched node of the harvested step, in graph order
struct seq_entry {
    std::string   key;                     // its signature
    ggml_tensor * real[GGML_MAX_SRC] = {}; // model weight tensors among its sources (owned by the model)
};

struct harvest {
    bool                             recording = false;
    std::map<std::string, op_desc>   sigs;
    std::map<std::string, int>       count;
    std::vector<std::string>         order; // first-seen order
    std::vector<seq_entry>           seq;   // every node of the step, in order
    int                              n_nodes = 0;
};

static bool is_empty_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE;
}

// record in the ask phase and never request the data: the scheduler then computes each split as one
// graph, exactly as without a callback
static bool cb_harvest(ggml_tensor * t, bool ask, void * user_data) {
    auto * h = (harvest *) user_data;
    if (!ask || !h->recording || is_empty_op(t->op)) {
        return false;
    }

    op_desc d;
    d.op = t->op;
    memcpy(d.op_params, t->op_params, sizeof(d.op_params));
    d.dst = describe(t);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        d.src[i] = describe(t->src[i]);
    }
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    d.buft = base->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(base->buffer)) : "?";
    d.name = t->name;

    const std::string k = key_of(d);
    if (h->sigs.find(k) == h->sigs.end()) {
        h->sigs[k] = d;
        h->order.push_back(k);
    }
    h->count[k]++;
    h->n_nodes++;

    seq_entry e;
    e.key = k;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // only tensors the model owns outlive this graph; views are rebuilt with every graph
        if (d.src[i].weight && t->src[i] && !t->src[i]->view_src) {
            e.real[i] = t->src[i];
        }
    }
    h->seq.push_back(e);

    return false;
}

//
// analytic cost
//

static bool is_heavy(const op_desc & d) {
    switch (d.op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
            return true;
        default:
            return false;
    }
}

static double nbytes_of(const tensor_desc & t) {
    if (!t.present) {
        return 0.0;
    }
    // bytes touched by a (possibly strided) view: whole rows times the row count
    const double rows = (double) t.ne[1] * t.ne[2] * t.ne[3];
    return rows * (double) ggml_row_size(t.type, t.ne[0]);
}

// expected share of the experts a batch touches with uniform routing: 1 - (1 - k/E)^n
static double expert_fraction(const op_desc & d) {
    const double E = (double) d.src[0].ne[2];
    const double k = (double) d.src[2].ne[0];
    const double n = (double) d.src[2].ne[1];
    if (E <= 0 || k <= 0) {
        return 1.0;
    }
    return 1.0 - std::pow(1.0 - std::min(1.0, k / E), n);
}

static double bytes_of(const op_desc & d) {
    double b = nbytes_of(d.dst);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        double s = nbytes_of(d.src[i]);
        if (d.op == GGML_OP_MUL_MAT_ID && i == 0) {
            s *= expert_fraction(d);
        }
        b += s;
    }
    return b;
}

static double flops_of(const op_desc & d) {
    switch (d.op) {
        case GGML_OP_MUL_MAT:
            return 2.0 * d.src[0].ne[0] * d.dst.ne[0] * d.dst.ne[1] * d.dst.ne[2] * d.dst.ne[3];
        case GGML_OP_MUL_MAT_ID:
            // dst: [n_rows, n_used, n_tokens]
            return 2.0 * d.src[0].ne[0] * d.dst.ne[0] * d.dst.ne[1] * d.dst.ne[2];
        case GGML_OP_FLASH_ATTN_EXT: {
            // q: [dk, n_q, n_head, n_seq], k: [dk, n_kv, ...], v: [dv, n_kv, ...]
            const double n_q   = (double) d.src[0].ne[1] * d.src[0].ne[2] * d.src[0].ne[3];
            const double n_kv  = (double) d.src[1].ne[1];
            return 2.0 * n_q * n_kv * (double) (d.src[1].ne[0] + d.src[2].ne[0]);
        }
        default:
            return (double) d.dst.ne[0] * d.dst.ne[1] * d.dst.ne[2] * d.dst.ne[3];
    }
}

static std::string shape_str(const tensor_desc & t) {
    char buf[128];
    if (t.ne[2] == 1 && t.ne[3] == 1) {
        snprintf(buf, sizeof(buf), "%" PRId64 "x%" PRId64, t.ne[0], t.ne[1]);
    } else {
        snprintf(buf, sizeof(buf), "%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64, t.ne[0], t.ne[1], t.ne[2], t.ne[3]);
    }
    return buf;
}

static std::string label_of(const op_desc & d) {
    std::string s = ggml_op_name(d.op);
    s += " ";
    s += ggml_type_name(d.src[0].type);
    s += " " + shape_str(d.src[0]);
    if (d.src[1].present) {
        s += " * " + shape_str(d.src[1]);
    }
    return s;
}

//
// replay
//

static ggml_backend_dev_t pick_device(const std::string & name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (!name.empty()) {
            if (name == ggml_backend_dev_name(d)) {
                return d;
            }
        } else if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return d;
        }
    }
    return name.empty() ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) : nullptr;
}

static std::mt19937 rng(42);

static void fill_float(std::vector<float> & v, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto & x : v) {
        x = dist(rng);
    }
}

// host bytes for a tensor of `type` with ne0 per row and nrows rows. quantized rows are produced for a
// bounded number of rows and repeated: the byte traffic is what matters for timing, not the values
static std::vector<uint8_t> make_data(ggml_type type, int64_t ne0, int64_t nrows, float scale) {
    const size_t row = ggml_row_size(type, ne0);
    std::vector<uint8_t> out(row * nrows);

    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        std::vector<float> f(ne0 * std::min<int64_t>(nrows, 64));
        fill_float(f, scale);
        const int64_t nr = (int64_t) (f.size() / ne0);
        std::vector<uint8_t> blk(row * nr);
        if (type == GGML_TYPE_F32) {
            memcpy(blk.data(), f.data(), blk.size());
        } else if (type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) blk.data(), (int64_t) f.size());
        } else {
            ggml_fp32_to_bf16_row(f.data(), (ggml_bf16_t *) blk.data(), (int64_t) f.size());
        }
        for (int64_t r = 0; r < nrows; r += nr) {
            memcpy(out.data() + r * row, blk.data(), row * std::min(nr, nrows - r));
        }
        return out;
    }

    if (ggml_is_quantized(type)) {
        const int64_t      nr = std::min<int64_t>(nrows, 32);
        std::vector<float> f(ne0 * nr);
        fill_float(f, scale);
        std::vector<float> imatrix(ne0, 1.0f);
        std::vector<uint8_t> blk(row * nr);
        ggml_quantize_chunk(type, f.data(), blk.data(), 0, nr, ne0,
                            ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr);
        for (int64_t r = 0; r < nrows; r += nr) {
            memcpy(out.data() + r * row, blk.data(), row * std::min(nr, nrows - r));
        }
        return out;
    }

    return out; // other types (I32 etc.) are filled by the caller
}

static const int MAX_REPLAY_NODES = 4096; // nodes per replay graph, repetitions included

static double time_graph(ggml_backend_t backend, ggml_cgraph * gf, int reps);

struct replay_result {
    bool        ok      = false;
    std::string why;
    double      us      = 0.0; // per op
    int         copies  = 0;
};

// why an op cannot be rebuilt from its description, or empty
static std::string replay_unsupported(const op_desc & d) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const tensor_desc & s = d.src[i];
        if (!s.present) {
            continue;
        }
        // integer inputs we know how to fill with valid values
        if (s.type == GGML_TYPE_I32 && !(d.op == GGML_OP_MUL_MAT_ID && i == 2)) {
            return "integer input without a known valid range";
        }
        if (s.nb[0] != ggml_type_size(s.type) && !ggml_is_quantized(s.type)) {
            return "non-unit innermost stride";
        }
    }
    return "";
}

struct src_alloc {
    ggml_tensor * base  = nullptr; // allocated tensor that holds the data, null for a real model tensor
    int64_t       ne0   = 0;       // per-row elements of base
    int64_t       nrows = 0;       // rows of base
};

// one instance of the op in ctx: sources with a real tensor use it, the others get their own allocation
// (contiguous, or the extent a strided view spans); allocs receives one entry per present source
static ggml_tensor * make_instance(ggml_context * ctx, const op_desc & d, ggml_tensor * const * real,
                                   std::vector<src_alloc> & allocs) {
    ggml_tensor * src_t[GGML_MAX_SRC] = {};
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const tensor_desc & s = d.src[i];
        if (!s.present) {
            continue;
        }
        if (real && real[i]) {
            src_t[i] = real[i];
            allocs.push_back({});
            continue;
        }
        const size_t row_bytes  = ggml_row_size(s.type, s.ne[0]);
        const bool   contiguous = s.nb[0] == ggml_type_size(s.type) && s.nb[1] == row_bytes &&
                                s.nb[2] == s.nb[1] * s.ne[1] && s.nb[3] == s.nb[2] * s.ne[2];
        if (contiguous) {
            ggml_tensor * t = ggml_new_tensor(ctx, s.type, 4, s.ne);
            src_t[i]        = t;
            allocs.push_back({ t, s.ne[0], s.ne[1] * s.ne[2] * s.ne[3] });
        } else {
            // a strided view (e.g. into the KV cache): allocate the extent it spans, view into it
            size_t extent = row_bytes;
            for (int k = 1; k < 4; k++) {
                extent += (size_t) (s.ne[k] - 1) * s.nb[k];
            }
            const int64_t ne0  = s.ne[0];
            const int64_t rows = (int64_t) ((extent + row_bytes - 1) / row_bytes);
            ggml_tensor * base = ggml_new_tensor_2d(ctx, s.type, ne0, rows);
            src_t[i] = ggml_view_4d(ctx, base, s.ne[0], s.ne[1], s.ne[2], s.ne[3], s.nb[1], s.nb[2], s.nb[3], 0);
            allocs.push_back({ base, ne0, rows });
        }
    }

    ggml_tensor * out = ggml_new_tensor(ctx, d.dst.type, 4, d.dst.ne);
    out->op           = d.op;
    memcpy(out->op_params, d.op_params, sizeof(d.op_params));
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        out->src[i] = src_t[i];
    }
    return out;
}

// host image for source i of the op, sized for allocation a
static std::vector<uint8_t> make_src_image(const op_desc & d, int i, const src_alloc & a) {
    const tensor_desc & s = d.src[i];
    if (s.type == GGML_TYPE_I32) {
        // MUL_MAT_ID ids: [n_used, n_tokens], distinct experts per token
        const int64_t n_expert = d.src[0].ne[2];
        std::vector<int32_t> ids(ggml_nelements(a.base));
        std::vector<int32_t> perm(n_expert);
        for (int64_t r = 0; r < (int64_t) ids.size() / a.ne0; r++) {
            for (int64_t e = 0; e < n_expert; e++) {
                perm[e] = (int32_t) e;
            }
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int64_t j = 0; j < a.ne0; j++) {
                ids[r * a.ne0 + j] = perm[j % n_expert];
            }
        }
        std::vector<uint8_t> data(ids.size() * sizeof(int32_t));
        memcpy(data.data(), ids.data(), data.size());
        return data;
    }
    // small values keep recurrent and exponential ops finite
    return make_data(s.type, a.ne0, a.nrows, s.weight ? 1.0f : 0.1f);
}

// the op alone: `copies` independent instances, each with its own source buffers, timed as one graph;
// returns the median time per instance. back to back, the same kernel never starts after a different
// one, so this is a throughput figure: it hides the cost a kernel pays when it follows other work
static replay_result replay(ggml_backend_t backend, const op_desc & d, const options & o) {
    replay_result res;

    res.why = replay_unsupported(d);
    if (!res.why.empty()) {
        return res;
    }

    const double per_copy = std::max(1.0, bytes_of(d));
    int copies = (int) std::ceil(o.cache_bust * 1024 * 1024 / per_copy);
    copies     = std::max(4, std::min(copies, 256));
    copies     = std::max(1, std::min<int>(copies, (int) (o.mem_cap * 1024 * 1024 / per_copy)));

    ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead() * (size_t) copies * (2 * GGML_MAX_SRC + 2) +
                                ggml_graph_overhead_custom((size_t) copies * (GGML_MAX_SRC + 2) + MAX_REPLAY_NODES + 16, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(ip));

    std::vector<std::vector<src_alloc>> srcs(copies);
    std::vector<ggml_tensor *>          outs(copies);
    for (int c = 0; c < copies; c++) {
        outs[c] = make_instance(ctx.get(), d, nullptr, srcs[c]);
    }

    if (!ggml_backend_supports_op(backend, outs[0])) {
        res.why = "not supported by the replay backend";
        return res;
    }

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) {
        res.why = "allocation failed";
        return res;
    }

    // one host image per source, uploaded to every copy
    for (size_t k = 0, i = 0; i < GGML_MAX_SRC; i++) {
        if (!d.src[i].present) {
            continue;
        }
        const std::vector<uint8_t> data = make_src_image(d, (int) i, srcs[0][k]);
        for (int c = 0; c < copies; c++) {
            ggml_backend_tensor_set(srcs[c][k].base, data.data(), 0, std::min(data.size(), ggml_nbytes(srcs[c][k].base)));
        }
        k++;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), (size_t) copies * (GGML_MAX_SRC + 2) + MAX_REPLAY_NODES + 16, false);
    for (int c = 0; c < copies; c++) {
        ggml_build_forward_expand(gf, outs[c]);
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        res.why = "compute failed";
        return res;
    }
    ggml_backend_synchronize(backend);

    // repeat the copies inside the graph until one graph runs for ~30 ms: the host round trip of a
    // graph compute is then negligible. the copies together exceed the device caches, so a repeated
    // pass still reads from DRAM
    const int64_t t0  = ggml_time_us();
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    const double once = std::max(1.0, (double) (ggml_time_us() - t0));
    const int    rep  = std::max(1, std::min((int) (30000.0 / once), MAX_REPLAY_NODES / copies));
    for (int k = 1; k < rep; k++) {
        for (int c = 0; c < copies; c++) {
            ggml_graph_add_node(gf, outs[c]);
        }
    }

    res.ok     = true;
    res.us     = time_graph(backend, gf, o.reps) / ((double) copies * rep);
    res.copies = copies;
    return res;
}

//
// ceiling probes
//

struct ceilings {
    double bw_gbs     = 0.0; // read bandwidth, GB/s
    double tflops     = 0.0; // matmul throughput
    double launch_us  = 0.0; // one dependent dispatch
};

static double time_graph(ggml_backend_t backend, ggml_cgraph * gf, int reps) {
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    std::vector<double> t(reps);
    for (int r = 0; r < reps; r++) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        t[r] = (double) (ggml_time_us() - t0);
    }
    std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
    return t[t.size() / 2];
}

static ceilings probe(ggml_backend_t backend, const options & o) {
    ceilings c;

    // read bandwidth: f16 mat-vec over distinct matrices (read-dominant, like decode)
    {
        const int64_t n = 8192, copies = 8;
        op_desc d;
        d.op       = GGML_OP_MUL_MAT;
        d.src[0]   = { true, GGML_TYPE_F16, { n, n, 1, 1 }, { 2, (size_t) n * 2, (size_t) n * n * 2, (size_t) n * n * 2 }, true };
        d.src[1]   = { true, GGML_TYPE_F32, { n, 1, 1, 1 }, { 4, (size_t) n * 4, (size_t) n * 4, (size_t) n * 4 }, false };
        d.dst      = { true, GGML_TYPE_F32, { n, 1, 1, 1 }, { 4, (size_t) n * 4, (size_t) n * 4, (size_t) n * 4 }, false };
        options oo = o;
        oo.cache_bust = (double) copies * n * n * 2 / (1024.0 * 1024.0);
        replay_result r = replay(backend, d, oo);
        if (r.ok) {
            c.bw_gbs = bytes_of(d) / (r.us * 1e3);
        }
    }

    // matmul throughput: f16 x f32 at 512 columns
    {
        const int64_t n = 4096, m = 512;
        op_desc d;
        d.op       = GGML_OP_MUL_MAT;
        d.src[0]   = { true, GGML_TYPE_F16, { n, n, 1, 1 }, { 2, (size_t) n * 2, (size_t) n * n * 2, (size_t) n * n * 2 }, true };
        d.src[1]   = { true, GGML_TYPE_F32, { n, m, 1, 1 }, { 4, (size_t) n * 4, (size_t) n * m * 4, (size_t) n * m * 4 }, false };
        d.dst      = { true, GGML_TYPE_F32, { n, m, 1, 1 }, { 4, (size_t) n * 4, (size_t) n * m * 4, (size_t) n * m * 4 }, false };
        options oo = o;
        oo.cache_bust = 0.0; // 4 copies
        replay_result r = replay(backend, d, oo);
        if (r.ok) {
            c.tflops = flops_of(d) / (r.us * 1e6);
        }
    }

    // dispatch cost: a dependent chain of tiny ops
    {
        const int n = 2048;
        ggml_init_params ip = { ggml_tensor_overhead() * (n + 8) + ggml_graph_overhead_custom(n + 8, false), nullptr, true };
        ggml_context_ptr ctx(ggml_init(ip));
        ggml_tensor * x0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
        ggml_tensor * x  = x0;
        for (int i = 0; i < n; i++) {
            x = ggml_scale(ctx.get(), x, 1.0001f);
        }
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        std::vector<float> zero(32, 0.0f);
        ggml_backend_tensor_set(x0, zero.data(), 0, zero.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), n + 8, false);
        ggml_build_forward_expand(gf, x);
        // the chain is long enough (n dependent dispatches) to amortize the host round trip
        c.launch_us = time_graph(backend, gf, o.reps) / n;
    }

    return c;
}

//
// sequence replay
//

// the heavy ops of the step in their harvested order on the replay backend: weights are the model's own
// tensors, the other sources fresh allocations. each kernel starts after the kernel that precedes it in
// the real step, so its time includes what a kernel pays for following different work, which a replay
// of one kernel back to back hides (it misranked kernel variants by up to 8 us per call on M1)
struct sequence {
    ggml_context_ptr                   ctx;
    ggml_backend_buffer_ptr            buf;
    std::vector<ggml_tensor *>         nodes;
    std::vector<int>                   sig;     // per node: index into keys
    std::vector<std::string>           keys;    // signatures in the sequence
    std::map<std::string, std::string> skipped; // signature -> why it is not in the sequence
};

static bool build_sequence(ggml_backend_t backend, ggml_backend_dev_t dev, const harvest & h, sequence & sq) {
    // signatures that cannot be rebuilt are left out as a whole
    std::vector<const seq_entry *> sel;
    for (const auto & e : h.seq) {
        const op_desc & d = h.sigs.at(e.key);
        if (!is_heavy(d) || d.buft.rfind("CPU", 0) == 0) {
            continue;
        }
        std::string why = replay_unsupported(d);
        for (int i = 0; i < GGML_MAX_SRC && why.empty(); i++) {
            const ggml_tensor * t = e.real[i];
            if (t && (!t->buffer || !ggml_backend_dev_supports_buft(dev, ggml_backend_buffer_get_type(t->buffer)))) {
                why = "weights not reachable from the replay device";
            }
        }
        if (!why.empty()) {
            sq.skipped[e.key] = why;
        }
        sel.push_back(&e);
    }

    ggml_init_params ip = { ggml_tensor_overhead() * (sel.size() * (2 * GGML_MAX_SRC + 2) + 16), nullptr, true };
    sq.ctx.reset(ggml_init(ip));

    std::map<std::string, int>                         index;
    std::vector<std::vector<src_alloc>>                allocs;
    std::vector<const seq_entry *>                     used;
    for (const seq_entry * e : sel) {
        if (sq.skipped.count(e->key)) {
            continue;
        }
        const op_desc & d = h.sigs.at(e->key);
        std::vector<src_alloc> a;
        ggml_tensor * out = make_instance(sq.ctx.get(), d, e->real, a);
        if (!index.count(e->key)) {
            if (!ggml_backend_supports_op(backend, out)) {
                sq.skipped[e->key] = "not supported by the replay backend";
                continue;
            }
            index[e->key] = (int) sq.keys.size();
            sq.keys.push_back(e->key);
        }
        out->flags |= GGML_TENSOR_FLAG_COMPUTE; // nodes are added directly, not through ggml_build_forward_expand
        sq.nodes.push_back(out);
        sq.sig.push_back(index[e->key]);
        allocs.push_back(a);
        used.push_back(e);
    }
    if (sq.nodes.empty()) {
        return false;
    }

    // all nodes write into one region: a backend must order writes to the same memory, so each node
    // waits for the one before it, as in the dependent chain of a real step (independent nodes could
    // otherwise overlap where the step cannot)
    ggml_tensor * shared = nullptr;
    {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        size_t size = 0;
        for (ggml_tensor * t : sq.nodes) {
            size = std::max(size, ggml_backend_buft_get_alloc_size(buft, t));
        }
        shared = ggml_new_tensor_1d(sq.ctx.get(), GGML_TYPE_I8, (int64_t) size);
    }

    sq.buf.reset(ggml_backend_alloc_ctx_tensors(sq.ctx.get(), backend));
    if (!sq.buf) {
        return false;
    }
    for (ggml_tensor * t : sq.nodes) {
        t->view_src  = shared;
        t->view_offs = 0;
        t->data      = shared->data;
        t->buffer    = shared->buffer;
    }

    // one host image per signature and source, uploaded to every instance
    std::map<std::string, std::vector<uint8_t>> images;
    for (size_t n = 0; n < sq.nodes.size(); n++) {
        const op_desc & d = h.sigs.at(used[n]->key);
        for (size_t k = 0, i = 0; i < GGML_MAX_SRC; i++) {
            if (!d.src[i].present) {
                continue;
            }
            const src_alloc & a = allocs[n][k++];
            if (!a.base) {
                continue;
            }
            const std::string ik = used[n]->key + "#" + std::to_string(i);
            if (!images.count(ik)) {
                images[ik] = make_src_image(d, (int) i, a);
            }
            const std::vector<uint8_t> & data = images[ik];
            ggml_backend_tensor_set(a.base, data.data(), 0, std::min(data.size(), ggml_nbytes(a.base)));
        }
    }
    return true;
}

struct sequence_timing {
    double              full_us  = 0.0; // the whole sequence, median
    double              noise_us = 0.0; // median absolute deviation of the full runs
    std::vector<double> delta_us;       // per signature: full minus the sequence without it, median
    std::vector<int>    n;              // per signature: nodes in the sequence
};

static double run_graph_once(ggml_backend_t backend, ggml_cgraph * gf) {
    const int64_t t0 = ggml_time_us();
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    return (double) (ggml_time_us() - t0);
}

static double median(std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

// leave one signature out at a time; full and reduced runs alternate and their paired differences are
// taken, so slow drift (clocks, temperature) cancels
static sequence_timing time_sequence(ggml_backend_t backend, const sequence & sq, int reps) {
    const size_t n_sig  = sq.keys.size();
    const size_t n_node = sq.nodes.size();

    ggml_init_params ip = { (n_sig + 1) * ggml_graph_overhead_custom(n_node + 16, false) + 1024, nullptr, true };
    ggml_context_ptr gctx(ggml_init(ip));

    // graph j leaves signature j out; graph n_sig is the whole sequence
    std::vector<ggml_cgraph *> g(n_sig + 1);
    for (size_t j = 0; j <= n_sig; j++) {
        g[j] = ggml_new_graph_custom(gctx.get(), n_node + 16, false);
        for (size_t k = 0; k < n_node; k++) {
            if ((size_t) sq.sig[k] != j) {
                ggml_graph_add_node(g[j], sq.nodes[k]);
            }
        }
    }

    sequence_timing st;
    st.delta_us.resize(n_sig);
    st.n.assign(n_sig, 0);
    for (size_t k = 0; k < n_node; k++) {
        st.n[sq.sig[k]]++;
    }

    run_graph_once(backend, g[n_sig]); // warm up: pipelines, residency
    run_graph_once(backend, g[n_sig]);

    std::vector<double> full;
    for (size_t j = 0; j < n_sig; j++) {
        fprintf(stderr, "sequence: leaving out %zu/%zu ...\r", j + 1, n_sig);
        std::vector<double> diff;
        for (int r = 0; r < reps; r++) {
            const double a = run_graph_once(backend, g[n_sig]);
            const double b = run_graph_once(backend, g[j]);
            full.push_back(a);
            diff.push_back(a - b);
        }
        st.delta_us[j] = median(diff);
    }
    fprintf(stderr, "\n");

    st.full_us = median(full);
    std::vector<double> dev;
    for (double t : full) {
        dev.push_back(std::fabs(t - st.full_us));
    }
    st.noise_us = median(dev);
    return st;
}

//
// running the model
//

static void log_quiet(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static bool decode_tokens(llama_context * ctx, int pos, int n, bool all_logits, uint32_t seed) {
    llama_batch b = llama_batch_init(n, 0, 1);
    std::mt19937 g(seed);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    for (int i = 0; i < n; i++) {
        b.token[i]     = (llama_token) (100 + g() % (uint32_t) std::max(1, n_vocab - 200));
        b.pos[i]       = pos + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = all_logits || i == n - 1;
    }
    b.n_tokens = n;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static llama_context * make_context(llama_model * model, const options & o, harvest * h) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) (o.depth + (o.n_steps + 2) * o.n_tokens + 64);
    cp.n_batch         = (uint32_t) std::max(512, o.n_tokens);
    cp.n_ubatch        = (uint32_t) std::max(512, o.n_tokens);
    cp.flash_attn_type = o.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf         = true;
    if (h) {
        cp.cb_eval           = cb_harvest;
        cp.cb_eval_user_data = h;
    }
    return llama_init_from_model(model, cp);
}

static bool prefill(llama_context * ctx, int depth) {
    for (int p = 0; p < depth; p += 512) {
        if (!decode_tokens(ctx, p, std::min(512, depth - p), false, 1234 + p)) {
            return false;
        }
    }
    return true;
}

//
// main
//

static int report(const options & o, const harvest & h, double step_us, const char * model_desc);

int main(int argc, char ** argv) {
    options o;
    if (!parse_args(argc, argv, o)) {
        return 1;
    }

    llama_log_set(log_quiet, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers       = o.n_gpu_layers;
    llama_model * model   = llama_model_load_from_file(o.model.c_str(), mp);
    if (!model) {
        fprintf(stderr, "error: failed to load %s\n", o.model.c_str());
        return 1;
    }

    const bool all_logits = o.n_tokens > 1; // verify steps need logits for every token

    // 1. harvest one step at depth D
    harvest h;
    {
        llama_context * ctx = make_context(model, o, &h);
        if (!ctx || !prefill(ctx, o.depth) || !decode_tokens(ctx, o.depth, o.n_tokens, all_logits, 7)) {
            fprintf(stderr, "error: harvest run failed\n");
            return 1;
        }
        h.recording = true;
        const bool ok = decode_tokens(ctx, o.depth + o.n_tokens, o.n_tokens, all_logits, 8);
        llama_synchronize(ctx);
        h.recording = false;
        llama_free(ctx);
        if (!ok) {
            fprintf(stderr, "error: harvest step failed\n");
            return 1;
        }
    }

    // 2. end-to-end step time, in a context without the callback
    double step_us = 0.0;
    {
        llama_context * ctx = make_context(model, o, nullptr);
        if (!ctx || !prefill(ctx, o.depth) || !decode_tokens(ctx, o.depth, o.n_tokens, all_logits, 7)) {
            fprintf(stderr, "error: timing run failed\n");
            return 1;
        }
        llama_synchronize(ctx);
        std::vector<double> t;
        for (int s = 0; s < o.n_steps; s++) {
            const int64_t t0 = ggml_time_us();
            decode_tokens(ctx, o.depth + (s + 1) * o.n_tokens, o.n_tokens, all_logits, 100 + s);
            llama_synchronize(ctx);
            t.push_back((double) (ggml_time_us() - t0));
        }
        std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
        step_us = t[t.size() / 2];
        llama_free(ctx);
    }

    char model_desc[256];
    llama_model_desc(model, model_desc, sizeof(model_desc));

    // the model stays loaded: the sequence replay reads its weight tensors
    const int rc = report(o, h, step_us, model_desc);

    llama_model_free(model);
    llama_backend_free();
    return rc;
}

static int report(const options & o, const harvest & h, double step_us, const char * model_desc) {
    // 3. replay backend: independent copies must not overlap, so ask backends that overlap
    //    independent nodes (Metal) to serialize them; the variable is read when the backend starts
    if (o.serial) {
        setenv("GGML_METAL_CONCURRENCY_DISABLE", "1", 1);
    }
    ggml_backend_dev_t dev = pick_device(o.device);
    if (!dev) {
        fprintf(stderr, "error: replay device not found\n");
        return 1;
    }
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) {
        fprintf(stderr, "error: failed to start the replay backend\n");
        return 1;
    }

    fprintf(stderr, "probing device ceilings ...\n");
    const ceilings cl = probe(backend.get(), o);

    // 5. the heavy ops in their real order, each signature timed by leaving it out
    fprintf(stderr, "building the sequence of heavy ops ...\n");
    sequence        sq;
    sequence_timing st;
    const bool      have_seq = build_sequence(backend.get(), dev, h, sq);
    if (have_seq) {
        st = time_sequence(backend.get(), sq, o.reps);
    } else {
        fprintf(stderr, "warning: the sequence replay failed, times are from ops replayed alone\n");
    }
    std::map<std::string, int> seq_index;
    for (size_t j = 0; j < sq.keys.size() && have_seq; j++) {
        seq_index[sq.keys[j]] = (int) j;
    }

    // 6. each heavy signature alone, back to back: a throughput figure, for comparison
    struct row {
        std::string   key;
        op_desc       d;
        int           count;
        replay_result r;        // alone
        double        bytes, flops;
        bool          in_seq = false;
        double        us     = 0.0; // per op: in the sequence when available, else alone
        bool          noisy  = false;
    };
    std::vector<row> rows;
    std::map<std::string, int> other_count; // op name -> dispatches per step (not replayed)
    int n_other = 0;

    for (const auto & k : h.order) {
        const op_desc & d = h.sigs.at(k);
        const int       n = h.count.at(k);
        const bool on_cpu = d.buft.rfind("CPU", 0) == 0;
        if (!is_heavy(d) || on_cpu) {
            other_count[std::string(ggml_op_name(d.op)) + (on_cpu ? " (CPU)" : "")] += n;
            n_other += n;
            continue;
        }
        row r { k, d, n, {}, bytes_of(d), flops_of(d) };
        fprintf(stderr, "replaying alone %-60s x%d ...\n", label_of(d).c_str(), n);
        r.r  = replay(backend.get(), d, o);
        r.us = r.r.us;
        auto it = seq_index.find(k);
        if (it != seq_index.end()) {
            const int j = it->second;
            r.in_seq    = true;
            r.us        = st.delta_us[j] / st.n[j];
            r.noisy     = std::fabs(st.delta_us[j]) < 2.0 * st.noise_us;
        }
        rows.push_back(r);
    }

    std::sort(rows.begin(), rows.end(), [](const row & a, const row & b) {
        return a.count * a.us > b.count * b.us;
    });

    // 7. report
    double heavy_us = 0.0; // sum over signatures: leave-one-out deltas, or alone times
    double sum_delta = 0.0;
    int    n_heavy  = 0;
    for (const auto & r : rows) {
        if (r.in_seq || r.r.ok) {
            heavy_us += r.count * r.us;
        }
        if (r.in_seq) {
            sum_delta += r.count * r.us;
        }
        n_heavy += r.count;
    }

    printf("# Roofline report\n\n");
    printf("- model: %s (%s)\n", model_desc, o.model.c_str());
    printf("- device: %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    printf("- step: %d token(s) at KV depth %d (end-to-end timing over depth %d..%d), flash attention %s\n",
           o.n_tokens, o.depth, o.depth, o.depth + (o.n_steps + 1) * o.n_tokens, o.flash_attn ? "on" : "off");
    printf("- ceilings (measured): read bandwidth %.0f GB/s, matmul %.1f TFLOPS, dependent dispatch %.1f us\n\n",
           cl.bw_gbs, cl.tflops, cl.launch_us);

    printf("Times are per call inside the real op sequence (leave-one-out), so they include what a kernel pays for\n");
    printf("following different work. \"Alone\" is the same op replayed back to back, a throughput figure only.\n");
    printf("~ marks a signature whose total is below twice the run-to-run noise (%.0f us).\n\n", st.noise_us);

    printf("| Op signature | Per step | us each | ms per step | Share of step | Alone us | GB/s | %% of bw | TFLOPS | Limit | Efficiency |\n");
    printf("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |\n");
    for (const auto & r : rows) {
        if (!r.in_seq && !r.r.ok) {
            printf("| %s | %d | not replayed: %s | | | | | | | | |\n", label_of(r.d).c_str(), r.count, r.r.why.c_str());
            continue;
        }
        const double us     = std::max(r.us, 1e-3);
        const double t_mem  = r.bytes / (cl.bw_gbs * 1e3);
        const double t_comp = r.flops / (cl.tflops * 1e6);
        const double t_disp = cl.launch_us;
        const char * limit  = t_mem >= t_comp && t_mem >= t_disp ? "memory" : (t_comp >= t_disp ? "compute" : "dispatch");
        const double floor  = std::max({ t_mem, t_comp, t_disp });
        char alone[32] = "-";
        if (r.r.ok) {
            snprintf(alone, sizeof(alone), "%.1f", r.r.us);
        }
        printf("| %s | %d | %s%.1f%s | %.2f | %.1f%% | %s | %.0f | %.0f%% | %.2f | %s | %.0f%% |\n",
               label_of(r.d).c_str(), r.count, r.noisy ? "~" : "", r.us, r.in_seq ? "" : " (alone)",
               r.count * r.us / 1e3, 100.0 * r.count * r.us / step_us, alone,
               r.bytes / (us * 1e3), 100.0 * t_mem / us, r.flops / (us * 1e6), limit, 100.0 * floor / us);
    }

    printf("\n## Reconciliation\n\n");
    printf("| | ms per step | Share |\n| --- | ---: | ---: |\n");
    const double seq_us = have_seq ? st.full_us : heavy_us;
    printf("| Measured end-to-end step | %.2f | 100%% |\n", step_us / 1e3);
    printf("| Heavy ops, %s (%d dispatches) | %.2f | %.1f%% |\n", have_seq ? "replayed in sequence" : "replayed alone",
           have_seq ? (int) sq.nodes.size() : n_heavy, seq_us / 1e3, 100.0 * seq_us / step_us);
    printf("| Residual: other ops, CPU work, overlap (%d graph nodes) | %.2f | %.1f%% |\n", n_other,
           (step_us - seq_us) / 1e3, 100.0 * (step_us - seq_us) / step_us);
    if (have_seq) {
        printf("\nAdditivity: the leave-one-out times add up to %.2f ms, %.1f%% of the sequence (%.2f ms). A large gap\n"
               "means leaving a signature out changes what its neighbours cost.\n",
               sum_delta / 1e3, 100.0 * sum_delta / st.full_us, st.full_us / 1e3);
        for (const auto & kv : sq.skipped) {
            printf("- not in the sequence: %s (%s)\n", label_of(h.sigs.at(kv.first)).c_str(), kv.second.c_str());
        }
    }

    printf("\nOther ops per step (graph nodes before backend fusion):\n\n");
    std::vector<std::pair<int, std::string>> oc;
    for (const auto & kv : other_count) {
        oc.push_back({ kv.second, kv.first });
    }
    std::sort(oc.rbegin(), oc.rend());
    for (const auto & p : oc) {
        printf("- %s: %d\n", p.second.c_str(), p.first);
    }

    if (!o.json_out.empty()) {
        FILE * f = fopen(o.json_out.c_str(), "w");
        if (!f) {
            fprintf(stderr, "error: cannot write %s\n", o.json_out.c_str());
            return 1;
        }
        fprintf(f, "{\n  \"schema\": \"llama-roofline/2\",\n");
        fprintf(f, "  \"model\": \"%s\",\n  \"device\": \"%s\",\n", model_desc, ggml_backend_dev_description(dev));
        fprintf(f, "  \"n_tokens\": %d,\n  \"depth\": %d,\n  \"flash_attn\": %s,\n", o.n_tokens, o.depth, o.flash_attn ? "true" : "false");
        fprintf(f, "  \"ceilings\": { \"bw_gbs\": %.1f, \"tflops\": %.3f, \"dispatch_us\": %.2f },\n", cl.bw_gbs, cl.tflops, cl.launch_us);
        fprintf(f, "  \"step_us\": %.1f,\n  \"heavy_us\": %.1f,\n  \"sequence\": %s,\n  \"sequence_us\": %.1f,\n  \"noise_us\": %.1f,\n  \"heavy\": [\n",
                step_us, heavy_us, have_seq ? "true" : "false", seq_us, st.noise_us);
        for (size_t i = 0; i < rows.size(); i++) {
            const auto & r = rows[i];
            fprintf(f, "    { \"op\": \"%s\", \"type\": \"%s\", \"label\": \"%s\", \"count\": %d, \"in_sequence\": %s, \"us\": %.2f, \"noisy\": %s, \"replayed_alone\": %s, \"us_alone\": %.2f, \"bytes\": %.0f, \"flops\": %.0f, \"copies\": %d, \"key\": \"%s\" }%s\n",
                    ggml_op_name(r.d.op), ggml_type_name(r.d.src[0].type), label_of(r.d).c_str(), r.count,
                    r.in_seq ? "true" : "false", r.us, r.noisy ? "true" : "false", r.r.ok ? "true" : "false", r.r.us,
                    r.bytes, r.flops, r.r.copies, r.key.c_str(), i + 1 < rows.size() ? "," : "");
        }
        fprintf(f, "  ],\n  \"other\": {");
        size_t i = 0;
        for (const auto & kv : other_count) {
            fprintf(f, "%s \"%s\": %d", i++ ? "," : "", kv.first.c_str(), kv.second);
        }
        fprintf(f, " }\n}\n");
        fclose(f);
    }

    return 0;
}
