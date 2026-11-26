// Observe real CPU MUL_MAT operands via the public llama.cpp scheduler callback.
// Usage: packed-llama MODEL.gguf TOKEN_IDS.txt THREADS {q4_k_m|q8_0|q4_0|mixed|f32} {audit|bench|score}
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <sys/resource.h>

struct Observation {
    std::map<std::string, std::pair<std::string, size_t>> weights;
    std::set<std::string> expected;
    bool inconsistent = false;
};

static bool observe(ggml_tensor *op, bool ask, void *p) {
    auto &s = *static_cast<Observation *>(p);
    auto *w = op->src[0];
    if (op->op != GGML_OP_MUL_MAT || !w || !s.expected.count(w->name)) return !ask;
    if (!ask) {
        const auto info = std::make_pair(std::string(ggml_type_name(w->type)), ggml_nbytes(w));
        auto existing = s.weights.emplace(w->name, info);
        if (!existing.second && existing.first->second != info) s.inconsistent = true;
    }
    return true; // ask=true requests the post-compute observation.
}

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    const std::string mode(argv[5]), kind(argv[4]);
    if ((mode != "audit" && mode != "bench" && mode != "score") ||
        (kind != "q4_k_m" && kind != "q8_0" && kind != "q4_0" && kind != "mixed" && kind != "f32")) return 2;
    const int threads = std::stoi(argv[3]);
    std::ifstream in(argv[2]); std::vector<llama_token> prompt;
    for (llama_token t; in >> t;) prompt.push_back(t);
    if (!in.eof() || prompt.empty() || prompt.size() > 2048 || threads < 1) return 2;
    ggml_backend_load_all(); llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    auto *m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 3;
    auto *vocab = llama_model_get_vocab(m);
    int nv = llama_vocab_n_tokens(vocab);
    if (std::any_of(prompt.begin(), prompt.end(), [nv](int t){return t<0 || t>=nv;})) return 2;
    Observation trace;
    for (int l = 0; l < llama_model_n_layer(m); ++l)
        for (auto *n : {"attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"})
            trace.expected.insert("blk."+std::to_string(l)+"."+n+".weight");
    auto cp = llama_context_default_params();
    cp.n_ctx = prompt.size()+8; cp.n_batch = prompt.size(); cp.n_ubatch = prompt.size();
    cp.n_threads = threads; cp.n_threads_batch = threads; cp.offload_kqv = false;
    if (mode == "audit") {cp.cb_eval = observe; cp.cb_eval_user_data = &trace;}
    auto *ctx = llama_init_from_model(m,cp);
    if (!ctx) return 4;
    if (mode=="score") {
        if (prompt.size()<2) return 2;
        auto b = llama_batch_init(prompt.size(),0,1);
        for (size_t i=0;i<prompt.size();++i) {
            b.token[i]=prompt[i]; b.pos[i]=i; b.n_seq_id[i]=1; b.seq_id[i][0]=0; b.logits[i]=1;
        }
        b.n_tokens=prompt.size();
        if(llama_decode(ctx,b)) return 5;
        for(size_t i=0;i+1<prompt.size();++i) {
            const float *a=llama_get_logits_ith(ctx,i); if(!a) return 6;
            auto top=std::max_element(a,a+nv);
            double sum=0.;
            for(int j=0;j<nv;++j) sum+=std::exp(double(a[j])-*top);
            double nll=std::log(sum)+*top-a[prompt[i+1]];
            if(!std::isfinite(nll)) return 6;
            std::cout<<"SCORE "<<i<<" "<<prompt[i+1]<<" "<<(top-a)<<" "<<std::setprecision(12)<<nll<<"\n";
        }
        llama_batch_free(b); llama_free(ctx); llama_model_free(m); llama_backend_free();
        std::cout<<"LLAMA_TEACHER_FORCED_OK\n"; return 0;
    }
    std::vector<double> ms;
    int token = 0;
    for (int r = 0; r < (mode == "audit" ? 1 : 8); ++r) {
        llama_memory_clear(llama_get_memory(ctx), true);
        auto start = std::chrono::steady_clock::now();
        auto batch = llama_batch_get_one(prompt.data(), (int) prompt.size());
        // Eight greedy output tokens: one prefill and seven single-token forwards.
        for (int i=0;i<8;++i) {
            if (llama_decode(ctx,batch)) return 5;
            float *a = llama_get_logits_ith(ctx,-1);
            if (!a || !std::all_of(a,a+nv,[](float x){return std::isfinite(x);})) return 6;
            token = (int)(std::max_element(a,a+nv)-a);
            batch = llama_batch_get_one(&token,1);
        }
        auto end=std::chrono::steady_clock::now();
        if(r>=3) ms.push_back(std::chrono::duration<double,std::milli>(end-start).count());
    }
    bool ok = true;
    if (mode=="audit") {
        for(const auto &e:trace.weights) {
            auto &t=e.second.first;
            ok &= kind=="q4_k_m" ? (t=="q4_K" || t=="q6_K") :
                  kind=="mixed" ? (t=="q4_K" || t=="q6_K" || t=="q8_0") : t==kind;
            std::cout<<"WEIGHT\t"<<e.first<<"\t"<<t<<"\t"<<e.second.second<<"\n";
        }
        ok &= !trace.inconsistent && trace.weights.size()==trace.expected.size() && !trace.expected.empty();
    }
    rusage ru{}; getrusage(RUSAGE_SELF,&ru);
    std::cout<<std::setprecision(10)<<"SUMMARY "<<kind<<" "<<mode<<" "<<trace.weights.size()<<" "
             <<trace.expected.size()<<" "<<llama_model_size(m)<<" "<<ru.ru_maxrss<<" "<<token;
    for(double t:ms) std::cout<<" "<<t;
    std::cout<<"\n";
    llama_free(ctx); llama_model_free(m); llama_backend_free();
    if(!ok) {std::cerr<<"PACKED_OPERAND_CHECK_FAILED\n";return 7;}
    std::cout<<"LLAMA_PACKED_RUN_OK "<<mode<<" "<<kind<<"\n";
}
