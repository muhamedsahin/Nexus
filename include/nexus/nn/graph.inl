/**
 * @file graph.inl
 * @brief Adlandirilmis cok girisli graf ve nedensel blok.
 * Bu dosya yalnizca modules.hpp icinden, nexus::nn acikken eklenir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

namespace {

constexpr int kGraphSingle = 0;
constexpr int kGraphAdd = 1;
constexpr int kGraphConcat = 32;

nexus_model::Tensor zeros_like(const nexus_model::Tensor& source) {
    nexus_model::Tensor out;
    out.resize_like(source);
    out.zero();
    return out;
}

void add_into(nexus_model::Tensor& dst, const nexus_model::Tensor& src) {
    if (dst.numel() != src.numel()) {
        throw std::invalid_argument("graph gradient shapes do not match");
    }
    nexus_model::Tensor sum;
    sum.resize_like(dst);
    nexus_model::kernels::add(dst.data(), src.data(), sum.data(), dst.numel());
    dst = sum;
}

void copy_last_axis(const nexus_model::Tensor& src, nexus_model::Tensor& dst, std::size_t src_offset) {
    const std::uint8_t rank = src.rank();
    if (rank <= 1) {
        std::copy(src.data() + src_offset, src.data() + src_offset + dst.numel(), dst.data());
        return;
    }
    std::size_t outer = 1;
    for (std::uint8_t axis = 0; axis + 1 < rank; ++axis) {
        outer *= src.size(axis);
    }
    const std::size_t src_last = src.size(rank - 1);
    const std::size_t dst_last = dst.size(rank - 1);
    for (std::size_t row = 0; row < outer; ++row) {
        const float* from = src.data() + row * src_last + src_offset;
        float* to = dst.data() + row * dst_last;
        std::copy(from, from + dst_last, to);
    }
}

void write_last_axis(const nexus_model::Tensor& src, nexus_model::Tensor& dst, std::size_t dst_offset) {
    const std::uint8_t rank = src.rank();
    if (rank <= 1) {
        std::copy(src.data(), src.data() + src.numel(), dst.data() + dst_offset);
        return;
    }
    std::size_t outer = 1;
    for (std::uint8_t axis = 0; axis + 1 < rank; ++axis) {
        outer *= src.size(axis);
    }
    const std::size_t src_last = src.size(rank - 1);
    const std::size_t dst_last = dst.size(rank - 1);
    for (std::size_t row = 0; row < outer; ++row) {
        const float* from = src.data() + row * src_last;
        float* to = dst.data() + row * dst_last + dst_offset;
        std::copy(from, from + src_last, to);
    }
}

nexus_model::Tensor merge_inputs(const std::vector<nexus_model::Tensor>& parts, int merge) {
    if (parts.empty()) {
        throw std::invalid_argument("graph node has no input");
    }
    if (parts.size() == 1 || merge == kGraphSingle) {
        return parts.front();
    }
    if (merge == kGraphAdd) {
        for (std::size_t i = 1; i < parts.size(); ++i) {
            if (parts[i].numel() != parts.front().numel()) {
                throw std::invalid_argument("graph add requires equal element counts");
            }
        }
        nexus_model::Tensor sum = parts.front().clone();
        for (std::size_t i = 1; i < parts.size(); ++i) {
            add_into(sum, parts[i]);
        }
        return sum;
    }
    const std::uint8_t rank = parts.front().rank();
    if (rank == 0) {
        throw std::invalid_argument("graph concat needs a ranked tensor");
    }
    std::size_t last = 0;
    for (const nexus_model::Tensor& part : parts) {
        if (part.rank() != rank) {
            throw std::invalid_argument("graph concat ranks differ");
        }
        for (std::uint8_t axis = 0; axis + 1 < rank; ++axis) {
            if (part.size(axis) != parts.front().size(axis)) {
                throw std::invalid_argument("graph concat axes differ");
            }
        }
        last += part.size(rank - 1);
    }
    std::size_t dims[nexus_model::kMaxRank];
    for (std::uint8_t axis = 0; axis < rank; ++axis) {
        dims[axis] = parts.front().size(axis);
    }
    dims[rank - 1] = last;
    nexus_model::Tensor out;
    out.resize_dims(dims, rank);
    out.zero();
    std::size_t offset = 0;
    for (const nexus_model::Tensor& part : parts) {
        write_last_axis(part, out, offset);
        offset += part.size(rank - 1);
    }
    return out;
}

std::string join_names(const std::vector<std::string>& names) {
    std::string joined;
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (i != 0) {
            joined.push_back(',');
        }
        joined += names[i];
    }
    return joined;
}

}  // namespace

struct GraphNode {
    std::string name;
    std::shared_ptr<Module> layer;
    std::vector<std::string> inputs;
    int merge = kGraphSingle;
    std::vector<nexus_model::Tensor> saved;
};

class Graph : public Module {
 public:
    void add(std::string name, std::shared_ptr<Module> layer) {
        std::vector<std::string> inputs;
        if (!nodes_->empty()) {
            inputs.push_back(nodes_->back().name);
        }
        restore(std::move(name), std::move(layer), std::move(inputs), kGraphSingle);
    }

    void add(std::string name, std::shared_ptr<Module> layer, std::vector<std::string> inputs, bool concatenate) {
        if (inputs.size() < 2) {
            throw std::invalid_argument("graph merge needs at least two inputs");
        }
        restore(std::move(name), std::move(layer), std::move(inputs), concatenate ? kGraphConcat : kGraphAdd);
    }

    void concat(std::string name, std::shared_ptr<Module> layer, std::vector<std::string> inputs) {
        add(std::move(name), std::move(layer), std::move(inputs), true);
    }

    void restore(std::string name, std::shared_ptr<Module> layer, std::vector<std::string> inputs, int merge) {
        if (name.empty() || name.find(',') != std::string::npos) {
            throw std::invalid_argument("graph node name is invalid");
        }
        if (layer == nullptr) {
            throw std::invalid_argument("graph node has no layer");
        }
        for (const GraphNode& node : *nodes_) {
            if (node.name == name) {
                throw std::invalid_argument("graph node name is duplicated");
            }
        }
        if (inputs.empty()) {
            if (merge != kGraphSingle) {
                throw std::invalid_argument("graph entry node cannot merge");
            }
        } else if (inputs.size() == 1) {
            if (merge != kGraphSingle) {
                throw std::invalid_argument("graph merge needs at least two inputs");
            }
            ensure_known(inputs.front());
        } else {
            if (merge != kGraphAdd && merge != kGraphConcat) {
                throw std::invalid_argument("graph merge kind is unknown");
            }
            for (const std::string& input : inputs) {
                ensure_known(input);
            }
        }
        GraphNode node;
        node.name = std::move(name);
        node.layer = std::move(layer);
        node.inputs = std::move(inputs);
        node.merge = merge;
        nodes_->push_back(std::move(node));
    }

    nexus_model::Tensor forward(const nexus_model::Tensor& input) override {
        if (nodes_->empty()) {
            throw std::invalid_argument("graph has no nodes");
        }
        std::map<std::string, nexus_model::Tensor> values;
        for (GraphNode& node : *nodes_) {
            std::vector<nexus_model::Tensor> parts;
            if (node.inputs.empty()) {
                parts.push_back(input);
            } else {
                for (const std::string& input_name : node.inputs) {
                    const auto found = values.find(input_name);
                    if (found == values.end()) {
                        throw std::invalid_argument("graph input is unknown: " + input_name);
                    }
                    parts.push_back(found->second);
                }
            }
            node.saved.clear();
            for (const nexus_model::Tensor& part : parts) {
                node.saved.push_back(part.clone());
            }
            const nexus_model::Tensor merged = merge_inputs(parts, node.merge);
            const nexus_model::Tensor output = node.layer->forward(merged);
            values[node.name] = output.clone();
        }
        return values.at(nodes_->back().name);
    }

    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override {
        if (nodes_->empty()) {
            throw std::invalid_argument("graph has no nodes");
        }
        std::map<std::string, nexus_model::Tensor> grads;
        grads[nodes_->back().name] = grad_output.clone();
        bool have_input = false;
        nexus_model::Tensor input_grad;
        for (std::size_t index = nodes_->size(); index > 0; --index) {
            GraphNode& node = (*nodes_)[index - 1];
            const auto found = grads.find(node.name);
            if (found == grads.end()) {
                continue;
            }
            const nexus_model::Tensor local = node.layer->backward(found->second).clone();
            if (node.inputs.empty()) {
                if (!have_input) {
                    input_grad = local.clone();
                    have_input = true;
                } else {
                    add_into(input_grad, local);
                }
                continue;
            }
            if (node.merge == kGraphConcat) {
                std::size_t offset = 0;
                for (std::size_t part = 0; part < node.inputs.size(); ++part) {
                    nexus_model::Tensor piece = zeros_like(node.saved.at(part));
                    copy_last_axis(local, piece, offset);
                    const std::uint8_t rank = piece.rank();
                    offset += rank == 0 ? piece.numel() : piece.size(rank - 1);
                    send(grads, node.inputs[part], piece);
                }
            } else if (node.merge == kGraphAdd) {
                for (const std::string& input_name : node.inputs) {
                    send(grads, input_name, local);
                }
            } else {
                send(grads, node.inputs.front(), local);
            }
        }
        if (!have_input) {
            input_grad = zeros_like(grad_output);
        }
        return input_grad;
    }

    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override {
        std::vector<nexus_model::Parameter*> values;
        for (const GraphNode& node : *nodes_) {
            const std::vector<nexus_model::Parameter*> owned = node.layer->parameters();
            values.insert(values.end(), owned.begin(), owned.end());
        }
        return values;
    }

    void zero_grad() override {
        for (const GraphNode& node : *nodes_) {
            node.layer->zero_grad();
        }
    }

    void train() override {
        for (const GraphNode& node : *nodes_) {
            node.layer->train();
        }
    }

    void eval() override {
        for (const GraphNode& node : *nodes_) {
            node.layer->eval();
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return nodes_->size(); }

    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        nxs::LayerSpec spec = nxs::spec_of(nxs::LayerKind::Graph, "Graph");
        for (const GraphNode& node : *nodes_) {
            nxs::LayerSpec child = node.layer->nxs_spec();
            child.attrs.push_back(nxs::str("graph_name", node.name));
            child.attrs.push_back(nxs::str("graph_inputs", join_names(node.inputs)));
            child.attrs.push_back(nxs::i32("graph_merge", node.merge));
            spec.children.push_back(std::move(child));
        }
        return spec;
    }

    void save_nxs(const std::string& path) const {
        nxs::Archive archive;
        archive.layers.push_back(nxs_spec());
        archive.weights = nxs::collect_weights(parameters());
        nxs::save_archive(path, archive);
    }

    [[nodiscard]] static Graph load_nxs(const std::string& path) {
        const nxs::Archive archive = nxs::load_archive(path);
        if (archive.layers.size() != 1 || static_cast<nxs::LayerKind>(archive.layers.front().type) != nxs::LayerKind::Graph) {
            throw std::runtime_error("nxs root is not a graph");
        }
        const std::shared_ptr<Module> module = nxs::module_from_spec(archive.layers.front());
        Graph* graph = dynamic_cast<Graph*>(module.get());
        if (graph == nullptr) {
            throw std::runtime_error("nxs root is not a graph");
        }
        nxs::apply_weights(graph->parameters(), archive.weights);
        return *graph;
    }

 private:
    void ensure_known(const std::string& name) const {
        for (const GraphNode& node : *nodes_) {
            if (node.name == name) {
                return;
            }
        }
        throw std::invalid_argument("graph input is unknown: " + name);
    }

    static void send(std::map<std::string, nexus_model::Tensor>& grads, const std::string& name, const nexus_model::Tensor& grad) {
        const auto found = grads.find(name);
        if (found == grads.end()) {
            grads.emplace(name, grad.clone());
            return;
        }
        add_into(found->second, grad);
    }

    std::shared_ptr<std::vector<GraphNode>> nodes_ = std::make_shared<std::vector<GraphNode>>();
};

class CausalTransformerBlock : public Module {
 public:
    CausalTransformerBlock(int embed, int heads, int feedforward)
        : embed_(embed),
          heads_(heads),
          feedforward_(feedforward),
          attn_(std::make_shared<nexus_model::MultiHeadAttention>(embed, heads, 0.0f, true)),
          drop1_(std::make_shared<nexus_model::Dropout>(0.0f)),
          drop2_(std::make_shared<nexus_model::Dropout>(0.0f)),
          ln1_(std::make_shared<nexus_model::LayerNorm>(static_cast<std::size_t>(embed))),
          ln2_(std::make_shared<nexus_model::LayerNorm>(static_cast<std::size_t>(embed))),
          fc1_(std::make_shared<nexus_model::Linear>(static_cast<std::size_t>(embed), static_cast<std::size_t>(feedforward))),
          fc2_(std::make_shared<nexus_model::Linear>(static_cast<std::size_t>(feedforward), static_cast<std::size_t>(embed))),
          act_(std::make_shared<nexus_model::GELU>()) {}

    void train() override {
        attn_->train();
        drop1_->train();
        drop2_->train();
        ln1_->train();
        ln2_->train();
        fc1_->train();
        fc2_->train();
        act_->train();
    }

    void eval() override {
        attn_->eval();
        drop1_->eval();
        drop2_->eval();
        ln1_->eval();
        ln2_->eval();
        fc1_->eval();
        fc2_->eval();
        act_->eval();
    }

    nexus_model::Tensor forward(const nexus_model::Tensor& input) override {
        x0_.copy_from(input);
        const nexus_model::Tensor attn = attn_->forward(x0_);
        const nexus_model::Tensor dropped = drop1_->forward(attn);
        res1_.resize_like(x0_);
        nexus_model::kernels::add(x0_.data(), dropped.data(), res1_.data(), x0_.numel());
        const nexus_model::Tensor normed = ln1_->forward(res1_);
        const nexus_model::Tensor hidden = fc1_->forward(normed);
        const nexus_model::Tensor activated = act_->forward(hidden);
        const nexus_model::Tensor projected = fc2_->forward(activated);
        const nexus_model::Tensor dropped2 = drop2_->forward(projected);
        res2_.resize_like(normed);
        nexus_model::kernels::add(normed.data(), dropped2.data(), res2_.data(), normed.numel());
        return ln2_->forward(res2_);
    }

    nexus_model::Tensor backward(const nexus_model::Tensor& grad_output) override {
        const nexus_model::Tensor g_res2 = ln2_->backward(grad_output);
        const nexus_model::Tensor g_drop2 = drop2_->backward(g_res2);
        const nexus_model::Tensor g_fc2 = fc2_->backward(g_drop2);
        const nexus_model::Tensor g_act = act_->backward(g_fc2);
        const nexus_model::Tensor g_norm_ff = fc1_->backward(g_act);
        acc_.resize_like(g_res2);
        nexus_model::kernels::add(g_res2.data(), g_norm_ff.data(), acc_.data(), g_res2.numel());
        const nexus_model::Tensor g_res1 = ln1_->backward(acc_);
        const nexus_model::Tensor g_drop1 = drop1_->backward(g_res1);
        const nexus_model::Tensor g_attn = attn_->backward(g_drop1);
        grad_input_.resize_like(g_attn);
        nexus_model::kernels::add(g_attn.data(), g_res1.data(), grad_input_.data(), g_attn.numel());
        return grad_input_;
    }

    [[nodiscard]] std::vector<nexus_model::Parameter*> parameters() const override {
        std::vector<nexus_model::Parameter*> values;
        auto append = [&](const std::shared_ptr<nexus_model::Module>& child) {
            const std::vector<nexus_model::Parameter*> owned = child->parameters();
            values.insert(values.end(), owned.begin(), owned.end());
        };
        append(attn_);
        append(ln1_);
        append(fc1_);
        append(fc2_);
        append(ln2_);
        return values;
    }

    void zero_grad() override {
        attn_->zero_grad();
        ln1_->zero_grad();
        fc1_->zero_grad();
        fc2_->zero_grad();
        ln2_->zero_grad();
    }

    [[nodiscard]] nxs::LayerSpec nxs_spec() const override {
        return nxs::spec_of(nxs::LayerKind::CausalBlock, "CausalTransformerBlock",
                            {nxs::i32("embed", embed_), nxs::i32("heads", heads_), nxs::i32("feedforward", feedforward_)});
    }

 private:
    int embed_ = 0;
    int heads_ = 0;
    int feedforward_ = 0;
    std::shared_ptr<nexus_model::MultiHeadAttention> attn_;
    std::shared_ptr<nexus_model::Dropout> drop1_;
    std::shared_ptr<nexus_model::Dropout> drop2_;
    std::shared_ptr<nexus_model::LayerNorm> ln1_;
    std::shared_ptr<nexus_model::LayerNorm> ln2_;
    std::shared_ptr<nexus_model::Linear> fc1_;
    std::shared_ptr<nexus_model::Linear> fc2_;
    std::shared_ptr<nexus_model::GELU> act_;
    nexus_model::Tensor x0_;
    nexus_model::Tensor res1_;
    nexus_model::Tensor res2_;
    nexus_model::Tensor acc_;
    nexus_model::Tensor grad_input_;
};
