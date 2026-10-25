/**
 * @file process_group.cpp
 * @brief FAZ 10.A/C/F/G: ProcessGroup — is parcacigi kuyrugu, indirgeme, halka/agac algoritmalari,
 *        dogrulama, hiyerarsik all-reduce; thread ve TCP tasimalari; ortamdan baslatma.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/process_group.hpp"

#include "net.hpp"
#include "nexus/core/half.hpp"
#include "nexus/dist/store.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <tuple>

namespace nexus {
namespace dist {

const char* to_string(ReduceOp op) noexcept {
  switch (op) {
    case ReduceOp::Sum: return "sum";
    case ReduceOp::Avg: return "avg";
    case ReduceOp::Max: return "max";
    case ReduceOp::Min: return "min";
    case ReduceOp::Product: return "product";
  }
  return "?";
}

#if defined(_MSC_VER)
#pragma warning(disable : 4996)  // getenv: ortam yalniz baslangicta okunur
#endif
int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' ? std::atoi(v) : fallback;
}

namespace {

constexpr std::uint32_t kCollective = 0;
constexpr std::uint32_t kP2P = 256;
std::uint32_t channel(std::uint32_t group, std::uint32_t kind, int tag = 0) {
  return group * 512u + kind + (static_cast<std::uint32_t>(tag) & 255u);
}

bool is_float(DType t) { return t == DType::F32 || t == DType::F64 || t == DType::F16 || t == DType::BF16; }

void check_dtype(DType t) {
  switch (t) {
    case DType::F32: case DType::F64: case DType::F16: case DType::BF16: case DType::I32: case DType::I64:
      return;
    default:
      throw DistError("Nexus dist: desteklenmeyen dtype (F32, F64, F16, BF16, I32, I64)");
  }
}

template <typename T>
void reduce_typed(T* dst, const T* src, std::size_t n, ReduceOp op) {
  switch (op) {
    case ReduceOp::Sum: case ReduceOp::Avg:
      for (std::size_t i = 0; i < n; ++i) dst[i] = dst[i] + src[i];
      break;
    case ReduceOp::Max:
      for (std::size_t i = 0; i < n; ++i) dst[i] = std::max(dst[i], src[i]);
      break;
    case ReduceOp::Min:
      for (std::size_t i = 0; i < n; ++i) dst[i] = std::min(dst[i], src[i]);
      break;
    case ReduceOp::Product:
      for (std::size_t i = 0; i < n; ++i) dst[i] = dst[i] * src[i];
      break;
  }
}

template <float (*Load)(std::uint16_t), std::uint16_t (*Store)(float)>
void reduce_half(std::uint16_t* dst, const std::uint16_t* src, std::size_t n, ReduceOp op) {
  for (std::size_t i = 0; i < n; ++i) {
    const float a = Load(dst[i]);
    const float b = Load(src[i]);
    float r = 0.0f;
    switch (op) {
      case ReduceOp::Sum: case ReduceOp::Avg: r = a + b; break;
      case ReduceOp::Max: r = std::max(a, b); break;
      case ReduceOp::Min: r = std::min(a, b); break;
      case ReduceOp::Product: r = a * b; break;
    }
    dst[i] = Store(r);
  }
}

float bf_load(std::uint16_t b) { return bfloat16_bits_to_float(b); }
std::uint16_t bf_store(float f) { return float_to_bfloat16_bits(f); }
float h_load(std::uint16_t b) { return half_bits_to_float(b); }
std::uint16_t h_store(float f) { return float_to_half_bits(f); }

void reduce(void* dst, const void* src, std::size_t n, DType t, ReduceOp op) {
  switch (t) {
    case DType::F32: reduce_typed(static_cast<float*>(dst), static_cast<const float*>(src), n, op); break;
    case DType::F64: reduce_typed(static_cast<double*>(dst), static_cast<const double*>(src), n, op); break;
    case DType::I32: reduce_typed(static_cast<std::int32_t*>(dst), static_cast<const std::int32_t*>(src), n, op); break;
    case DType::I64: reduce_typed(static_cast<std::int64_t*>(dst), static_cast<const std::int64_t*>(src), n, op); break;
    case DType::BF16:
      reduce_half<bf_load, bf_store>(static_cast<std::uint16_t*>(dst), static_cast<const std::uint16_t*>(src), n, op);
      break;
    case DType::F16:
      reduce_half<h_load, h_store>(static_cast<std::uint16_t*>(dst), static_cast<const std::uint16_t*>(src), n, op);
      break;
    default: throw DistError("Nexus dist: indirgeme dtype'i desteklenmiyor");
  }
}

void scale(void* data, std::size_t n, DType t, double factor) {
  switch (t) {
    case DType::F32: { auto* p = static_cast<float*>(data); for (std::size_t i = 0; i < n; ++i) p[i] = static_cast<float>(p[i] * factor); break; }
    case DType::F64: { auto* p = static_cast<double*>(data); for (std::size_t i = 0; i < n; ++i) p[i] *= factor; break; }
    case DType::BF16: { auto* p = static_cast<std::uint16_t*>(data); for (std::size_t i = 0; i < n; ++i) p[i] = bf_store(static_cast<float>(bf_load(p[i]) * factor)); break; }
    case DType::F16: { auto* p = static_cast<std::uint16_t*>(data); for (std::size_t i = 0; i < n; ++i) p[i] = h_store(static_cast<float>(h_load(p[i]) * factor)); break; }
    default: throw DistError("Nexus dist: Avg yalniz kayan nokta dtype'larinda");
  }
}

// ---------------------------------------------------------------------------- Work
class WorkImpl final : public Work {
 public:
  explicit WorkImpl(std::chrono::milliseconds timeout) : timeout_(timeout) {}
  bool is_completed() const override {
    std::lock_guard<std::mutex> lock(mu_);
    return done_;
  }
  void wait(std::chrono::milliseconds timeout) override {
    std::unique_lock<std::mutex> lock(mu_);
    const auto t = timeout.count() < 0 ? timeout_ + std::chrono::seconds(5) : timeout;
    if (!cv_.wait_for(lock, t, [this] { return done_; })) {
      throw DistTimeout("Nexus dist: Work::wait zaman asimi");
    }
    if (error_) {
      std::rethrow_exception(error_);
    }
  }
  std::chrono::steady_clock::time_point started() const override {
    std::lock_guard<std::mutex> lock(mu_);
    return started_;
  }
  std::chrono::steady_clock::time_point finished() const override {
    std::lock_guard<std::mutex> lock(mu_);
    return finished_;
  }
  void start() {
    std::lock_guard<std::mutex> lock(mu_);
    started_ = std::chrono::steady_clock::now();
  }
  void finish(std::exception_ptr error) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      finished_ = std::chrono::steady_clock::now();
      error_ = std::move(error);
      done_ = true;
    }
    cv_.notify_all();
  }

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  bool done_ = false;
  std::exception_ptr error_;
  std::chrono::milliseconds timeout_;
  std::chrono::steady_clock::time_point started_{}, finished_{};
};

// ---------------------------------------------------------------------------- thread tasima
class ThreadTransport final : public Transport {
 public:
  ThreadTransport(std::shared_ptr<ThreadWorld::State> state, int rank) : state_(std::move(state)), rank_(rank) {}
  int world_rank() const noexcept override { return rank_; }
  int world_size() const noexcept override;
  std::uint64_t isend(int peer, std::uint32_t group, const void* data, std::size_t bytes) override;
  void wait_send(int, std::uint64_t, std::chrono::milliseconds) override {}
  void recv(int peer, std::uint32_t group, void* data, std::size_t bytes, std::chrono::milliseconds timeout) override;
  const char* name() const noexcept override { return "thread"; }

 private:
  std::shared_ptr<ThreadWorld::State> state_;
  int rank_;
};

}  // namespace

struct ThreadWorld::State {
  int size = 1;
  std::mutex mu;
  std::condition_variable cv;
  std::map<std::tuple<int, int, std::uint32_t>, std::deque<std::vector<unsigned char>>> boxes;
};

namespace {

int ThreadTransport::world_size() const noexcept { return state_->size; }

std::uint64_t ThreadTransport::isend(int peer, std::uint32_t group, const void* data, std::size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  std::vector<unsigned char> msg(p, p + bytes);
  {
    std::lock_guard<std::mutex> lock(state_->mu);
    state_->boxes[{rank_, peer, group}].push_back(std::move(msg));
  }
  state_->cv.notify_all();
  return 0;
}

void ThreadTransport::recv(int peer, std::uint32_t group, void* data, std::size_t bytes, std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(state_->mu);
  auto& box = state_->boxes[{peer, rank_, group}];
  if (!state_->cv.wait_for(lock, timeout, [&] { return !box.empty(); })) {
    throw DistTimeout("Nexus dist (thread): rank " + std::to_string(peer) + "'den veri gelmedi (zaman asimi)");
  }
  std::vector<unsigned char> msg = std::move(box.front());
  box.pop_front();
  lock.unlock();
  if (msg.size() != bytes) {
    throw DistError("Nexus dist: rank " + std::to_string(peer) + " " + std::to_string(msg.size()) + " bayt gonderdi, " +
                    std::to_string(bytes) + " bekleniyordu");
  }
  if (bytes != 0) {
    std::memcpy(data, msg.data(), bytes);
  }
}

// ---------------------------------------------------------------------------- TCP tasima
struct FrameHeader {
  std::uint32_t group;
  std::uint32_t pad;
  std::uint64_t bytes;
};

class TcpTransport final : public Transport {
 public:
  TcpTransport(int rank, int size, std::vector<net::Handle> peers) : rank_(rank), size_(size), peers_(std::move(peers)) {
    lanes_.resize(static_cast<std::size_t>(size));
    for (int p = 0; p < size; ++p) {
      if (p == rank) continue;
      lanes_[static_cast<std::size_t>(p)] = std::make_unique<Lane>();
      Lane* lane = lanes_[static_cast<std::size_t>(p)].get();
      const net::Handle h = peers_[static_cast<std::size_t>(p)];
      lane->sender = std::thread([lane, h] { lane->run(h); });
    }
  }
  ~TcpTransport() override {
    for (auto& lane : lanes_) {
      if (!lane) continue;
      {
        std::lock_guard<std::mutex> lock(lane->mu);
        lane->stop = true;
      }
      lane->cv.notify_all();
      lane->sender.join();
    }
    for (net::Handle h : peers_) net::close(h);
  }
  int world_rank() const noexcept override { return rank_; }
  int world_size() const noexcept override { return size_; }
  const char* name() const noexcept override { return "tcp"; }

  std::uint64_t isend(int peer, std::uint32_t group, const void* data, std::size_t bytes) override {
    Lane& lane = *lanes_.at(static_cast<std::size_t>(peer));
    std::lock_guard<std::mutex> lock(lane.mu);
    if (lane.error) std::rethrow_exception(lane.error);
    const std::uint64_t ticket = ++lane.issued;
    lane.jobs.push_back(Job{FrameHeader{group, 0, bytes}, data, ticket});
    lane.cv.notify_all();
    return ticket;
  }

  void wait_send(int peer, std::uint64_t ticket, std::chrono::milliseconds timeout) override {
    Lane& lane = *lanes_.at(static_cast<std::size_t>(peer));
    std::unique_lock<std::mutex> lock(lane.mu);
    if (!lane.cv.wait_for(lock, timeout, [&] { return lane.done >= ticket || lane.error != nullptr; })) {
      throw DistTimeout("Nexus dist (tcp): rank " + std::to_string(peer) + "'e gonderim zaman asimi");
    }
    if (lane.error) std::rethrow_exception(lane.error);
  }

  void recv(int peer, std::uint32_t group, void* data, std::size_t bytes, std::chrono::milliseconds timeout) override {
    Lane& lane = *lanes_.at(static_cast<std::size_t>(peer));
    std::lock_guard<std::mutex> lock(lane.recv_mu);
    auto& side = lane.side[group];
    if (!side.empty()) {
      std::vector<unsigned char> msg = std::move(side.front());
      side.pop_front();
      check_size(peer, msg.size(), bytes);
      if (bytes != 0) std::memcpy(data, msg.data(), bytes);
      return;
    }
    const net::Handle h = peers_[static_cast<std::size_t>(peer)];
    for (;;) {
      FrameHeader hdr{};
      receive(peer, h, &hdr, sizeof(hdr), timeout);
      if (hdr.group == group) {
        check_size(peer, hdr.bytes, bytes);
        receive(peer, h, data, bytes, timeout);
        return;
      }
      std::vector<unsigned char> other(static_cast<std::size_t>(hdr.bytes));
      receive(peer, h, other.data(), other.size(), timeout);
      lane.side[hdr.group].push_back(std::move(other));
    }
  }

 private:
  struct Job {
    FrameHeader header;
    const void* data;
    std::uint64_t ticket;
  };
  struct Lane {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Job> jobs;
    std::uint64_t issued = 0;
    std::uint64_t done = 0;
    bool stop = false;
    std::exception_ptr error;
    std::thread sender;
    std::mutex recv_mu;
    std::map<std::uint32_t, std::deque<std::vector<unsigned char>>> side;
    void run(net::Handle h) {
      for (;;) {
        Job job{};
        {
          std::unique_lock<std::mutex> lock(mu);
          cv.wait(lock, [&] { return stop || !jobs.empty(); });
          if (jobs.empty()) return;
          job = jobs.front();
          jobs.pop_front();
        }
        try {
          net::send_all(h, &job.header, sizeof(job.header));
          if (job.header.bytes != 0) net::send_all(h, job.data, static_cast<std::size_t>(job.header.bytes));
          std::lock_guard<std::mutex> lock(mu);
          done = job.ticket;
        } catch (...) {
          std::lock_guard<std::mutex> lock(mu);
          error = std::current_exception();
        }
        cv.notify_all();
      }
    }
  };

  static void check_size(int peer, std::uint64_t got, std::size_t want) {
    if (got != want) {
      throw DistError("Nexus dist: rank " + std::to_string(peer) + " " + std::to_string(got) + " bayt gonderdi, " +
                      std::to_string(want) + " bekleniyordu (kollektifler ayni sirada mi?)");
    }
  }
  static void receive(int peer, net::Handle h, void* data, std::size_t bytes, std::chrono::milliseconds timeout) {
    try {
      net::recv_all(h, data, bytes, timeout);
    } catch (const DistTimeout& e) {
      throw DistTimeout(std::string(e.what()) + " [rank " + std::to_string(peer) + "]");
    } catch (const DistError& e) {
      throw DistError(std::string(e.what()) + " [rank " + std::to_string(peer) + "]");
    }
  }

  int rank_;
  int size_;
  std::vector<net::Handle> peers_;
  std::vector<std::unique_ptr<Lane>> lanes_;
};

struct Fingerprint {
  std::uint64_t seq;
  std::uint64_t count;
  std::int32_t root;
  std::uint8_t dtype;
  char op[19];
};
static_assert(sizeof(Fingerprint) == 40, "parmak izi 40 bayt");

}  // namespace

// ---------------------------------------------------------------------------- ProcessGroup
struct ProcessGroup::Impl {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::pair<std::function<void()>, std::shared_ptr<WorkImpl>>> tasks;
  bool stop = false;
  std::string broken;
  std::thread worker;
  std::map<int, std::pair<GroupPtr, GroupPtr>> hierarchy;  // ranks_per_node -> (dugum ici, dugumler arasi)
  std::shared_ptr<void> keep;                              // ornegin init_process_group'un store'u
};

std::shared_ptr<ProcessGroup> make_group(std::shared_ptr<Transport> transport, std::vector<int> members, int rank,
                                         std::string backend, GroupOptions options) {
  return std::shared_ptr<ProcessGroup>(
      new ProcessGroup(std::move(transport), std::move(members), rank, std::move(backend), options));
}

ProcessGroup::ProcessGroup(std::shared_ptr<Transport> transport, std::vector<int> members, int rank, std::string backend,
                           GroupOptions options)
    : transport_(std::move(transport)), members_(std::move(members)), rank_(rank),
      size_(static_cast<int>(members_.size())), backend_(std::move(backend)), options_(options),
      impl_(std::make_unique<Impl>()) {
  if (env_int("NEXUS_DIST_DEBUG", 0) != 0) {
    options_.validate = true;
  }
  Impl* impl = impl_.get();
  impl->worker = std::thread([impl] {
    for (;;) {
      std::pair<std::function<void()>, std::shared_ptr<WorkImpl>> task;
      {
        std::unique_lock<std::mutex> lock(impl->mu);
        impl->cv.wait(lock, [&] { return impl->stop || !impl->tasks.empty(); });
        if (impl->tasks.empty()) return;
        task = std::move(impl->tasks.front());
        impl->tasks.pop_front();
      }
      std::exception_ptr error;
      {
        std::lock_guard<std::mutex> lock(impl->mu);
        if (!impl->broken.empty()) {
          error = std::make_exception_ptr(DistError("Nexus dist: grup onceki hatadan sonra kullanilamaz: " + impl->broken));
        }
      }
      task.second->start();
      if (!error) {
        try {
          task.first();
        } catch (const std::exception& e) {
          error = std::current_exception();
          std::lock_guard<std::mutex> lock(impl->mu);
          impl->broken = e.what();
        }
      }
      task.second->finish(error);
    }
  });
}

ProcessGroup::~ProcessGroup() {
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->stop = true;
  }
  impl_->cv.notify_all();
  impl_->worker.join();
  impl_->hierarchy.clear();
}

WorkPtr ProcessGroup::enqueue(const char* op, DType dtype, std::size_t count, int root, std::function<void()> body,
                              bool collective) {
  auto work = std::make_shared<WorkImpl>(options_.timeout);
  const std::uint64_t seq = collective ? ++seq_ : seq_;
  std::function<void()> task;
  if (collective && options_.validate && size_ > 1) {
    Fingerprint mine{};
    mine.seq = seq;
    mine.count = count;
    mine.root = root;
    mine.dtype = static_cast<std::uint8_t>(dtype);
    std::memcpy(mine.op, op, std::min(std::strlen(op), sizeof(mine.op) - 1));
    task = [this, mine, body = std::move(body)] {
      // Parmak izleri halka all-gather ile (dogrulamasiz) degis tokus edilir.
      std::vector<Fingerprint> all(static_cast<std::size_t>(size_));
      all[static_cast<std::size_t>(rank_)] = mine;
      const std::uint32_t ch = channel(group_id_, kCollective);
      for (int s = 0; s < size_ - 1; ++s) {
        const int send_idx = (rank_ - s + size_) % size_;
        const int recv_idx = (rank_ - s - 1 + size_) % size_;
        const int right = members_[static_cast<std::size_t>((rank_ + 1) % size_)];
        const int left = members_[static_cast<std::size_t>((rank_ - 1 + size_) % size_)];
        const std::uint64_t t = transport_->isend(right, ch, &all[static_cast<std::size_t>(send_idx)], sizeof(Fingerprint));
        transport_->recv(left, ch, &all[static_cast<std::size_t>(recv_idx)], sizeof(Fingerprint), options_.timeout);
        transport_->wait_send(right, t, options_.timeout);
      }
      for (int r = 0; r < size_; ++r) {
        const Fingerprint& f = all[static_cast<std::size_t>(r)];
        if (f.seq != mine.seq || f.count != mine.count || f.root != mine.root || f.dtype != mine.dtype ||
            std::strncmp(f.op, mine.op, sizeof(mine.op)) != 0) {
          std::ostringstream m;
          m << "Nexus dist: kollektif uyusmazligi — rank " << rank_ << ": " << mine.op << " #" << mine.seq
            << " (dtype " << int(mine.dtype) << ", sayi " << mine.count << ", kok " << mine.root << ") / rank " << r
            << ": " << f.op << " #" << f.seq << " (dtype " << int(f.dtype) << ", sayi " << f.count << ", kok " << f.root
            << ")";
          throw DistError(m.str());
        }
      }
      body();
    };
  } else {
    task = std::move(body);
  }
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (!impl_->broken.empty()) {
      work->finish(std::make_exception_ptr(DistError("Nexus dist: grup bozuk: " + impl_->broken)));
      return work;
    }
    impl_->tasks.emplace_back(std::move(task), work);
  }
  impl_->cv.notify_all();
  return work;
}

namespace {
struct Chunks {
  std::size_t base, rem;
  std::size_t begin(int c) const { return static_cast<std::size_t>(c) * base + std::min<std::size_t>(static_cast<std::size_t>(c), rem); }
  std::size_t len(int c) const { return base + (static_cast<std::size_t>(c) < rem ? 1 : 0); }
};
}  // namespace

WorkPtr ProcessGroup::all_reduce(void* data, std::size_t count, DType dtype, ReduceOp op) {
  check_dtype(dtype);
  if (op == ReduceOp::Avg && !is_float(dtype)) {
    throw DistError("Nexus dist: Avg tamsayi dtype'ta tanimsiz (Sum kullanip bolun)");
  }
  return enqueue("all_reduce", dtype, count, static_cast<int>(op), [this, data, count, dtype, op] {
    const int n = size_;
    if (n > 1 && count > 0) {
      const std::size_t es = size_of(dtype);
      auto* bytes = static_cast<unsigned char*>(data);
      const Chunks ck{count / static_cast<std::size_t>(n), count % static_cast<std::size_t>(n)};
      std::vector<unsigned char> tmp((ck.base + 1) * es);
      const int right = members_[static_cast<std::size_t>((rank_ + 1) % n)];
      const int left = members_[static_cast<std::size_t>((rank_ - 1 + n) % n)];
      const std::uint32_t ch = channel(group_id_, kCollective);
      for (int s = 0; s < n - 1; ++s) {  // reduce-scatter
        const int si = (rank_ - s - 1 + 2 * n) % n;
        const int ri = (rank_ - s - 2 + 2 * n) % n;
        const std::uint64_t t = transport_->isend(right, ch, bytes + ck.begin(si) * es, ck.len(si) * es);
        transport_->recv(left, ch, tmp.data(), ck.len(ri) * es, options_.timeout);
        reduce(bytes + ck.begin(ri) * es, tmp.data(), ck.len(ri), dtype, op);
        transport_->wait_send(right, t, options_.timeout);
      }
      for (int s = 0; s < n - 1; ++s) {  // all-gather
        const int si = (rank_ - s + n) % n;
        const int ri = (rank_ - s - 1 + n) % n;
        const std::uint64_t t = transport_->isend(right, ch, bytes + ck.begin(si) * es, ck.len(si) * es);
        transport_->recv(left, ch, bytes + ck.begin(ri) * es, ck.len(ri) * es, options_.timeout);
        transport_->wait_send(right, t, options_.timeout);
      }
    }
    if (op == ReduceOp::Avg && n > 1) {
      scale(data, count, dtype, 1.0 / n);
    }
  });
}

WorkPtr ProcessGroup::reduce_scatter(const void* in, void* out, std::size_t count, DType dtype, ReduceOp op) {
  check_dtype(dtype);
  if (op == ReduceOp::Avg && !is_float(dtype)) {
    throw DistError("Nexus dist: Avg tamsayi dtype'ta tanimsiz");
  }
  return enqueue("reduce_scatter", dtype, count, static_cast<int>(op), [this, in, out, count, dtype, op] {
    const int n = size_;
    const std::size_t es = size_of(dtype);
    const std::size_t chunk = count * es;
    std::vector<unsigned char> work(static_cast<const unsigned char*>(in),
                                    static_cast<const unsigned char*>(in) + static_cast<std::size_t>(n) * chunk);
    std::vector<unsigned char> tmp(chunk);
    const int right = members_[static_cast<std::size_t>((rank_ + 1) % n)];
    const int left = members_[static_cast<std::size_t>((rank_ - 1 + n) % n)];
    const std::uint32_t ch = channel(group_id_, kCollective);
    for (int s = 0; s < n - 1 && chunk > 0; ++s) {
      const int si = (rank_ - s - 1 + 2 * n) % n;
      const int ri = (rank_ - s - 2 + 2 * n) % n;
      const std::uint64_t t = transport_->isend(right, ch, work.data() + static_cast<std::size_t>(si) * chunk, chunk);
      transport_->recv(left, ch, tmp.data(), chunk, options_.timeout);
      reduce(work.data() + static_cast<std::size_t>(ri) * chunk, tmp.data(), count, dtype, op);
      transport_->wait_send(right, t, options_.timeout);
    }
    if (chunk > 0) {
      std::memcpy(out, work.data() + static_cast<std::size_t>(rank_) * chunk, chunk);
    }
    if (op == ReduceOp::Avg && n > 1) {
      scale(out, count, dtype, 1.0 / n);
    }
  });
}

WorkPtr ProcessGroup::all_gather(const void* in, void* out, std::size_t count, DType dtype) {
  check_dtype(dtype);
  return enqueue("all_gather", dtype, count, -1, [this, in, out, count, dtype] {
    const int n = size_;
    const std::size_t chunk = count * size_of(dtype);
    auto* o = static_cast<unsigned char*>(out);
    if (chunk > 0 && o + static_cast<std::size_t>(rank_) * chunk != in) {
      std::memmove(o + static_cast<std::size_t>(rank_) * chunk, in, chunk);
    }
    const int right = members_[static_cast<std::size_t>((rank_ + 1) % n)];
    const int left = members_[static_cast<std::size_t>((rank_ - 1 + n) % n)];
    const std::uint32_t ch = channel(group_id_, kCollective);
    for (int s = 0; s < n - 1 && chunk > 0; ++s) {
      const int si = (rank_ - s + n) % n;
      const int ri = (rank_ - s - 1 + n) % n;
      const std::uint64_t t = transport_->isend(right, ch, o + static_cast<std::size_t>(si) * chunk, chunk);
      transport_->recv(left, ch, o + static_cast<std::size_t>(ri) * chunk, chunk, options_.timeout);
      transport_->wait_send(right, t, options_.timeout);
    }
  });
}

WorkPtr ProcessGroup::broadcast(void* data, std::size_t count, DType dtype, int root) {
  check_dtype(dtype);
  if (root < 0 || root >= size_) {
    throw DistError("Nexus dist: broadcast koku gecersiz");
  }
  return enqueue("broadcast", dtype, count, root, [this, data, count, dtype, root] {
    const int n = size_;
    const std::size_t bytes = count * size_of(dtype);
    const int vr = (rank_ - root + n) % n;
    const std::uint32_t ch = channel(group_id_, kCollective);
    auto member = [&](int v) { return members_[static_cast<std::size_t>((v + root) % n)]; };
    int mask = 1;
    while (mask < n) {  // binom agaci: once ebeveynden al
      if ((vr & mask) != 0) {
        transport_->recv(member(vr - mask), ch, data, bytes, options_.timeout);
        break;
      }
      mask <<= 1;
    }
    mask >>= 1;
    std::vector<std::pair<int, std::uint64_t>> sends;
    while (mask > 0) {
      if (vr + mask < n) {
        const int peer = member(vr + mask);
        sends.emplace_back(peer, transport_->isend(peer, ch, data, bytes));
      }
      mask >>= 1;
    }
    for (const auto& [peer, t] : sends) {
      transport_->wait_send(peer, t, options_.timeout);
    }
  });
}

WorkPtr ProcessGroup::all_to_all_v(const void* in, const std::vector<std::size_t>& send_counts, void* out,
                                   const std::vector<std::size_t>& recv_counts, DType dtype) {
  check_dtype(dtype);
  if (send_counts.size() != static_cast<std::size_t>(size_) || recv_counts.size() != static_cast<std::size_t>(size_)) {
    throw DistError("Nexus dist: all_to_all_v sayi listeleri grup boyunda olmali");
  }
  // Rank basina toplam farkli olabilir: parmak izinde sayi 0 (yalniz islem ve dtype denetlenir).
  return enqueue("all_to_all_v", dtype, 0, -1, [this, in, send_counts, out, recv_counts, dtype] {
    const int n = size_;
    const std::size_t es = size_of(dtype);
    std::vector<std::size_t> so(static_cast<std::size_t>(n) + 1, 0), ro(static_cast<std::size_t>(n) + 1, 0);
    for (int r = 0; r < n; ++r) {
      so[static_cast<std::size_t>(r) + 1] = so[static_cast<std::size_t>(r)] + send_counts[static_cast<std::size_t>(r)];
      ro[static_cast<std::size_t>(r) + 1] = ro[static_cast<std::size_t>(r)] + recv_counts[static_cast<std::size_t>(r)];
    }
    const auto* i = static_cast<const unsigned char*>(in);
    auto* o = static_cast<unsigned char*>(out);
    const std::size_t self = static_cast<std::size_t>(rank_);
    if (send_counts[self] != recv_counts[self]) {
      throw DistError("Nexus dist: all_to_all_v kendine gonderilen ve alinan sayi farkli");
    }
    if (send_counts[self] != 0) std::memcpy(o + ro[self] * es, i + so[self] * es, send_counts[self] * es);
    const std::uint32_t ch = channel(group_id_, kCollective);
    for (int k = 1; k < n; ++k) {
      const int dst = (rank_ + k) % n;
      const int src = (rank_ - k + n) % n;
      const auto d = static_cast<std::size_t>(dst);
      const auto s = static_cast<std::size_t>(src);
      const int wd = members_[d];
      const std::uint64_t t = transport_->isend(wd, ch, i + so[d] * es, send_counts[d] * es);
      transport_->recv(members_[s], ch, o + ro[s] * es, recv_counts[s] * es, options_.timeout);
      transport_->wait_send(wd, t, options_.timeout);
    }
  });
}

WorkPtr ProcessGroup::send(const void* data, std::size_t count, DType dtype, int dst, int tag) {
  if (dst < 0 || dst >= size_ || dst == rank_) throw DistError("Nexus dist: send hedefi gecersiz");
  return enqueue("send", dtype, count, dst, [this, data, count, dtype, dst, tag] {
    const int peer = members_[static_cast<std::size_t>(dst)];
    const std::uint64_t t = transport_->isend(peer, channel(group_id_, kP2P, tag), data, count * size_of(dtype));
    transport_->wait_send(peer, t, options_.timeout);
  }, false);
}

WorkPtr ProcessGroup::recv(void* data, std::size_t count, DType dtype, int src, int tag) {
  if (src < 0 || src >= size_ || src == rank_) throw DistError("Nexus dist: recv kaynagi gecersiz");
  // Bayt sayisi uyusmazligi tasimada yakalanir; send/recv iki rank'lik oldugu icin el sikismasi yok.
  return enqueue("recv", dtype, count, src, [this, data, count, dtype, src, tag] {
    transport_->recv(members_[static_cast<std::size_t>(src)], channel(group_id_, kP2P, tag), data,
                     count * size_of(dtype), options_.timeout);
  }, false);
}

WorkPtr ProcessGroup::barrier() {
  auto buf = std::make_shared<std::vector<std::int32_t>>(static_cast<std::size_t>(size_), 1);
  WorkPtr w = all_reduce(buf->data(), buf->size(), DType::I32, ReduceOp::Sum);
  // tampon Work bitene kadar yasamali: sonuna bos bir gorev ekleyip tamponu orada birak
  struct Holder final : Work {
    WorkPtr inner;
    std::shared_ptr<std::vector<std::int32_t>> keep;
    bool is_completed() const override { return inner->is_completed(); }
    void wait(std::chrono::milliseconds t) override { inner->wait(t); }
    std::chrono::steady_clock::time_point started() const override { return inner->started(); }
    std::chrono::steady_clock::time_point finished() const override { return inner->finished(); }
  };
  auto h = std::make_shared<Holder>();
  h->inner = std::move(w);
  h->keep = std::move(buf);
  return h;
}

void ProcessGroup::synchronize() { barrier()->wait(); }

GroupPtr ProcessGroup::new_group(const std::vector<int>& ranks) {
  const std::uint32_t id = group_id_ * 64u + next_child_++;
  if (ranks.empty()) throw DistError("Nexus dist: new_group bos liste");
  std::vector<int> world;
  int my = -1;
  for (std::size_t i = 0; i < ranks.size(); ++i) {
    const int r = ranks[i];
    if (r < 0 || r >= size_) throw DistError("Nexus dist: new_group rank'i gecersiz");
    world.push_back(members_[static_cast<std::size_t>(r)]);
    if (r == rank_) my = static_cast<int>(i);
  }
  if (my < 0) return nullptr;
  GroupPtr g = make_group(transport_, std::move(world), my, backend_, options_);
  g->group_id_ = id;
  return g;
}

WorkPtr ProcessGroup::all_reduce_hierarchical(void* data, std::size_t count, DType dtype, ReduceOp op, int ranks_per_node) {
  const int k = ranks_per_node;
  if (k < 1 || size_ % k != 0) {
    throw DistError("Nexus dist: ranks_per_node grup boyunu bolmeli");
  }
  if (k == 1 || k == size_ || count % static_cast<std::size_t>(k) != 0) {
    last_algorithm_ = k == 1 || k == size_ ? "ring (tek seviye)" : "ring (count % ranks_per_node != 0)";
    return all_reduce(data, count, dtype, op);
  }
  auto it = impl_->hierarchy.find(k);
  if (it == impl_->hierarchy.end()) {
    const int nodes = size_ / k;
    GroupPtr intra, inter;
    for (int node = 0; node < nodes; ++node) {  // her rank tum alt gruplari ayni sirada olusturur
      std::vector<int> r;
      for (int j = 0; j < k; ++j) r.push_back(node * k + j);
      GroupPtr g = new_group(r);
      if (g) intra = g;
    }
    for (int local = 0; local < k; ++local) {
      std::vector<int> r;
      for (int node = 0; node < nodes; ++node) r.push_back(node * k + local);
      GroupPtr g = new_group(r);
      if (g) inter = g;
    }
    it = impl_->hierarchy.emplace(k, std::make_pair(intra, inter)).first;
  }
  last_algorithm_ = "hierarchical (dugum ici RS -> dugumler arasi AR -> dugum ici AG)";
  ProcessGroup* intra = it->second.first.get();
  ProcessGroup* inter = it->second.second.get();
  return enqueue("all_reduce_hier", dtype, count, static_cast<int>(op), [this, intra, inter, data, count, dtype, op, k] {
    const std::size_t shard = count / static_cast<std::size_t>(k);
    std::vector<unsigned char> part(shard * size_of(dtype));
    const ReduceOp sub = op == ReduceOp::Avg ? ReduceOp::Sum : op;
    intra->reduce_scatter(data, part.data(), shard, dtype, sub)->wait();
    inter->all_reduce(part.data(), shard, dtype, sub)->wait();
    intra->all_gather(part.data(), data, shard, dtype)->wait();
    if (op == ReduceOp::Avg) scale(data, count, dtype, 1.0 / size_);
  });
}

// ---------------------------------------------------------------------------- arka uc fabrikalari
ThreadWorld::ThreadWorld(int size) : state_(std::make_shared<State>()) {
  if (size < 1) throw DistError("Nexus dist: ThreadWorld boyu >= 1");
  state_->size = size;
}
ThreadWorld::~ThreadWorld() = default;
int ThreadWorld::size() const noexcept { return state_->size; }

GroupPtr ThreadWorld::create_group(int rank, GroupOptions options) {
  if (rank < 0 || rank >= state_->size) throw DistError("Nexus dist: rank gecersiz");
  std::vector<int> members(static_cast<std::size_t>(state_->size));
  for (int i = 0; i < state_->size; ++i) members[static_cast<std::size_t>(i)] = i;
  return make_group(std::make_shared<ThreadTransport>(state_, rank), std::move(members), rank, "thread", options);
}

GroupPtr create_tcp_group(TcpStore& store, int rank, int size, GroupOptions options) {
  if (rank < 0 || rank >= size) throw DistError("Nexus dist: rank gecersiz");
  std::vector<net::Handle> peers(static_cast<std::size_t>(size), net::kInvalid);
  int port = 0;
  const net::Handle listener = net::listen_on("0.0.0.0", port);
  store.set("nexus/tcp/addr/" + std::to_string(rank), store.local_address() + ":" + std::to_string(port));
  try {
    for (int lower = 0; lower < rank; ++lower) {  // yuksek rank dusuge baglanir
      const std::string addr = store.get("nexus/tcp/addr/" + std::to_string(lower));
      const std::size_t colon = addr.rfind(':');
      const net::Handle h = net::connect_to(addr.substr(0, colon), std::stoi(addr.substr(colon + 1)), options.timeout);
      const auto me = static_cast<std::uint32_t>(rank);
      net::send_all(h, &me, 4);
      peers[static_cast<std::size_t>(lower)] = h;
    }
    for (int i = rank + 1; i < size; ++i) {
      const net::Handle h = net::accept_from(listener, options.timeout);
      std::uint32_t who = 0;
      net::recv_all(h, &who, 4, options.timeout);
      if (who <= static_cast<std::uint32_t>(rank) || who >= static_cast<std::uint32_t>(size) ||
          peers[who] != net::kInvalid) {
        throw DistError("Nexus dist: beklenmeyen baglanti (rank " + std::to_string(who) + ")");
      }
      peers[who] = h;
    }
  } catch (...) {
    net::close(listener);
    for (net::Handle h : peers) net::close(h);
    throw;
  }
  net::close(listener);
  store.barrier("nexus/tcp/ready");
  std::vector<int> members(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i) members[static_cast<std::size_t>(i)] = i;
  return make_group(std::make_shared<TcpTransport>(rank, size, std::move(peers)), std::move(members), rank, "tcp", options);
}

GroupPtr create_nccl_group(TcpStore&, int, int, GroupOptions) {
#if defined(_WIN32)
  throw DistError(
      "Nexus dist: NCCL arka ucu yok — NCCL Windows'ta resmi olarak desteklenmez. Tek dugumde 'tcp' arka ucunu "
      "(GPU tamponlari icin DDP host uzerinden aktarir) kullanin.");
#else
  throw DistError(
      "Nexus dist: NCCL arka ucu bu derlemede yok (NEXUS_ENABLE_NCCL desteklenmiyor; gelistirme makinesinde "
      "dogrulanamadi). 'tcp' arka ucunu kullanin.");
#endif
}

GroupPtr init_process_group(const std::string& backend, GroupOptions options) {
  const int rank = env_int("RANK", 0);
  const int world = env_int("WORLD_SIZE", 1);
  const char* addr = std::getenv("MASTER_ADDR");
  const int port = env_int("MASTER_PORT", 29500);
  auto store = std::make_shared<TcpStore>(addr != nullptr ? addr : "127.0.0.1", port, world, rank == 0, options.timeout);
  GroupPtr g;
  if (backend == "tcp" || backend == "gloo") {
    g = create_tcp_group(*store, rank, world, options);
  } else if (backend == "nccl") {
    g = create_nccl_group(*store, rank, world, options);
  } else {
    throw DistError("Nexus dist: bilinmeyen arka uc '" + backend + "' (tcp | nccl)");
  }
  g->impl_->keep = store;
  return g;
}

}  // namespace dist
}  // namespace nexus
