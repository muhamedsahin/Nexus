/**
 * @file store.cpp
 * @brief FAZ 10.2: TCPStore — sunucu (istemci basina is parcacigi) ve istemci.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/store.hpp"

#include "net.hpp"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

namespace nexus {
namespace dist {
namespace {

enum : std::uint8_t { kSet = 1, kGet = 2, kAdd = 3, kCheck = 4 };
enum : std::uint8_t { kOk = 0, kTimeout = 1, kBad = 2 };

void put_frame(net::Handle h, std::uint8_t tag, const std::string& a, const std::string& b) {
  std::string buf;
  buf.push_back(static_cast<char>(tag));
  const auto alen = static_cast<std::uint32_t>(a.size());
  const auto blen = static_cast<std::uint64_t>(b.size());
  buf.append(reinterpret_cast<const char*>(&alen), 4);
  buf += a;
  buf.append(reinterpret_cast<const char*>(&blen), 8);
  buf += b;
  net::send_all(h, buf.data(), buf.size());
}

}  // namespace

struct TcpStore::Server {
  net::Handle listener = net::kInvalid;
  std::atomic<bool> stop{false};
  std::mutex mu;
  std::condition_variable cv;
  std::map<std::string, std::string> data;
  std::thread acceptor;
  std::vector<std::thread> clients;
  std::vector<net::Handle> sockets;
  std::chrono::milliseconds timeout{};

  void serve(net::Handle h) {
    try {
      for (;;) {
        std::uint8_t cmd = 0;
        std::uint32_t klen = 0;
        std::uint64_t vlen = 0;
        for (;;) {  // durma bayragini 200 ms'de bir yokla
          if (stop) {
            return;
          }
          try {
            net::recv_all(h, &cmd, 1, std::chrono::milliseconds(200));
            break;
          } catch (const DistTimeout&) {
          }
        }
        net::recv_all(h, &klen, 4, timeout);
        std::string key(klen, '\0');
        net::recv_all(h, key.data(), klen, timeout);
        net::recv_all(h, &vlen, 8, timeout);
        std::string value(static_cast<std::size_t>(vlen), '\0');
        net::recv_all(h, value.data(), value.size(), timeout);
        std::uint8_t status = kOk;
        std::string reply;
        {
          std::unique_lock<std::mutex> lock(mu);
          if (cmd == kSet) {
            data[key] = value;
            cv.notify_all();
          } else if (cmd == kGet) {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            while (!stop && data.find(key) == data.end()) {
              if (cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                break;
              }
            }
            auto it = data.find(key);
            if (it == data.end()) {
              status = kTimeout;
            } else {
              reply = it->second;
            }
          } else if (cmd == kAdd) {
            std::int64_t delta = 0;
            std::memcpy(&delta, value.data(), std::min<std::size_t>(8, value.size()));
            std::int64_t cur = 0;
            auto it = data.find(key);
            if (it != data.end() && it->second.size() == 8) {
              std::memcpy(&cur, it->second.data(), 8);
            }
            cur += delta;
            std::string enc(8, '\0');
            std::memcpy(enc.data(), &cur, 8);
            data[key] = enc;
            reply = enc;
            cv.notify_all();
          } else if (cmd == kCheck) {
            reply = data.count(key) != 0 ? "1" : "0";
          } else {
            status = kBad;
          }
        }
        put_frame(h, status, std::string(), reply);
      }
    } catch (const std::exception&) {
      // istemci kapandi
    }
  }
};

TcpStore::TcpStore(const std::string& host, int port, int world_size, bool is_server, std::chrono::milliseconds timeout)
    : port_(port), world_(world_size), timeout_(timeout) {
  if (world_size < 1) {
    throw DistError("Nexus TcpStore: dunya boyu >= 1");
  }
  net::startup();
  if (is_server) {
    server_ = std::make_unique<Server>();
    server_->timeout = timeout;
    server_->listener = net::listen_on(host == "localhost" ? "127.0.0.1" : "0.0.0.0", port_);
    Server* srv = server_.get();
    srv->acceptor = std::thread([srv] {
      while (!srv->stop) {
        net::Handle h = net::kInvalid;
        try {
          h = net::accept_from(srv->listener, std::chrono::milliseconds(200));
        } catch (const DistTimeout&) {
          continue;  // durma bayragini yokla
        } catch (const std::exception&) {
          return;
        }
        std::lock_guard<std::mutex> lock(srv->mu);
        srv->sockets.push_back(h);
        srv->clients.emplace_back([srv, h] { srv->serve(h); });
      }
    });
  }
  const std::string connect_host = host.empty() || host == "0.0.0.0" ? "127.0.0.1" : host;
  socket_ = net::connect_to(connect_host == "localhost" ? "127.0.0.1" : connect_host, port_, timeout);
  local_address_ = net::local_ip(socket_);
}

TcpStore::~TcpStore() {
  net::close(socket_);
  if (server_) {
    server_->stop = true;
    server_->cv.notify_all();
    if (server_->acceptor.joinable()) {
      server_->acceptor.join();
    }
    for (std::thread& t : server_->clients) {
      if (t.joinable()) {
        t.join();  // en gec 200 ms'de durma bayragini gorur
      }
    }
    for (net::Handle h : server_->sockets) {
      net::close(h);
    }
    net::close(server_->listener);
  }
}

std::string TcpStore::request(std::uint8_t cmd, const std::string& key, const std::string& value) {
  put_frame(socket_, cmd, key, value);
  std::uint8_t status = 0;
  std::uint32_t alen = 0;
  std::uint64_t blen = 0;
  const auto wait = timeout_ + std::chrono::seconds(5);  // sunucu get icin timeout_ kadar bekler
  net::recv_all(socket_, &status, 1, wait);
  net::recv_all(socket_, &alen, 4, wait);
  std::string skip(alen, '\0');
  net::recv_all(socket_, skip.data(), alen, wait);
  net::recv_all(socket_, &blen, 8, wait);
  std::string out(static_cast<std::size_t>(blen), '\0');
  net::recv_all(socket_, out.data(), out.size(), wait);
  if (status == kTimeout) {
    throw DistError("Nexus TcpStore: '" + key + "' anahtari zaman asimina kadar gelmedi (bir rank baslamadi ya da oldu)");
  }
  if (status != kOk) {
    throw DistError("Nexus TcpStore: sunucu istegi reddetti");
  }
  return out;
}

void TcpStore::set(const std::string& key, const std::string& value) { (void)request(kSet, key, value); }

std::string TcpStore::get(const std::string& key) { return request(kGet, key, std::string()); }

std::int64_t TcpStore::add(const std::string& key, std::int64_t delta) {
  std::string enc(8, '\0');
  std::memcpy(enc.data(), &delta, 8);
  const std::string r = request(kAdd, key, enc);
  std::int64_t v = 0;
  std::memcpy(&v, r.data(), 8);
  return v;
}

bool TcpStore::check(const std::string& key) { return request(kCheck, key, std::string()) == "1"; }

void TcpStore::wait(const std::vector<std::string>& keys) {
  for (const std::string& k : keys) {
    (void)get(k);
  }
}

void TcpStore::barrier(const std::string& name) {
  const std::int64_t arrived = add("barrier/" + name, 1);
  if (arrived == world_) {
    set("barrier/" + name + "/done", "1");
  }
  (void)get("barrier/" + name + "/done");
}

}  // namespace dist
}  // namespace nexus
