/**
 * @file net.cpp
 * @brief FAZ 10 ic: TCP soket yardimcilari (Winsock2 / POSIX).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "net.hpp"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
using socklen_type = int;
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
using socklen_type = socklen_t;
#endif

#include <cstring>
#include <mutex>
#include <thread>

namespace nexus {
namespace dist {
namespace net {
namespace {

#if defined(_WIN32)
using Native = SOCKET;
Native native(Handle h) { return static_cast<Native>(h); }
int last_error() { return WSAGetLastError(); }
#else
using Native = int;
Native native(Handle h) { return static_cast<Native>(h); }
int last_error() { return errno; }
#endif

[[noreturn]] void fail(const std::string& what) {
  throw DistError("Nexus dist: " + what + " (hata " + std::to_string(last_error()) + ")");
}

// readable: 1 hazir, 0 zaman asimi, -1 hata
int wait_readable(Handle h, std::chrono::milliseconds timeout) {
  const int ms = timeout.count() < 0 ? -1 : static_cast<int>(std::min<long long>(timeout.count(), 0x7fffffff));
#if defined(_WIN32)
  WSAPOLLFD p{};
  p.fd = native(h);
  p.events = POLLRDNORM;
  const int r = WSAPoll(&p, 1, ms);
#else
  pollfd p{};
  p.fd = native(h);
  p.events = POLLIN;
  const int r = ::poll(&p, 1, ms);
#endif
  return r > 0 ? 1 : (r == 0 ? 0 : -1);
}

sockaddr_in resolve(const std::string& host, int port) {
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<unsigned short>(port));
  if (host.empty() || host == "0.0.0.0") {
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    return addr;
  }
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) == 1) {
    return addr;
  }
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || res == nullptr) {
    throw DistError("Nexus dist: adres cozulemedi: " + host);
  }
  addr.sin_addr = reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr;
  freeaddrinfo(res);
  return addr;
}

}  // namespace

void startup() {
#if defined(_WIN32)
  static std::once_flag once;
  std::call_once(once, [] {
    WSADATA data;
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
      throw DistError("Nexus dist: WSAStartup basarisiz");
    }
  });
#endif
}

void close(Handle h) noexcept {
  if (h == kInvalid) {
    return;
  }
#if defined(_WIN32)
  ::closesocket(native(h));
#else
  ::close(native(h));
#endif
}

void tune(Handle h) {
  int one = 1;
  setsockopt(native(h), IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
  int buf = 4 << 20;
  setsockopt(native(h), SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&buf), sizeof(buf));
  setsockopt(native(h), SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&buf), sizeof(buf));
}

Handle listen_on(const std::string& host, int& port) {
  startup();
  const Native s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#if defined(_WIN32)
  if (s == INVALID_SOCKET) fail("socket");
#else
  if (s < 0) fail("socket");
  int reuse = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#endif
  sockaddr_in addr = resolve(host, port);
  if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(static_cast<Handle>(s));
    fail("bind " + host + ":" + std::to_string(port));
  }
  if (::listen(s, 128) != 0) {
    close(static_cast<Handle>(s));
    fail("listen");
  }
  socklen_type len = sizeof(addr);
  getsockname(s, reinterpret_cast<sockaddr*>(&addr), &len);
  port = ntohs(addr.sin_port);
  return static_cast<Handle>(s);
}

Handle connect_to(const std::string& host, int port, std::chrono::milliseconds timeout) {
  startup();
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  sockaddr_in addr = resolve(host, port);
  for (;;) {
    const Native s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#if defined(_WIN32)
    if (s == INVALID_SOCKET) fail("socket");
#else
    if (s < 0) fail("socket");
#endif
    if (::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      tune(static_cast<Handle>(s));
      return static_cast<Handle>(s);
    }
    close(static_cast<Handle>(s));
    if (std::chrono::steady_clock::now() >= deadline) {
      throw DistError("Nexus dist: " + host + ":" + std::to_string(port) + " adresine baglanilamadi (zaman asimi)");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

Handle accept_from(Handle listener, std::chrono::milliseconds timeout) {
  const int r = wait_readable(listener, timeout);
  if (r == 0) {
    throw DistTimeout("Nexus dist: accept zaman asimi (eksik rank?)");
  }
  if (r < 0) {
    fail("poll(accept)");
  }
  const Native s = ::accept(native(listener), nullptr, nullptr);
#if defined(_WIN32)
  if (s == INVALID_SOCKET) fail("accept");
#else
  if (s < 0) fail("accept");
#endif
  tune(static_cast<Handle>(s));
  return static_cast<Handle>(s);
}

void send_all(Handle h, const void* data, std::size_t bytes) {
  const char* p = static_cast<const char*>(data);
  while (bytes > 0) {
    const int chunk = static_cast<int>(std::min<std::size_t>(bytes, 1u << 30));
#if defined(_WIN32)
    const int n = ::send(native(h), p, chunk, 0);
#else
    const int n = static_cast<int>(::send(native(h), p, static_cast<std::size_t>(chunk), MSG_NOSIGNAL));
#endif
    if (n <= 0) {
      fail("baglanti koptu (send)");
    }
    p += n;
    bytes -= static_cast<std::size_t>(n);
  }
}

void recv_all(Handle h, void* data, std::size_t bytes, std::chrono::milliseconds timeout) {
  char* p = static_cast<char*>(data);
  while (bytes > 0) {
    const int r = wait_readable(h, timeout);
    if (r == 0) {
      throw DistTimeout("Nexus dist: karsi rank'ten veri bekleme zaman asimi");
    }
    if (r < 0) {
      fail("poll(recv)");
    }
    const int chunk = static_cast<int>(std::min<std::size_t>(bytes, 1u << 30));
    const int n = static_cast<int>(::recv(native(h), p, chunk, 0));
    if (n == 0) {
      throw DistError("Nexus dist: karsi rank baglantiyi kapatti (surec oldu mu?)");
    }
    if (n < 0) {
      fail("baglanti koptu (recv)");
    }
    p += n;
    bytes -= static_cast<std::size_t>(n);
  }
}

std::string local_ip(Handle connected) {
  sockaddr_in addr{};
  socklen_type len = sizeof(addr);
  if (getsockname(native(connected), reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
    return "127.0.0.1";
  }
  char buf[INET_ADDRSTRLEN] = {};
  inet_ntop(AF_INET, &addr.sin_addr, buf, sizeof(buf));
  return buf;
}

}  // namespace net
}  // namespace dist
}  // namespace nexus
