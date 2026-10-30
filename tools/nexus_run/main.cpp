/**
 * @file main.cpp
 * @brief FAZ 10.2 nexus-run: torchrun esdegeri baslatici (tek dugum basina bir cagri).
 *
 * Kullanim:
 *   nexus-run [--nproc-per-node N] [--nnodes 1] [--node-rank 0] [--master-addr 127.0.0.1]
 *             [--master-port 29500] [--grace-s 30] [--] PROGRAM [ARG ...]
 * Her surece: RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, GROUP_RANK, MASTER_ADDR,
 * MASTER_PORT, NEXUS_RUN=1. Store sunucusunu RANK 0 kendi icinde calistirir (init_process_group).
 * Hata: bir surec sifir disi kodla cikarsa digerlerine --grace-s saniye taninir (iletisim hatasi
 * ya da zaman asimiyla kendileri cikmalari beklenir); hala calisanlar sonlandirilir. Sonunda
 * rank basina cikis kodu tablosu yazilir; donus kodu ilk gozlenen hata grubunun kodudur.
 * FAZ 15 elastik yeniden baslatma (temiz yeniden baslatma, torchrun --max-restarts esdegeri):
 *   [--max-restarts K] [--restart-nproc M]: bir surec sifir disi cikarsa (ve hak varsa) digerleri HEMEN
 *   sonlandirilir ve grup M surecle (varsayilan: ayni sayi) yeniden baslatilir. Her denemede NEXUS_RESTART_COUNT
 *   artar, MASTER_PORT = taban + deneme (onceki store soketi TIME_WAIT'te kalabilir). Program kaldigi yerden
 *   devam etmek icin kendi kaydini yukler (LanguageModelTrainer::resume_latest; rank sayisi degistiyse yeniden
 *   parcalama). Kalan rank'lerle ayni surecte iletisim dunyasini yeniden kurma YOK: tum grup yeniden baslar.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

struct Options {
  int nproc = 1;
  int nnodes = 1;
  int node_rank = 0;
  std::string master_addr = "127.0.0.1";
  int master_port = 29500;
  int grace_s = 30;
  int max_restarts = 0;
  int restart_nproc = 0;  // 0: ayni
  std::vector<std::string> command;
};

[[noreturn]] void usage(const char* why) {
  std::fprintf(stderr, "nexus-run: %s\n(bkz. tools/nexus_run/main.cpp basligi)\n", why);
  std::exit(2);
}

Options parse(int argc, char** argv) {
  Options o;
  int i = 1;
  for (; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) usage(("deger eksik: " + a).c_str());
      return argv[++i];
    };
    if (a == "--") {
      ++i;
      break;
    }
    if (a == "--nproc-per-node" || a == "--nproc_per_node") o.nproc = std::stoi(value());
    else if (a == "--nnodes") o.nnodes = std::stoi(value());
    else if (a == "--node-rank" || a == "--node_rank") o.node_rank = std::stoi(value());
    else if (a == "--master-addr" || a == "--master_addr") o.master_addr = value();
    else if (a == "--master-port" || a == "--master_port") o.master_port = std::stoi(value());
    else if (a == "--grace-s") o.grace_s = std::stoi(value());
    else if (a == "--max-restarts" || a == "--max_restarts") o.max_restarts = std::stoi(value());
    else if (a == "--restart-nproc") o.restart_nproc = std::stoi(value());
    else if (a.rfind("--", 0) == 0) usage(("bilinmeyen bayrak: " + a).c_str());
    else break;
  }
  for (; i < argc; ++i) o.command.emplace_back(argv[i]);
  if (o.command.empty()) usage("calistirilacak program yok");
  if (o.nproc < 1 || o.nnodes < 1 || o.node_rank < 0 || o.node_rank >= o.nnodes) usage("gecersiz rank yapilandirmasi");
  return o;
}

std::vector<std::pair<std::string, std::string>> rank_env(const Options& o, int local, int attempt) {
  const int rank = o.node_rank * o.nproc + local;
  return {{"RANK", std::to_string(rank)},
          {"LOCAL_RANK", std::to_string(local)},
          {"WORLD_SIZE", std::to_string(o.nnodes * o.nproc)},
          {"LOCAL_WORLD_SIZE", std::to_string(o.nproc)},
          {"GROUP_RANK", std::to_string(o.node_rank)},
          {"MASTER_ADDR", o.master_addr},
          {"MASTER_PORT", std::to_string(o.master_port + attempt)},
          {"NEXUS_RESTART_COUNT", std::to_string(attempt)},
          {"NEXUS_RUN", "1"}};
}

#if defined(_WIN32)
std::string quote(const std::string& a) {
  if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
  std::string q = "\"";
  for (char c : a) {
    if (c == '"') q += '\\';
    q += c;
  }
  return q + "\"";
}

std::vector<char> environment_block(const std::vector<std::pair<std::string, std::string>>& extra) {
  std::vector<std::string> vars;
  LPCH env = GetEnvironmentStringsA();
  for (LPCH p = env; *p != '\0'; p += std::strlen(p) + 1) {
    const std::string v = p;
    bool overridden = false;
    for (const auto& [k, val] : extra) {
      if (_strnicmp(v.c_str(), (k + "=").c_str(), k.size() + 1) == 0) overridden = true;
    }
    if (!overridden) vars.push_back(v);
  }
  FreeEnvironmentStringsA(env);
  for (const auto& [k, val] : extra) vars.push_back(k + "=" + val);
  std::vector<char> block;
  for (const std::string& v : vars) {
    block.insert(block.end(), v.begin(), v.end());
    block.push_back('\0');
  }
  block.push_back('\0');
  return block;
}
#endif

}  // namespace

int run_once(const Options& o, int attempt, bool can_restart) {
  const int n = o.nproc;
  std::vector<int> codes(static_cast<std::size_t>(n), -1);
  std::vector<bool> running(static_cast<std::size_t>(n), true);
  std::vector<bool> killed(static_cast<std::size_t>(n), false);
  // Cikis zamani yalniz ilk gozlenen hata grubundaki surecleri siralar.
  // Bu zaman damgasi nedensel kok hatayi tek basina kanitlamaz.
  // POSIX'te waitpid yoklama sirasi kullanilir.
  std::vector<std::uint64_t> exit_time(static_cast<std::size_t>(n), ~std::uint64_t{0});
  const auto t0 = std::chrono::steady_clock::now();
#if defined(_WIN32)
  std::string cmdline;
  for (const std::string& a : o.command) cmdline += (cmdline.empty() ? "" : " ") + quote(a);
  std::vector<PROCESS_INFORMATION> procs(static_cast<std::size_t>(n));
  for (int r = 0; r < n; ++r) {
    std::vector<char> env = environment_block(rank_env(o, r, attempt));
    std::vector<char> line(cmdline.begin(), cmdline.end());
    line.push_back('\0');
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    if (!CreateProcessA(nullptr, line.data(), nullptr, nullptr, TRUE, 0, env.data(), nullptr, &si,
                        &procs[static_cast<std::size_t>(r)])) {
      std::fprintf(stderr, "nexus-run: rank %d baslatilamadi (hata %lu): %s\n", r, GetLastError(), cmdline.c_str());
      for (int k = 0; k < r; ++k) TerminateProcess(procs[static_cast<std::size_t>(k)].hProcess, 1);
      return 1;
    }
  }
  auto poll = [&](int r) -> bool {  // true: bitti
    DWORD code = 0;
    GetExitCodeProcess(procs[static_cast<std::size_t>(r)].hProcess, &code);
    if (code == STILL_ACTIVE && WaitForSingleObject(procs[static_cast<std::size_t>(r)].hProcess, 0) == WAIT_TIMEOUT) return false;
    codes[static_cast<std::size_t>(r)] = static_cast<int>(code);
    FILETIME c{}, e{}, k{}, u{};
    if (GetProcessTimes(procs[static_cast<std::size_t>(r)].hProcess, &c, &e, &k, &u)) {
      exit_time[static_cast<std::size_t>(r)] = (static_cast<std::uint64_t>(e.dwHighDateTime) << 32) | e.dwLowDateTime;
    }
    return true;
  };
  auto kill = [&](int r) { TerminateProcess(procs[static_cast<std::size_t>(r)].hProcess, 137); };
#else
  std::uint64_t poll_order = 0;
  std::vector<pid_t> pids(static_cast<std::size_t>(n));
  for (int r = 0; r < n; ++r) {
    const pid_t pid = fork();
    if (pid == 0) {
      for (const auto& [k, v] : rank_env(o, r, attempt)) setenv(k.c_str(), v.c_str(), 1);
      std::vector<char*> args;
      for (const std::string& a : o.command) args.push_back(const_cast<char*>(a.c_str()));
      args.push_back(nullptr);
      execvp(args[0], args.data());
      std::perror("nexus-run: execvp");
      _exit(127);
    }
    pids[static_cast<std::size_t>(r)] = pid;
  }
  auto poll = [&](int r) -> bool {
    int status = 0;
    if (waitpid(pids[static_cast<std::size_t>(r)], &status, WNOHANG) == 0) return false;
    codes[static_cast<std::size_t>(r)] = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    exit_time[static_cast<std::size_t>(r)] = ++poll_order;
    return true;
  };
  auto kill = [&](int r) { ::kill(pids[static_cast<std::size_t>(r)], SIGKILL); };
#endif
  int first_fail = -1;
  std::chrono::steady_clock::time_point fail_time{};
  for (;;) {
    int alive = 0;
    // Later cleanup exits must not replace a failure already observed by the
    // launcher. OS exit-completion timestamps do not establish causal order.
    const bool selecting_first_failure = first_fail < 0;
    for (int r = 0; r < n; ++r) {
      if (!running[static_cast<std::size_t>(r)]) continue;
      if (poll(r)) {
        running[static_cast<std::size_t>(r)] = false;
        if (selecting_first_failure && codes[static_cast<std::size_t>(r)] != 0) {
          if (first_fail < 0 || exit_time[static_cast<std::size_t>(r)] < exit_time[static_cast<std::size_t>(first_fail)]) {
            first_fail = r;
          }
        }
      } else {
        ++alive;
      }
    }
    if (selecting_first_failure && first_fail >= 0) {
      fail_time = std::chrono::steady_clock::now();
      std::fprintf(stderr, "nexus-run: rank %d kod %d ile cikti; digerlerine %d s taninir\n", first_fail,
                   codes[static_cast<std::size_t>(first_fail)], can_restart ? 0 : o.grace_s);
    }
    if (alive == 0) break;
    if (first_fail >= 0 && std::chrono::steady_clock::now() - fail_time > std::chrono::seconds(can_restart ? 0 : o.grace_s)) {
      for (int r = 0; r < n; ++r) {
        if (running[static_cast<std::size_t>(r)]) {
          kill(r);
          killed[static_cast<std::size_t>(r)] = true;
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "nexus-run: deneme %d, %d surec, %.1f s\n", attempt, n, secs);
  for (int r = 0; r < n; ++r) {
    std::fprintf(stderr, "  rank %d: %s %d\n", o.node_rank * n + r, killed[static_cast<std::size_t>(r)] ? "sonlandirildi" : "cikis",
                 codes[static_cast<std::size_t>(r)]);
  }
#if defined(_WIN32)
  for (auto& p : procs) {
    CloseHandle(p.hProcess);
    CloseHandle(p.hThread);
  }
#endif
  if (first_fail >= 0) {
    std::fprintf(stderr, "nexus-run: ilk basarisiz: rank %d (kod %d)\n", o.node_rank * n + first_fail,
                 codes[static_cast<std::size_t>(first_fail)]);
  }
  return first_fail >= 0 ? (codes[static_cast<std::size_t>(first_fail)] != 0 ? codes[static_cast<std::size_t>(first_fail)] : 1) : 0;
}

int main(int argc, char** argv) {
  Options o = parse(argc, argv);
  for (int attempt = 0;; ++attempt) {
    const bool can_restart = attempt < o.max_restarts;
    const int rc = run_once(o, attempt, can_restart);
    if (rc == 0 || !can_restart) {
      return rc;
    }
    if (o.restart_nproc > 0) {
      o.nproc = o.restart_nproc;
    }
    std::fprintf(stderr, "nexus-run: yeniden baslatma %d / %d, %d surec\n", attempt + 1, o.max_restarts, o.nproc);
  }
}
