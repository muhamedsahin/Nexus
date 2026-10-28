/** Official PJRT ABI and failure ownership tests; vendor execution is verified separately.
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif
 * ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.
 */
#include "nexus/backend/pjrt.hpp"
#include "nexus/core/error.hpp"
#include "nexus_test.hpp"
#include <string>
#include <vector>
extern "C" void NexusPjrtTestMode(int);
extern "C" int NexusPjrtTestLive();
extern "C" int NexusPjrtTestInitializations();
using nexus_test::expect;
namespace pjrt = nexus::backend::pjrt;

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string path = argv[1];
  const std::string mlir = R"(module { func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    %s = stablehlo.add %a, %b : tensor<2x2xf32>
    %d = stablehlo.subtract %a, %b : tensor<2x2xf32>
    return %s, %d : tensor<2x2xf32>, tensor<2x2xf32>
  } })";
  try {
    NexusPjrtTestMode(0);
    const auto init = NexusPjrtTestInitializations();
    {
      pjrt::Client a(path), b(path);
      expect(a.uses_external_plugin() && a.platform_name() == "nexus-pjrt-contract-test" && a.device_count() == 1,
             "official ABI client reports the real plugin identity");
      expect(NexusPjrtTestInitializations() == init + 1, "shared library is initialized once for concurrent clients");
      auto result = a.compile_and_execute(mlir, {{1,2,3,4}, {5,6,7,8}}, {{2,2},{2,2}});
      expect(result == std::vector<std::vector<float>>{{6,8,10,12},{-4,-4,-4,-4}}, "nested arguments and multiple opaque outputs have correct values/layout");
      expect(NexusPjrtTestLive() == 2, "only the two clients remain after execution");
      const auto empty = a.compile_and_execute(mlir, {{},{}}, {{0},{0}});
      expect(empty.size() == 2 && empty[0].empty() && empty[1].empty(), "zero-size host results release all events and buffers");
    }
    expect(NexusPjrtTestLive() == 0, "clients are destroyed before the adapter releases its library");
    for (int mode = 1; mode <= 14; ++mode) {
      NexusPjrtTestMode(mode);
      bool rejected = false;
      try { pjrt::Client c(path); (void)c.compile_and_execute(mlir, {{1,2,3,4},{5,6,7,8}}, {{2,2},{2,2}}); }
      catch (const nexus::TensorError&) { rejected = true; }
      expect(rejected, "injected API failure " + std::to_string(mode) + " is reported without interpreter fallback");
      expect(NexusPjrtTestLive() == 0, "injected API failure " + std::to_string(mode) + " releases every owned object/error/event");
    }
    NexusPjrtTestMode(0);
  } catch (const std::exception& error) { nexus_test::expect(false, error.what()); }
  return nexus_test::finish("official PJRT ABI contract (test double, not vendor XLA)");
}
