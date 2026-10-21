#include <iostream>

int test_trainer_basic_loop();
int test_callbacks();
int test_checkpoint();
int test_metrics();
int test_gradient_accumulation();
int test_early_stopping();
int test_prefetcher();
int test_mixed_precision();
int test_reproducibility();
int test_distributed();

int main() {
  int failed = 0;
  failed += test_trainer_basic_loop();
  failed += test_callbacks();
  failed += test_metrics();
  failed += test_gradient_accumulation();
  failed += test_early_stopping();
  failed += test_checkpoint();
  failed += test_prefetcher();
  failed += test_mixed_precision();
  failed += test_reproducibility();
  failed += test_distributed();
  if (failed == 0) {
    std::cout << "nexus_train tests passed\n";
    return 0;
  }
  std::cerr << failed << " check(s) failed\n";
  return 1;
}
