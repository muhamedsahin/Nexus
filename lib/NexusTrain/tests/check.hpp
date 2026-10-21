#pragma once

#include <iostream>

inline void check_impl(bool ok, const char* expr, const char* file, int line, int& failed) {
  if (!ok) {
    std::cerr << file << ":" << line << "  " << expr << "\n";
    ++failed;
  }
}

#define CHECK(expr) ::check_impl(static_cast<bool>(expr), #expr, __FILE__, __LINE__, failed)
