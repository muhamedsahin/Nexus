#pragma once

/**
 * @file trace.hpp
 * @brief Chrome iz biciminde (trace event JSON) zamanlama dokumu.
 *
 * `dump_chrome_trace(path)` Profiler kaydindaki araliklari ve sayac
 * anliklarini `chrome://tracing` / Perfetto'nun actigi JSON bicimine yazar:
 * araliklar "X" (complete), sayaclar "C" (counter) olayidir. Iz dosyasi
 * FAZ 0 benchmark klasorune makinayla okunur sekilde kaydedilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <string>
#include <string_view>

namespace nexus {
namespace prof {

/**
 * @brief Profiler araliklarini ve sayaclari Chrome iz JSON'u olarak yazar.
 *
 * Dosya acilamazsa false doner; kayit bos olsa da gecerli (bos) iz yazar.
 * Zaman damgalari mikrosaniyedir (Chrome iz sozlesmesi).
 */
[[nodiscard]] bool dump_chrome_trace(std::string_view path);

/**
 * @brief Iz JSON metnini bir dizeye uretir; testler dosya sistemine
 *        dokunmadan bicimi dogrulamak icin kullanir.
 */
[[nodiscard]] std::string chrome_trace_json();

}  // namespace prof
}  // namespace nexus
