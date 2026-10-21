#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image_write.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace nexus::pyimage {

bool write_image(const std::string& path, int width, int height, int channels,
                 const std::vector<unsigned char>& hwc) {
    std::string ext;
    const auto dot = path.find_last_of('.');
    if (dot != std::string::npos) {
        ext = path.substr(dot + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    }
    if (ext == "jpg" || ext == "jpeg") {
        return stbi_write_jpg(path.c_str(), width, height, channels, hwc.data(), 90) != 0;
    }
    if (ext == "bmp") {
        return stbi_write_bmp(path.c_str(), width, height, channels, hwc.data()) != 0;
    }
    const int stride = width * channels;
    return stbi_write_png(path.c_str(), width, height, channels, hwc.data(), stride) != 0;
}

}  // namespace nexus::pyimage
