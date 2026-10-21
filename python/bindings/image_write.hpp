#pragma once

#include <string>
#include <vector>

namespace nexus::pyimage {

bool write_image(const std::string& path, int width, int height, int channels,
                 const std::vector<unsigned char>& hwc);

}
