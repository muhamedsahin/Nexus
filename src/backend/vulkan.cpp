/**
 * @file vulkan.cpp
 * @brief FAZ 19.C: Vulkan 1.1+ / 1.2 SPIR-V compute surucusu (dinamik yukleyici, havuzlu VkBuffer,
 *        descriptor set onbellegi ve GEMM / elementwise / reduce / norm / softmax / RoPE / SwiGLU /
 *        FlashAttention / Q8-Q4 nicemlenmis GEMM / AdamW SPIR-V cekirdekleri).
 *
 * Sinirlama (FAZ 19.C): Vulkan arka ucu tasinabilir cikarim ve kucuk/orta olcekli egitim icindir;
 * buyuk olcekli dagitik egitim CUDA/ROCm uzerindedir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/backend/backend.hpp"
#include "nexus/core/error.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(NEXUS_ENABLE_VULKAN)
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nexus::backend {
namespace {

#define NEXUS_VK_LIST(X) \
  X(vkGetInstanceProcAddr) X(vkCreateInstance) X(vkEnumeratePhysicalDevices) \
  X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceMemoryProperties) \
  X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkCreateDevice) X(vkDestroyDevice) X(vkGetDeviceQueue) \
  X(vkDeviceWaitIdle) X(vkCreateBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) \
  X(vkBindBufferMemory) X(vkFreeMemory) X(vkDestroyBuffer) X(vkMapMemory) X(vkUnmapMemory) \
  X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout) \
  X(vkCreatePipelineLayout) X(vkCreateComputePipelines) X(vkDestroyPipeline) \
  X(vkDestroyPipelineLayout) X(vkDestroyDescriptorSetLayout) X(vkCreateDescriptorPool) \
  X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) \
  X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) \
  X(vkResetCommandBuffer) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) \
  X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) \
  X(vkCmdCopyBuffer) X(vkCmdFillBuffer) X(vkCreateFence) X(vkDestroyFence) \
  X(vkResetFences) X(vkWaitForFences) X(vkQueueSubmit)

struct Api {
  void* library = nullptr;
#define NEXUS_VK_FIELD(name) PFN_##name name = nullptr;
  NEXUS_VK_LIST(NEXUS_VK_FIELD)
#undef NEXUS_VK_FIELD
};

void* load_library(const char* name) {
#if defined(_WIN32)
  return static_cast<void*>(LoadLibraryA(name));
#else
  return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}
void* load_symbol(void* library, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
  return dlsym(library, name);
#endif
}

Api& api() {
  static Api instance = [] {
    Api a;
    const char* names[] = {"vulkan-1.dll", "libvulkan.so.1", "libvulkan.so", "libMoltenVK.dylib"};
    for (const char* name : names) if ((a.library = load_library(name)) != nullptr) break;
    if (a.library == nullptr) return a;
    a.vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(load_symbol(a.library, "vkGetInstanceProcAddr"));
    if (a.vkGetInstanceProcAddr != nullptr) {
      a.vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(a.vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
    }
    return a;
  }();
  return instance;
}

void check(VkResult result, const char* what) {
  if (result != VK_SUCCESS) throw TensorError(std::string("Nexus Vulkan: ") + what + " hata=" + std::to_string(result));
}
std::uint32_t ceil_div(std::uint64_t value, std::uint32_t divisor) {
  return static_cast<std::uint32_t>((value + divisor - 1) / divisor);
}
std::string version_string(std::uint32_t value) {
  return std::to_string(VK_VERSION_MAJOR(value)) + "." + std::to_string(VK_VERSION_MINOR(value)) + "." +
         std::to_string(VK_VERSION_PATCH(value));
}

struct Buffer {
  VkBuffer handle = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
};
struct Pipeline {
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDescriptorSet descriptors = VK_NULL_HANDLE;
};

struct Context {
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  std::uint32_t queue_family = 0;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkCommandBuffer command = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  Pipeline elementwise, matmul, reduce, norm_softmax, special_ops, attention, quant_matmul;
  Buffer staging{};
  void* staging_mapped = nullptr;
  Buffer dummy{};
  std::unordered_map<VkDeviceSize, std::vector<Buffer*>> free;
  std::unordered_set<Buffer*> all;
  mutable std::mutex mutex;
  AllocatorStats stats{};

  ~Context();
};

VkInstance instance_handle() {
  static VkInstance instance = [] {
    Api& a = api();
    if (a.vkCreateInstance == nullptr) return VkInstance(VK_NULL_HANDLE);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Nexus";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "Nexus FAZ19";
    app.engineVersion = VK_MAKE_VERSION(19, 0, 0);
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    VkInstance out = VK_NULL_HANDLE;
    if (a.vkCreateInstance(&info, nullptr, &out) != VK_SUCCESS) return VkInstance(VK_NULL_HANDLE);
#define NEXUS_VK_LOAD(name) if constexpr (std::string_view(#name) != "vkGetInstanceProcAddr" && std::string_view(#name) != "vkCreateInstance") \
    a.name = reinterpret_cast<PFN_##name>(a.vkGetInstanceProcAddr(out, #name));
    NEXUS_VK_LIST(NEXUS_VK_LOAD)
#undef NEXUS_VK_LOAD
    return out;
  }();
  return instance;
}

std::uint32_t memory_type(Context& c, const VkMemoryRequirements& requirements, VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties properties{};
  api().vkGetPhysicalDeviceMemoryProperties(c.physical, &properties);
  for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
        (properties.memoryTypes[i].propertyFlags & flags) == flags) return i;
  }
  throw TensorError("Nexus Vulkan: uygun bellek turu yok");
}

Buffer* raw_buffer(Context& c, std::size_t bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
  if (bytes == 0) throw TensorError("Nexus Vulkan: sifir bayt tampon ayirilamaz");
  auto* out = new Buffer();
  out->size = (bytes + 255u) & ~std::size_t(255u);
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = out->size;
  info.usage = usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  try {
    check(api().vkCreateBuffer(c.device, &info, nullptr, &out->handle), "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    api().vkGetBufferMemoryRequirements(c.device, out->handle, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type(c, requirements, properties);
    check(api().vkAllocateMemory(c.device, &allocation, nullptr, &out->memory), "vkAllocateMemory");
    check(api().vkBindBufferMemory(c.device, out->handle, out->memory, 0), "vkBindBufferMemory");
  } catch (...) {
    if (out->handle != VK_NULL_HANDLE) api().vkDestroyBuffer(c.device, out->handle, nullptr);
    if (out->memory != VK_NULL_HANDLE) api().vkFreeMemory(c.device, out->memory, nullptr);
    delete out;
    throw;
  }
  return out;
}

void destroy_buffer(Context& c, Buffer* buffer) {
  if (buffer == nullptr) return;
  if (buffer->handle != VK_NULL_HANDLE) api().vkDestroyBuffer(c.device, buffer->handle, nullptr);
  if (buffer->memory != VK_NULL_HANDLE) api().vkFreeMemory(c.device, buffer->memory, nullptr);
  delete buffer;
}

void destroy_pipeline(VkDevice dev, Pipeline& p) {
  if (p.pipeline != VK_NULL_HANDLE && api().vkDestroyPipeline) api().vkDestroyPipeline(dev, p.pipeline, nullptr);
  if (p.layout != VK_NULL_HANDLE && api().vkDestroyPipelineLayout) api().vkDestroyPipelineLayout(dev, p.layout, nullptr);
  if (p.descriptor_layout != VK_NULL_HANDLE && api().vkDestroyDescriptorSetLayout) {
    api().vkDestroyDescriptorSetLayout(dev, p.descriptor_layout, nullptr);
  }
  p = Pipeline{};
}

Context::~Context() {
  if (device == VK_NULL_HANDLE) return;
  if (api().vkDeviceWaitIdle) api().vkDeviceWaitIdle(device);
  if (staging_mapped != nullptr && api().vkUnmapMemory) {
    api().vkUnmapMemory(device, staging.memory);
    staging_mapped = nullptr;
  }
  if (staging.handle != VK_NULL_HANDLE) api().vkDestroyBuffer(device, staging.handle, nullptr);
  if (staging.memory != VK_NULL_HANDLE) api().vkFreeMemory(device, staging.memory, nullptr);
  if (dummy.handle != VK_NULL_HANDLE) api().vkDestroyBuffer(device, dummy.handle, nullptr);
  if (dummy.memory != VK_NULL_HANDLE) api().vkFreeMemory(device, dummy.memory, nullptr);
  for (Buffer* buf : all) {
    destroy_buffer(*this, buf);
  }
  all.clear();
  free.clear();
  destroy_pipeline(device, elementwise);
  destroy_pipeline(device, matmul);
  destroy_pipeline(device, reduce);
  destroy_pipeline(device, norm_softmax);
  destroy_pipeline(device, special_ops);
  destroy_pipeline(device, attention);
  destroy_pipeline(device, quant_matmul);
  if (descriptor_pool != VK_NULL_HANDLE && api().vkDestroyDescriptorPool) {
    api().vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  }
  if (fence != VK_NULL_HANDLE && api().vkDestroyFence) api().vkDestroyFence(device, fence, nullptr);
  if (command_pool != VK_NULL_HANDLE && api().vkDestroyCommandPool) {
    api().vkDestroyCommandPool(device, command_pool, nullptr);
  }
  if (api().vkDestroyDevice) api().vkDestroyDevice(device, nullptr);
}

Pipeline make_pipeline(Context& c, const std::uint32_t* code, std::size_t code_bytes,
                       std::uint32_t bindings, std::uint32_t push_bytes) {
  Pipeline out;
  std::vector<VkDescriptorSetLayoutBinding> layout_bindings(bindings);
  for (std::uint32_t i = 0; i < bindings; ++i) {
    layout_bindings[i].binding = i;
    layout_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layout_bindings[i].descriptorCount = 1;
    layout_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layout.bindingCount = bindings;
  layout.pBindings = layout_bindings.data();
  check(api().vkCreateDescriptorSetLayout(c.device, &layout, nullptr, &out.descriptor_layout), "descriptor layout");
  VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
  VkPipelineLayoutCreateInfo pipeline_layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeline_layout.setLayoutCount = 1;
  pipeline_layout.pSetLayouts = &out.descriptor_layout;
  pipeline_layout.pushConstantRangeCount = 1;
  pipeline_layout.pPushConstantRanges = &range;
  check(api().vkCreatePipelineLayout(c.device, &pipeline_layout, nullptr, &out.layout), "pipeline layout");
  VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  shader.codeSize = code_bytes;
  shader.pCode = code;
  VkShaderModule module = VK_NULL_HANDLE;
  check(api().vkCreateShaderModule(c.device, &shader, nullptr, &module), "shader module");
  VkComputePipelineCreateInfo compute{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  compute.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  compute.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  compute.stage.module = module;
  compute.stage.pName = "main";
  compute.layout = out.layout;
  const VkResult result = api().vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &compute, nullptr, &out.pipeline);
  api().vkDestroyShaderModule(c.device, module, nullptr);
  check(result, "compute pipeline");
  return out;
}

constexpr std::uint32_t elementwise_spirv[] = {
#include "spirv/elementwise.inc"
};
constexpr std::uint32_t matmul_spirv[] = {
#include "spirv/matmul.inc"
};
constexpr std::uint32_t reduce_spirv[] = {
#include "spirv/reduce_sum.inc"
};
constexpr std::uint32_t norm_softmax_spirv[] = {
#include "spirv/norm_softmax.inc"
};
constexpr std::uint32_t special_ops_spirv[] = {
#include "spirv/special_ops.inc"
};
constexpr std::uint32_t attention_spirv[] = {
#include "spirv/attention.inc"
};
constexpr std::uint32_t quant_matmul_spirv[] = {
#include "spirv/quant_matmul.inc"
};

std::unique_ptr<Context> make_context(VkPhysicalDevice physical) {
  auto c = std::make_unique<Context>();
  c->physical = physical;
  std::uint32_t family_count = 0;
  api().vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  api().vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
  for (std::uint32_t i = 0; i < family_count; ++i) {
    if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) { c->queue_family = i; break; }
  }
  if (family_count == 0 || (families[c->queue_family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
    throw TensorError("Nexus Vulkan: compute kuyrugu yok");
  }
  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue.queueFamilyIndex = c->queue_family;
  queue.queueCount = 1;
  queue.pQueuePriorities = &priority;
  VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device.queueCreateInfoCount = 1;
  device.pQueueCreateInfos = &queue;
  check(api().vkCreateDevice(physical, &device, nullptr, &c->device), "vkCreateDevice");
  api().vkGetDeviceQueue(c->device, c->queue_family, 0, &c->queue);
  VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool.queueFamilyIndex = c->queue_family;
  check(api().vkCreateCommandPool(c->device, &pool, nullptr, &c->command_pool), "command pool");
  VkCommandBufferAllocateInfo command{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  command.commandPool = c->command_pool;
  command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command.commandBufferCount = 1;
  check(api().vkAllocateCommandBuffers(c->device, &command, &c->command), "command buffer");
  VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  check(api().vkCreateFence(c->device, &fence, nullptr, &c->fence), "fence");

  Buffer* d = raw_buffer(*c, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  c->dummy = *d;
  delete d;

  c->elementwise = make_pipeline(*c, elementwise_spirv, sizeof(elementwise_spirv), 3, 12);
  c->matmul = make_pipeline(*c, matmul_spirv, sizeof(matmul_spirv), 3, 36);
  c->reduce = make_pipeline(*c, reduce_spirv, sizeof(reduce_spirv), 2, 4);
  c->norm_softmax = make_pipeline(*c, norm_softmax_spirv, sizeof(norm_softmax_spirv), 4, 16);
  c->special_ops = make_pipeline(*c, special_ops_spirv, sizeof(special_ops_spirv), 4, 44);
  c->attention = make_pipeline(*c, attention_spirv, sizeof(attention_spirv), 4, 20);
  c->quant_matmul = make_pipeline(*c, quant_matmul_spirv, sizeof(quant_matmul_spirv), 4, 20);

  VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32};
  VkDescriptorPoolCreateInfo descriptors{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  descriptors.maxSets = 7;
  descriptors.poolSizeCount = 1;
  descriptors.pPoolSizes = &size;
  check(api().vkCreateDescriptorPool(c->device, &descriptors, nullptr, &c->descriptor_pool), "descriptor pool");
  const VkDescriptorSetLayout layouts[] = {
      c->elementwise.descriptor_layout,  c->matmul.descriptor_layout,
      c->reduce.descriptor_layout,       c->norm_softmax.descriptor_layout,
      c->special_ops.descriptor_layout,  c->attention.descriptor_layout,
      c->quant_matmul.descriptor_layout,
  };
  VkDescriptorSetAllocateInfo sets{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  sets.descriptorPool = c->descriptor_pool;
  sets.descriptorSetCount = 7;
  sets.pSetLayouts = layouts;
  VkDescriptorSet allocated[7]{};
  check(api().vkAllocateDescriptorSets(c->device, &sets, allocated), "descriptor sets");
  c->elementwise.descriptors = allocated[0];
  c->matmul.descriptors = allocated[1];
  c->reduce.descriptors = allocated[2];
  c->norm_softmax.descriptors = allocated[3];
  c->special_ops.descriptors = allocated[4];
  c->attention.descriptors = allocated[5];
  c->quant_matmul.descriptors = allocated[6];
  return c;
}

void begin(Context& c) {
  check(api().vkResetFences(c.device, 1, &c.fence), "reset fence");
  check(api().vkResetCommandBuffer(c.command, 0), "reset command");
  VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(api().vkBeginCommandBuffer(c.command, &info), "begin command");
}
void submit(Context& c) {
  check(api().vkEndCommandBuffer(c.command), "end command");
  VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  info.commandBufferCount = 1;
  info.pCommandBuffers = &c.command;
  check(api().vkQueueSubmit(c.queue, 1, &info, c.fence), "queue submit");
  check(api().vkWaitForFences(c.device, 1, &c.fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max()), "fence wait");
}
void bind_buffers(Context& c, Pipeline& pipeline, const Buffer* const* buffers, std::uint32_t count) {
  VkDescriptorBufferInfo infos[4]{};
  VkWriteDescriptorSet writes[4]{};
  for (std::uint32_t i = 0; i < count; ++i) {
    const Buffer* b = buffers[i] != nullptr ? buffers[i] : &c.dummy;
    infos[i] = {b->handle, 0, b->size};
    writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[i].dstSet = pipeline.descriptors;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  api().vkUpdateDescriptorSets(c.device, count, writes, 0, nullptr);
}
void dispatch(Context& c, Pipeline& pipeline, const Buffer* const* buffers, std::uint32_t binding_count,
              const void* constants, std::uint32_t constant_bytes, std::uint32_t x, std::uint32_t y = 1) {
  std::lock_guard lock(c.mutex);
  bind_buffers(c, pipeline, buffers, binding_count);
  begin(c);
  api().vkCmdBindPipeline(c.command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
  api().vkCmdBindDescriptorSets(c.command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1,
                                &pipeline.descriptors, 0, nullptr);
  api().vkCmdPushConstants(c.command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constant_bytes, constants);
  api().vkCmdDispatch(c.command, x, y, 1);
  submit(c);
}

class VulkanDriver final : public Driver {
 public:
  VulkanDriver() {
    if (instance_handle() == VK_NULL_HANDLE) { reason_ = "Vulkan 1.1 loader/instance olusturulamadi"; return; }
    std::uint32_t count = 0;
    if (api().vkEnumeratePhysicalDevices(instance_handle(), &count, nullptr) != VK_SUCCESS || count == 0) {
      reason_ = "Vulkan fiziksel cihazi yok";
      return;
    }
    std::vector<VkPhysicalDevice> physical(count);
    check(api().vkEnumeratePhysicalDevices(instance_handle(), &count, physical.data()), "physical devices");
    for (VkPhysicalDevice device : physical) {
      try {
        auto context = make_context(device);
        VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &subgroup;
        VkPhysicalDeviceProperties properties{};
        if (api().vkGetPhysicalDeviceProperties2 != nullptr) {
          api().vkGetPhysicalDeviceProperties2(device, &props2);
          properties = props2.properties;
        } else {
          api().vkGetPhysicalDeviceProperties(device, &properties);
          subgroup.subgroupSize = 32;
        }
        VkPhysicalDeviceMemoryProperties memory{};
        api().vkGetPhysicalDeviceMemoryProperties(device, &memory);
        std::uint64_t bytes = 0;
        for (std::uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
          if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) bytes += memory.memoryHeaps[i].size;
        }
        DeviceInfo info;
        info.device = DeviceRef::vulkan(static_cast<int>(contexts_.size()));
        info.name = properties.deviceName;
        info.vendor = std::to_string(properties.vendorID) + ":" + std::to_string(properties.deviceID);
        info.api_version = version_string(properties.apiVersion);
        info.driver_version = version_string(properties.driverVersion);
        info.memory_bytes = bytes;
        info.subgroup_size = subgroup.subgroupSize > 0 ? subgroup.subgroupSize : 32;
        info.unified_memory = (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
        devices_.push_back(std::move(info));
        contexts_.push_back(std::move(context));
      } catch (const std::exception& e) {
        if (reason_.empty()) reason_ = e.what();
      }
    }
    if (contexts_.empty() && reason_.empty()) reason_ = "Vulkan compute cihazi acilamadi";
  }
  Kind kind() const noexcept override { return Kind::Vulkan; }
  std::string_view name() const noexcept override { return "vulkan-spirv"; }
  std::string_view version() const noexcept override { return "1.2"; }
  const std::vector<DeviceInfo>& devices() const noexcept override { return devices_; }
  std::string unavailable_reason() const override { return reason_; }
  bool supports(Operation op) const noexcept override {
    return static_cast<unsigned>(op) <= static_cast<unsigned>(Operation::AdamW);
  }
  void* allocate(int device, std::size_t bytes) override {
    if (bytes == 0) return nullptr;
    Context& c = context(device);
    std::lock_guard lock(c.mutex);
    const VkDeviceSize size = (bytes + 255u) & ~std::size_t(255u);
    Buffer* buffer = nullptr;
    auto& free = c.free[size];
    ++c.stats.allocations;
    if (!free.empty()) {
      buffer = free.back();
      free.pop_back();
      ++c.stats.cache_hits;
    } else {
      buffer = raw_buffer(c, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      c.all.insert(buffer);
      c.stats.reserved_bytes += buffer->size;
    }
    ++c.stats.live_blocks;
    c.stats.current_bytes += buffer->size;
    return buffer;
  }
  void deallocate(int device, void* pointer) noexcept override {
    if (pointer == nullptr) return;
    try {
      Context& c = context(device);
      std::lock_guard lock(c.mutex);
      auto* buffer = static_cast<Buffer*>(pointer);
      if (c.all.find(buffer) == c.all.end()) return;
      c.free[buffer->size].push_back(buffer);
      if (c.stats.live_blocks > 0) --c.stats.live_blocks;
      c.stats.current_bytes = c.stats.current_bytes >= buffer->size ? c.stats.current_bytes - buffer->size : 0;
    } catch (...) {}
  }
  void copy(DeviceRef dst_device, void* dst, DeviceRef src_device, const void* src, std::size_t bytes) override;
  void synchronize(int device) override {
    Context& c = context(device);
    std::lock_guard lock(c.mutex);
    check(api().vkDeviceWaitIdle(c.device), "device idle");
  }
  void fill(DeviceRef device, float* dst, std::size_t count, float value) override {
    if (count == 0) return;
    Context& c = context(device.index);
    auto* out = static_cast<Buffer*>(static_cast<void*>(dst));
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    std::lock_guard lock(c.mutex);
    begin(c);
    api().vkCmdFillBuffer(c.command, out->handle, 0, count * sizeof(float), bits);
    submit(c);
  }
  void elementwise(Operation op, DeviceRef device, const float* a, const float* b, float* out,
                   std::size_t count, float scalar) override;
  void matmul(DeviceRef device, const MatMulArgs& args) override;
  void reduce_sum(DeviceRef device, const float* input, float* output, std::size_t count) override;
  void softmax(DeviceRef device, const float* input, float* output, int rows, int cols) override;
  void rms_norm(DeviceRef device, const float* x, const float* weight, float* out,
                int rows, int cols, float eps) override;
  void layer_norm(DeviceRef device, const float* x, const float* weight, const float* bias,
                  float* out, int rows, int cols, float eps) override;
  void rope(DeviceRef device, const float* x, float* out, int batch_heads, int seq_len,
            int head_dim, int pos_offset, float theta) override;
  void swiglu(DeviceRef device, const float* x, float* out, int rows, int hidden_dim) override;
  void attention(DeviceRef device, const float* q, const float* k, const float* v, float* out,
                 int batch_heads, int seq_len, int head_dim, bool causal) override;
  void quant_matmul_q8(DeviceRef device, const float* a, const void* w_q8,
                       const float* w_scales, float* c, int m, int n, int k, int group_size) override;
  void quant_matmul_q4(DeviceRef device, const float* a, const void* w_q4,
                       const float* w_scales, float* c, int m, int n, int k, int group_size) override;
  void adamw_step(DeviceRef device, float* param, const float* grad, float* m, float* v,
                  std::size_t count, float lr, float beta1, float beta2, float eps,
                  float weight_decay, int step) override;
  AllocatorStats allocator_stats(int device) const override {
    Context& c = context(device);
    std::lock_guard lock(c.mutex);
    return c.stats;
  }
 private:
  Context& context(int index) const {
    if (index < 0 || static_cast<std::size_t>(index) >= contexts_.size()) throw TensorError("Nexus Vulkan: gecersiz cihaz indeksi");
    return *contexts_[static_cast<std::size_t>(index)];
  }
  std::vector<std::unique_ptr<Context>> contexts_;
  std::vector<DeviceInfo> devices_;
  std::string reason_;
};

void rebuild_staging(Context& c, std::size_t bytes) {
  if (c.staging_mapped != nullptr && c.staging.size >= bytes) return;
  check(api().vkDeviceWaitIdle(c.device), "staging idle");
  if (c.staging_mapped != nullptr) api().vkUnmapMemory(c.device, c.staging.memory);
  c.staging_mapped = nullptr;
  if (c.staging.handle != VK_NULL_HANDLE) api().vkDestroyBuffer(c.device, c.staging.handle, nullptr);
  if (c.staging.memory != VK_NULL_HANDLE) api().vkFreeMemory(c.device, c.staging.memory, nullptr);
  c.staging = Buffer{};
  Buffer* staging = raw_buffer(c, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  c.staging = *staging;
  delete staging;
  check(api().vkMapMemory(c.device, c.staging.memory, 0, VK_WHOLE_SIZE, 0, &c.staging_mapped), "map staging");
}

void VulkanDriver::copy(DeviceRef dst_device, void* dst, DeviceRef src_device, const void* src, std::size_t bytes) {
  if (bytes == 0) return;
  const bool dst_vk = dst_device.kind == DeviceKind::Vulkan;
  const bool src_vk = src_device.kind == DeviceKind::Vulkan;
  if (!dst_vk && !src_vk) { std::memcpy(dst, src, bytes); return; }
  if (dst_vk && src_vk && dst_device.index != src_device.index) {
    std::vector<std::byte> host(bytes);
    copy(DeviceRef::cpu(), host.data(), src_device, src, bytes);
    copy(dst_device, dst, DeviceRef::cpu(), host.data(), bytes);
    return;
  }
  Context& c = context(dst_vk ? dst_device.index : src_device.index);
  std::lock_guard lock(c.mutex);
  rebuild_staging(c, bytes);
  VkBufferCopy region{0, 0, bytes};
  begin(c);
  if (dst_vk && src_vk) {
    region.dstOffset = 0;
    api().vkCmdCopyBuffer(c.command, static_cast<const Buffer*>(src)->handle, static_cast<Buffer*>(dst)->handle, 1, &region);
  } else if (dst_vk) {
    std::memcpy(c.staging_mapped, src, bytes);
    api().vkCmdCopyBuffer(c.command, c.staging.handle, static_cast<Buffer*>(dst)->handle, 1, &region);
  } else {
    api().vkCmdCopyBuffer(c.command, static_cast<const Buffer*>(src)->handle, c.staging.handle, 1, &region);
  }
  submit(c);
  if (!dst_vk) std::memcpy(dst, c.staging_mapped, bytes);
}

struct ElementPush { std::uint32_t count; float scalar; std::uint32_t op; };
void VulkanDriver::elementwise(Operation op, DeviceRef device, const float* a, const float* b, float* out,
                               std::size_t count, float scalar) {
  if (count == 0) return;
  if (count > std::numeric_limits<std::uint32_t>::max()) throw TensorError("Nexus Vulkan: elementwise 32-bit sinirini asiyor");
  Context& c = context(device.index);
  const Buffer* buffers[] = {static_cast<const Buffer*>(static_cast<const void*>(a)),
                             b != nullptr ? static_cast<const Buffer*>(static_cast<const void*>(b))
                                          : static_cast<const Buffer*>(static_cast<const void*>(a)),
                             static_cast<const Buffer*>(static_cast<const void*>(out))};
  ElementPush push{static_cast<std::uint32_t>(count), scalar, static_cast<std::uint32_t>(op)};
  dispatch(c, c.elementwise, buffers, 3, &push, sizeof(push), ceil_div((count + 3) / 4, 64));
}

struct MatMulPush {
  std::uint32_t m, n, k, lda, ldb, ldc, flags;
  float alpha, beta;
};
void VulkanDriver::matmul(DeviceRef device, const MatMulArgs& args) {
  if (args.m <= 0 || args.n <= 0 || args.k < 0 || args.lda <= 0 || args.ldb <= 0 || args.ldc <= 0) {
    throw TensorError("Nexus Vulkan: gecersiz GEMM boyutu veya leading dimension");
  }
  Context& c = context(device.index);
  const Buffer* buffers[] = {static_cast<const Buffer*>(static_cast<const void*>(args.a)),
                             static_cast<const Buffer*>(static_cast<const void*>(args.b)),
                             static_cast<const Buffer*>(static_cast<const void*>(args.c))};
  MatMulPush push{static_cast<std::uint32_t>(args.m), static_cast<std::uint32_t>(args.n), static_cast<std::uint32_t>(args.k),
                  static_cast<std::uint32_t>(args.lda), static_cast<std::uint32_t>(args.ldb), static_cast<std::uint32_t>(args.ldc),
                  static_cast<std::uint32_t>((args.a_trans ? 1 : 0) | (args.b_trans ? 2 : 0)), args.alpha, args.beta};
  dispatch(c, c.matmul, buffers, 3, &push, sizeof(push), ceil_div(args.n, 16), ceil_div(args.m, 16));
}

void VulkanDriver::reduce_sum(DeviceRef device, const float* input, float* output, std::size_t count) {
  if (count == 0) {
    fill(device, output, 1, 0.0f);
    return;
  }
  if (count > std::numeric_limits<std::uint32_t>::max()) throw TensorError("Nexus Vulkan: reduce 32-bit sinirini asiyor");
  Context& c = context(device.index);
  const Buffer* current = static_cast<const Buffer*>(static_cast<const void*>(input));
  Buffer* owned = nullptr;
  std::size_t remaining = count;
  while (true) {
    const std::uint32_t groups = std::min(ceil_div(remaining, 256), 65535u);
    const bool last = groups == 1;
    Buffer* next = last ? static_cast<Buffer*>(static_cast<void*>(output))
                        : static_cast<Buffer*>(allocate(device.index, groups * sizeof(float)));
    const Buffer* buffers[] = {current, next};
    const std::uint32_t push = static_cast<std::uint32_t>(remaining);
    dispatch(c, c.reduce, buffers, 2, &push, sizeof(push), groups);
    if (owned != nullptr) deallocate(device.index, owned);
    if (last) break;
    owned = next;
    current = next;
    remaining = groups;
  }
}

struct NormSoftmaxPush {
  std::uint32_t rows;
  std::uint32_t cols;
  std::uint32_t op;
  float eps;
};

void VulkanDriver::softmax(DeviceRef device, const float* input, float* output, int rows, int cols) {
  if (rows <= 0 || cols <= 0) return;
  Context& c = context(device.index);
  const Buffer* buffers[] = {
      static_cast<const Buffer*>(static_cast<const void*>(input)),
      &c.dummy,
      &c.dummy,
      static_cast<const Buffer*>(static_cast<const void*>(output)),
  };
  NormSoftmaxPush push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols), 0u, 0.0f};
  dispatch(c, c.norm_softmax, buffers, 4, &push, sizeof(push), static_cast<std::uint32_t>(rows));
}

void VulkanDriver::rms_norm(DeviceRef device, const float* x, const float* weight, float* out,
                            int rows, int cols, float eps) {
  if (rows <= 0 || cols <= 0) return;
  Context& c = context(device.index);
  const Buffer* buffers[] = {
      static_cast<const Buffer*>(static_cast<const void*>(x)),
      static_cast<const Buffer*>(static_cast<const void*>(weight)),
      &c.dummy,
      static_cast<const Buffer*>(static_cast<const void*>(out)),
  };
  NormSoftmaxPush push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols), 1u, eps};
  dispatch(c, c.norm_softmax, buffers, 4, &push, sizeof(push), static_cast<std::uint32_t>(rows));
}

void VulkanDriver::layer_norm(DeviceRef device, const float* x, const float* weight, const float* bias,
                              float* out, int rows, int cols, float eps) {
  if (rows <= 0 || cols <= 0) return;
  Context& c = context(device.index);
  const Buffer* buffers[] = {
      static_cast<const Buffer*>(static_cast<const void*>(x)),
      static_cast<const Buffer*>(static_cast<const void*>(weight)),
      bias != nullptr ? static_cast<const Buffer*>(static_cast<const void*>(bias)) : &c.dummy,
      static_cast<const Buffer*>(static_cast<const void*>(out)),
  };
  NormSoftmaxPush push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols), 2u, eps};
  dispatch(c, c.norm_softmax, buffers, 4, &push, sizeof(push), static_cast<std::uint32_t>(rows));
}

struct SpecialPush {
  std::uint32_t d0, d1, d2, d3;
  float f0, f1, f2, f3, f4, f5, f6;
};

void VulkanDriver::rope(DeviceRef device, const float* x, float* out, int batch_heads, int seq_len,
                        int head_dim, int pos_offset, float theta) {
  if (batch_heads <= 0 || seq_len <= 0 || head_dim <= 0 || (head_dim & 1) != 0) {
    throw TensorError("Nexus Vulkan: gecersiz RoPE boyutu");
  }
  Context& c = context(device.index);
  const std::uint32_t half_d = static_cast<std::uint32_t>(head_dim / 2);
  const std::uint32_t total_pairs = static_cast<std::uint32_t>(batch_heads) * static_cast<std::uint32_t>(seq_len) * half_d;
  const Buffer* buffers[] = {
      static_cast<const Buffer*>(static_cast<const void*>(out)),
      static_cast<const Buffer*>(static_cast<const void*>(x)),
      &c.dummy,
      &c.dummy,
  };
  SpecialPush push{total_pairs, static_cast<std::uint32_t>(seq_len), half_d,
                   (static_cast<std::uint32_t>(pos_offset) << 8u) | 0u,
                   theta, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  dispatch(c, c.special_ops, buffers, 4, &push, sizeof(push), ceil_div(total_pairs, 64));
}

void VulkanDriver::swiglu(DeviceRef device, const float* x, float* out, int rows, int hidden_dim) {
  if (rows <= 0 || hidden_dim <= 0) return;
  Context& c = context(device.index);
  const std::uint32_t total = static_cast<std::uint32_t>(rows) * static_cast<std::uint32_t>(hidden_dim);
  const Buffer* buffers[] = {
      static_cast<const Buffer*>(static_cast<const void*>(out)),
      static_cast<const Buffer*>(static_cast<const void*>(x)),
      &c.dummy,
      &c.dummy,
  };
  SpecialPush push{total, static_cast<std::uint32_t>(hidden_dim), 0u, 1u,
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  dispatch(c, c.special_ops, buffers, 4, &push, sizeof(push), ceil_div(total, 64));
}

void VulkanDriver::adamw_step(DeviceRef device, float* param, const float* grad, float* m, float* v,
                              std::size_t count, float lr, float beta1, float beta2, float eps,
                              float weight_decay, int step) {
  if (count == 0) return;
  Context& c = context(device.index);
  const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(std::max(1, step)));
  const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(std::max(1, step)));
  const Buffer* buffers[] = {
      static_cast<const Buffer*>(static_cast<const void*>(param)),
      static_cast<const Buffer*>(static_cast<const void*>(grad)),
      static_cast<const Buffer*>(static_cast<const void*>(m)),
      static_cast<const Buffer*>(static_cast<const void*>(v)),
  };
  SpecialPush push{static_cast<std::uint32_t>(count), 0u, 0u, 2u,
                   lr, beta1, beta2, eps, weight_decay, bc1, bc2};
  dispatch(c, c.special_ops, buffers, 4, &push, sizeof(push), ceil_div(count, 64));
}

struct AttentionPush {
  std::uint32_t bh;
  std::uint32_t seq_len;
  std::uint32_t head_dim;
  std::uint32_t causal;
  float scale;
};

void VulkanDriver::attention(DeviceRef device, const float* q, const float* k, const float* v, float* out,
                             int batch_heads, int seq_len, int head_dim, bool causal) {
  if (batch_heads <= 0 || seq_len <= 0 || head_dim <= 0 || head_dim > 128) {
    throw TensorError("Nexus Vulkan: gecersiz dikkat boyutu (head_dim <= 128 olmali)");
  }
  Context& c = context(device.index);
  const std::uint32_t total_rows = static_cast<std::uint32_t>(batch_heads) * static_cast<std::uint32_t>(seq_len);
  const Buffer* buffers[] = {
      static_cast<const Buffer*>(static_cast<const void*>(q)),
      static_cast<const Buffer*>(static_cast<const void*>(k)),
      static_cast<const Buffer*>(static_cast<const void*>(v)),
      static_cast<const Buffer*>(static_cast<const void*>(out)),
  };
  AttentionPush push{static_cast<std::uint32_t>(batch_heads), static_cast<std::uint32_t>(seq_len),
                     static_cast<std::uint32_t>(head_dim), causal ? 1u : 0u,
                     1.0f / std::sqrt(static_cast<float>(head_dim))};
  dispatch(c, c.attention, buffers, 4, &push, sizeof(push), ceil_div(total_rows, 64));
}

struct QuantMatMulPush {
  std::uint32_t m;
  std::uint32_t n;
  std::uint32_t k;
  std::uint32_t group_size;
  std::uint32_t op;
};

void VulkanDriver::quant_matmul_q8(DeviceRef device, const float* a, const void* w_q8,
                                   const float* w_scales, float* c, int m, int n, int k, int group_size) {
  if (m <= 0 || n <= 0 || k <= 0 || group_size <= 0 || (k % group_size) != 0) {
    throw TensorError("Nexus Vulkan: gecersiz Q8 GEMM boyutu");
  }
  Context& ctx = context(device.index);
  const Buffer* buffers[] = {
      static_cast<const Buffer*>(static_cast<const void*>(a)),
      static_cast<const Buffer*>(w_q8),
      static_cast<const Buffer*>(static_cast<const void*>(w_scales)),
      static_cast<const Buffer*>(static_cast<const void*>(c)),
  };
  QuantMatMulPush push{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(n),
                       static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(group_size), 0u};
  dispatch(ctx, ctx.quant_matmul, buffers, 4, &push, sizeof(push), ceil_div(n, 16), ceil_div(m, 16));
}

void VulkanDriver::quant_matmul_q4(DeviceRef device, const float* a, const void* w_q4,
                                   const float* w_scales, float* c, int m, int n, int k, int group_size) {
  if (m <= 0 || n <= 0 || k <= 0 || group_size <= 0 || (k % group_size) != 0) {
    throw TensorError("Nexus Vulkan: gecersiz Q4 GEMM boyutu");
  }
  Context& ctx = context(device.index);
  const Buffer* buffers[] = {
      static_cast<const Buffer*>(static_cast<const void*>(a)),
      static_cast<const Buffer*>(w_q4),
      static_cast<const Buffer*>(static_cast<const void*>(w_scales)),
      static_cast<const Buffer*>(static_cast<const void*>(c)),
  };
  QuantMatMulPush push{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(n),
                       static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(group_size), 1u};
  dispatch(ctx, ctx.quant_matmul, buffers, 4, &push, sizeof(push), ceil_div(n, 16), ceil_div(m, 16));
}

}  // namespace

std::unique_ptr<Driver> make_vulkan_driver() { return std::make_unique<VulkanDriver>(); }

}  // namespace nexus::backend

#else
namespace nexus::backend {
namespace {
class DisabledVulkan final : public Driver {
 public:
  Kind kind() const noexcept override { return Kind::Vulkan; }
  std::string_view name() const noexcept override { return "vulkan-spirv"; }
  std::string_view version() const noexcept override { return "0"; }
  const std::vector<DeviceInfo>& devices() const noexcept override { return devices_; }
  std::string unavailable_reason() const override { return "Vulkan basliklari olmadan derlendi"; }
  bool supports(Operation) const noexcept override { return false; }
  void* allocate(int, std::size_t) override { fail(); }
  void deallocate(int, void*) noexcept override {}
  void copy(DeviceRef, void*, DeviceRef, const void*, std::size_t) override { fail(); }
  void synchronize(int) override { fail(); }
  void elementwise(Operation, DeviceRef, const float*, const float*, float*, std::size_t, float) override { fail(); }
  void matmul(DeviceRef, const MatMulArgs&) override { fail(); }
  void reduce_sum(DeviceRef, const float*, float*, std::size_t) override { fail(); }
 private:
  [[noreturn]] void fail() const { throw TensorError("Nexus: Vulkan bu derlemede etkin degil"); }
  std::vector<DeviceInfo> devices_;
};
}  // namespace
std::unique_ptr<Driver> make_vulkan_driver() { return std::make_unique<DisabledVulkan>(); }
}  // namespace nexus::backend
#endif
