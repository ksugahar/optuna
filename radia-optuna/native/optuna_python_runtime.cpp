#include "optuna_runtime.hpp"
#ifdef OPTUNA_HOST_ONLY
#include <stdexcept>

namespace optuna_cuda {
namespace {
// CUDA-free build: no driver is loaded and every allocation request fails.
[[noreturn]] void unavailable() {
    throw std::runtime_error("Optuna native build has no CUDA backend; use the cpu backend");
}
void initialize() { unavailable(); }
void* allocate(std::size_t, void**) { unavailable(); }
bool valid(void*) noexcept { return false; }
void release(void*) noexcept {}
}
RuntimeHooks cuda_runtime_hooks() { return {initialize, allocate, valid, release}; }
}
#else
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <memory>
#include <stdexcept>

namespace optuna_cuda {
namespace {
struct Allocation { void* data = nullptr; unsigned long long context_id = 0; };
using GetContext = CUresult (CUDAAPI*)(CUcontext*);
using GetId = CUresult (CUDAAPI*)(CUcontext,unsigned long long*);
#ifdef _WIN32
HMODULE driver = nullptr;
#else
void* driver = nullptr;
#endif
// The driver handle is retained for process lifetime and loaded only on GPU use.
GetContext get_context = nullptr;
GetId get_id = nullptr;
unsigned long long context_id() noexcept {
    CUcontext context = nullptr;
    unsigned long long id = 0;
    if (!get_context || !get_id || get_context(&context) != CUDA_SUCCESS || !context ||
        get_id(context, &id) != CUDA_SUCCESS) return 0;
    return id;
}
void initialize() {
    if (!driver) {
#ifdef _WIN32
        driver=LoadLibraryExW(L"nvcuda.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!driver) throw std::runtime_error("Optuna GPU requires the NVIDIA system driver");
        get_context=reinterpret_cast<GetContext>(GetProcAddress(driver,"cuCtxGetCurrent"));
        get_id=reinterpret_cast<GetId>(GetProcAddress(driver,"cuCtxGetId"));
#else
        driver=dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!driver) throw std::runtime_error("Optuna GPU requires the NVIDIA system driver");
        get_context=reinterpret_cast<GetContext>(dlsym(driver,"cuCtxGetCurrent"));
        get_id=reinterpret_cast<GetId>(dlsym(driver,"cuCtxGetId"));
#endif
    }
    if (!get_context || !get_id)
        throw std::runtime_error("Optuna GPU driver lacks context lifetime identity API");
    if (cudaFree(nullptr) != cudaSuccess || !context_id())
        throw std::runtime_error("Optuna explicit GPU context initialization failed");
}
bool valid(void* owner) noexcept {
    const auto* allocation = static_cast<Allocation*>(owner);
    return allocation && allocation->context_id && allocation->context_id == context_id();
}
void* allocate(std::size_t bytes, void** owner) {
    auto allocation = std::make_unique<Allocation>();
    allocation->context_id = context_id();
    if (!allocation->context_id || cudaMalloc(&allocation->data, bytes) != cudaSuccess)
        throw std::runtime_error("Optuna CUDA workspace allocation failed");
    void* result = allocation->data;
    *owner = allocation.release();
    return result;
}
void release(void* owner) noexcept {
    auto* allocation = static_cast<Allocation*>(owner);
    // Device reset already freed the allocation; never free a reused address.
    if (valid(owner)) cudaFree(allocation->data);
    delete allocation;
}
}
RuntimeHooks cuda_runtime_hooks() { return {initialize, allocate, valid, release}; }
}
#endif
