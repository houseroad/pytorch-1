#include <nvrtc.h>
#include <vector>

#include "compile_ptx.h"

inline void NVRTC_CHECK(nvrtcResult result)
{
  if(result != NVRTC_SUCCESS)
    THError(nvrtcGetErrorString(result));
}

void compilePTX(const char* src,
                const char* headers[],
                const char* includeNames[],
                std::vector<char>& ptx)
{
  nvrtcProgram program;
  NVRTC_CHECK(nvrtcCreateProgram(&program, src, NULL, 1, headers, includeNames));

  nvrtcResult result = nvrtcCompileProgram(program, 0, NULL); 
  if(result == NVRTC_ERROR_COMPILATION)
  {
    size_t logsize;
    nvrtcGetProgramLogSize(program, &logsize);

    std::vector<char> log(logsize);
    nvrtcGetProgramLog(program, log.data());
    THError(log.data());
  }
  else
    NVRTC_CHECK(result);

  size_t ptx_size;
  NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
  ptx.resize(ptx_size);
  NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));
  NVRTC_CHECK(nvrtcDestroyProgram(&program));
}

inline void CUDA_CHECK(CUresult result)
{
  if(result != CUDA_SUCCESS)
  {
    const char* errstr;
    cuGetErrorString(result, &errstr);
    THError(errstr);
  }
}

void launch(const char* ptx, const char* name, void* args[], dim3 grid, dim3 block, CUstream stream)
{
  CUmodule module;
  CUfunction func;

  CUDA_CHECK(cuModuleLoadData(&module, ptx));
  CUDA_CHECK(cuModuleGetFunction(&func, module, name));

  CUDA_CHECK(cuLaunchKernel(func,
                            grid.x, grid.y, grid.z,
                            block.x, block.y, block.z,
                            0, stream, args, NULL));

  CUDA_CHECK(cuModuleUnload(module));
}

/**
 * @param state Torch CUDA state
 * @param ptx   Pointer to buffer containing compiled ptx (from compilePTX)
 * @param name  Name of the function in the PTX to run
 * @param args  Arguments to pass to the function.
 * @param grid  Grid dimensions to run with
 * @param block Block dimensions to run with
 */
// NB: don't call this directly, use THC_pointwiseApply{1,2,3} instead.
extern "C"
void launchPTX(THCState* state, const char* ptx, const char* name, void* args[], int* grid, int* block)
{
  cudaStream_t stream = THCState_getCurrentStream(state);
  launch(ptx, name, args, dim3(grid[0], grid[1], grid[2]), dim3(block[0], block[1], block[2]), (CUstream)stream);
}
