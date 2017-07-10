#pragma once

#include "THC/THC.h"

namespace torch { namespace rtc {

void launch(const char* ptx, const char* name, void* args[], dim3 grid, dim3 block, CUstream stream);

void compilePTX(const char* src,
        const char* headers[],
    const char* includeNames[],
    std::vector<char>& ptx);

}} // namespace torch::rtc
