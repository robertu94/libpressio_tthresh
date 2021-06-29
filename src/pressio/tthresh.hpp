/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

#ifndef __TTHRESH_HPP__
#define __TTHRESH_HPP__

#include <vector>
#include <chrono>
#include <string>
using namespace std;
using namespace std::chrono;

// Size (in bytes) for all I/O buffers
#define CHUNK (1<<18)

// Rows whose squared norm is larger than this will be cropped away
#define AUTOCROP_THRESHOLD (1e-10)

// Compression parameters
enum Mode { none_mode, input_mode, compressed_mode, output_mode, io_type_mode, sizes_mode, target_mode, skip_bytes_mode };
enum Target { eps, rmse, psnr };

// Tensor dimensionality, ranks and sizes. They are only set, never modified
uint8_t n;
vector<uint32_t> r;
vector<size_t> rprod;
vector<uint32_t> s, snew;
vector<size_t> sprod, snewprod;

void cumulative_products(vector<uint32_t>& in, vector<size_t>& out) {
    uint8_t n = s.size();
    out = vector<size_t> (n+1); // Cumulative size products. The i-th element contains s[0]*...*s[i-1]
    out[0] = 1;
    for (uint8_t i = 0; i < n; ++i)
        out[i+1] = out[i]*in[i];
}

#endif // TTHRESH_HPP
