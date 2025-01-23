// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TYPES
#define TYPES

#include <algorithm>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

struct BuildParams {
    long L; //vamana
    long R; //vamana
    double alpha; //vamana
    int num_passes; //vamana
    int single_batch; //vamana

    bool verbose;

    std::string alg_type;

    BuildParams(long R, long L, double a, int num_passes, int single_batch = 0,
                bool verbose = false)
            : R(R), L(L), alpha(a), num_passes(num_passes), single_batch(single_batch),
              verbose(verbose) {
        assert(R != 0 && L != 0 && alpha != 0);
        alg_type = "Vamana";
    }

    BuildParams() {}

    BuildParams(long R, long L, double a, int num_passes, bool verbose = false)
            : R(R), L(L), alpha(a), num_passes(num_passes), single_batch(0), verbose(verbose) { alg_type = "Vamana"; }

    long max_degree() {
        return R;
    }
};


struct QueryParams {
    long k;
    long beamSize;
    double cut;
    long limit;
    long degree_limit;

    QueryParams(long k, long beamSize, double cut, long limit, long dg) : k(k), beamSize(beamSize), cut(cut),
                                                                          limit(limit),
                                                                          degree_limit(dg) {}

    QueryParams() {}

};

#endif
