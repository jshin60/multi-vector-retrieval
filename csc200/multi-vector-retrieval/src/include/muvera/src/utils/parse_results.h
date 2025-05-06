#ifndef ALGORITHMS_UTILS_PARSE_RESULTS_H_
#define ALGORITHMS_UTILS_PARSE_RESULTS_H_

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

#include <algorithm>
#include <set>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

struct Graph_ {
    std::string name;
    std::string params;
    long size;
    double avg_deg;
    int max_deg;
    double time;

    Graph_(std::string n, std::string p, long s, double ad, int md, double t)
            : name(n), params(p), size(s), avg_deg(ad), max_deg(md), time(t) {}

    void print() {
        std::cout << name << " graph built with " << size
                  << " points and parameters " << params << std::endl;
        std::cout << "Graph has average degree " << avg_deg
                  << " and maximum degree " << max_deg << std::endl;
        std::cout << "Graph built in " << time << " seconds" << std::endl;
    }
};


#endif  // ALGORITHMS_UTILS_PARSE_RESULTS_H_
