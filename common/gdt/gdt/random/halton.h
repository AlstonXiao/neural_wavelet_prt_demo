// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

/* pieces originally taken from optixPathTracer/random.h example,
   under following license */

/* 
 * Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "gdt/gdt.h"
#include "gdt/math/vec.h"

namespace gdt {

  /*! simple 24-bit linear congruence generator */
  template<unsigned int N=16>
  struct halton {
    
    inline __both__ halton()
    { /* intentionally empty so we can use it in device vars that
         don't allow dynamic initialization (ie, PRD) */
      init(2,3, N);
    }
    inline __both__ halton(unsigned int val0, unsigned int val1)
    { init(val0,val1, N); }
    
    inline __both__ void init(unsigned int base0, unsigned int base1, unsigned int iterations)
    {
      n0 = n1 = 0.0f;
      d0 = d1 = 1.0f;

      b0 = base0; // should be base 2
      b1 = base1; // should be base 3

      for (unsigned int n = 0; n < iterations; n++) {
        auto ret = operator()();
      }
    }
    
    // Generate random unsigned int in [0, 2^24)
    inline __both__ vec2f operator() ()
    {
      int x0 = d0 - n0;
      if (x0 == 1) {
        n0 = 1;
        d0 *= b0;
      } else {
        int y0 = d0 / b0;
        while (x0 <= y0) {
          y0 /= b0;
        }
        n0 = (b0 + 1) * y0 - x0;
      }
      state_x = (double)n0 / d0;

      int x1 = d1 - n1;
      if (x1 == 1) {
        n1 = 1;
        d1 *= b1;
      } else {
        int y1 = d1 / b1;
        while (x1 <= y1) {
          y1 /= b1;
        }
        n1 = (b1 + 1) * y1 - x1;
      }
      state_y = (double)n1 / d1;

      return vec2f(state_x, state_y);
    }
    
    double state_x, state_y;
    uint32_t n0, n1, d0, d1, b0, b1;
  };

} // ::gdt
