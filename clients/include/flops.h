/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#pragma once
#ifndef _ROCBLAS_FLOPS_H_
#define _ROCBLAS_FLOPS_H_

#include "rocblas.h"
#include <typeinfo>

/*!\file
 * \brief provides Floating point counts of Basic Linear Algebra Subprograms (BLAS) of Level 2 and 3. No flops counts for Level 1 BLAS
*/

    /*
     * ===========================================================================
     *    level 2 BLAS
     * ===========================================================================
     */

    /* \brief floating point counts of GEMV */
    template<typename T>
    double  gemv_gflops(rocblas_int m, rocblas_int n){
        return (2.0 * m * n)/1e9;
    }

    /* \brief floating point counts of SY(HE)MV */
    template<typename T>
    double  symv_gflops(rocblas_int n){
        return (2.0 * n * n)/1e9;
    }

    /*
     * ===========================================================================
     *    level 3 BLAS
     * ===========================================================================
     */


    /* \brief floating point counts of GEMM */
    template<typename T>
    double  gemm_gflops(rocblas_int m, rocblas_int n, rocblas_int k){
        return (2.0 * m * n * k)/1e9;
    }

#endif  /* _ROCBLAS_FLOPS_H_ */
