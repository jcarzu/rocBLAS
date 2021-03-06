/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_trsv.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_gemv.hpp"
#include "utility.h"
#include <algorithm>
#include <cstdio>
#include <tuple>

namespace
{
    constexpr rocblas_int STRSV_BLOCK = 128;
    constexpr rocblas_int DTRSV_BLOCK = 128;

    template <typename>
    constexpr char rocblas_trsv_name[] = "unknown";
    template <>
    constexpr char rocblas_trsv_name<float>[] = "rocblas_strsv";
    template <>
    constexpr char rocblas_trsv_name<double>[] = "rocblas_dtrsv";

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsv_ex_impl(rocblas_handle    handle,
                                        rocblas_fill      uplo,
                                        rocblas_operation transA,
                                        rocblas_diagonal  diag,
                                        rocblas_int       m,
                                        const T*          A,
                                        rocblas_int       lda,
                                        T*                B,
                                        rocblas_int       incx,
                                        const T*          supplied_invA      = nullptr,
                                        rocblas_int       supplied_invA_size = 0)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_trsv_name<T>, uplo, transA, diag, m, A, lda, B, incx);

        if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);
            auto diag_letter   = rocblas_diag_letter(diag);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                if(handle->pointer_mode == rocblas_pointer_mode_host)
                    log_bench(handle,
                              "./rocblas-bench -f trsv -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "--transposeA",
                              transA_letter,
                              "--diag",
                              diag_letter,
                              "-m",
                              m,
                              "--lda",
                              lda,
                              "--incx",
                              incx);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_trsv_name<T>,
                            "uplo",
                            uplo_letter,
                            "transA",
                            transA_letter,
                            "diag",
                            diag_letter,
                            "M",
                            m,
                            "lda",
                            lda,
                            "incx",
                            incx);
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(!A || !B)
            return rocblas_status_invalid_pointer;
        if(m < 0 || lda < m || lda < 1 || !incx)
            return rocblas_status_invalid_size;

        // quick return if possible.
        // return rocblas_status_size_unchanged if device memory size query
        if(!m)
            return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                         : rocblas_status_success;

        void* mem_x_temp;
        void* mem_x_temp_arr;
        void* mem_invA;
        void* mem_invA_arr;

        rocblas_status status = rocblas_trsv_template_mem<BLOCK, false, T>(handle,
                                                                           m,
                                                                           1,
                                                                           &mem_x_temp,
                                                                           &mem_x_temp_arr,
                                                                           &mem_invA,
                                                                           &mem_invA_arr,
                                                                           supplied_invA,
                                                                           supplied_invA_size);

        rocblas_status status2 = rocblas_trsv_template<BLOCK, false, T>(handle,
                                                                        uplo,
                                                                        transA,
                                                                        diag,
                                                                        m,
                                                                        A,
                                                                        0,
                                                                        lda,
                                                                        0,
                                                                        B,
                                                                        0,
                                                                        incx,
                                                                        0,
                                                                        1,
                                                                        mem_x_temp,
                                                                        mem_x_temp_arr,
                                                                        mem_invA,
                                                                        mem_invA_arr,
                                                                        supplied_invA,
                                                                        supplied_invA_size);

        return (status2 == rocblas_status_success) ? status : status2;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strsv(rocblas_handle    handle,
                             rocblas_fill      uplo,
                             rocblas_operation transA,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             const float*      A,
                             rocblas_int       lda,
                             float*            x,
                             rocblas_int       incx)
{
    return rocblas_trsv_ex_impl<STRSV_BLOCK>(handle, uplo, transA, diag, m, A, lda, x, incx);
}

rocblas_status rocblas_dtrsv(rocblas_handle    handle,
                             rocblas_fill      uplo,
                             rocblas_operation transA,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             const double*     A,
                             rocblas_int       lda,
                             double*           x,
                             rocblas_int       incx)
{
    return rocblas_trsv_ex_impl<DTRSV_BLOCK>(handle, uplo, transA, diag, m, A, lda, x, incx);
}

rocblas_status rocblas_trsv_ex(rocblas_handle    handle,
                               rocblas_fill      uplo,
                               rocblas_operation transA,
                               rocblas_diagonal  diag,
                               rocblas_int       m,
                               const void*       A,
                               rocblas_int       lda,
                               void*             x,
                               rocblas_int       incx,
                               const void*       invA,
                               rocblas_int       invA_size,
                               rocblas_datatype  compute_type)

{
    switch(compute_type)
    {
    case rocblas_datatype_f64_r:
        return rocblas_trsv_ex_impl<DTRSV_BLOCK>(handle,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 m,
                                                 static_cast<const double*>(A),
                                                 lda,
                                                 static_cast<double*>(x),
                                                 incx,
                                                 static_cast<const double*>(invA),
                                                 invA_size);

    case rocblas_datatype_f32_r:
        return rocblas_trsv_ex_impl<STRSV_BLOCK>(handle,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 m,
                                                 static_cast<const float*>(A),
                                                 lda,
                                                 static_cast<float*>(x),
                                                 incx,
                                                 static_cast<const float*>(invA),
                                                 invA_size);

    default:
        return rocblas_status_not_implemented;
    }
}

} // extern "C"
