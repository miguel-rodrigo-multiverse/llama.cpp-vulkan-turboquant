#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include "../ggml/src/ggml-turboquant-codec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using ggml_backend_turboquant_materialize_fn = bool (*)(
        ggml_backend_t backend,
        ggml_tensor * tensor,
        ggml_type type,
        uint32_t n_row_el,
        uint32_t kv_size,
        bool transposed,
        uint32_t group_size,
        uint32_t residual_bits,
        bool qjl,
        const uint8_t * packed_rows,
        size_t packed_row_size,
        const uint32_t * row_indices,
        size_t n_rows);

using ggml_backend_turboquant_compress_fn = bool (*)(
        ggml_backend_t backend,
        const ggml_tensor * src_tensor,
        ggml_tensor * dst_tensor,
        uint32_t n_row_el,
        uint32_t kv_size,
        bool transposed,
        uint32_t group_size,
        uint32_t residual_bits,
        bool qjl,
        size_t packed_row_size,
        const uint32_t * row_indices,
        size_t n_rows);

using ggml_backend_turboquant_get_packed_rows_fn = bool (*)(
        ggml_backend_t backend,
        const ggml_tensor * tensor,
        uint8_t * out_data,
        size_t size);

static void require(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "test-turboquant-backend: %s\n", msg);
        std::exit(1);
    }
}

static std::vector<float> make_row(uint32_t row_index, uint32_t n_row_el) {
    std::vector<float> row(n_row_el);
    for (uint32_t i = 0; i < n_row_el; ++i) {
        const float x = float(row_index * 17 + i);
        row[i] = 0.4f * std::sin(0.11f * x) + 0.7f * std::cos(0.05f * x);
    }
    return row;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size(), "size mismatch");
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::fabs(a[i] - b[i]));
    }
    return result;
}

static ggml_backend_dev_t find_vulkan_device() {
    ggml_backend_load_all();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        if (name != nullptr && std::strstr(name, "Vulkan") != nullptr) {
            return dev;
        }
    }

    return nullptr;
}

static ggml_tensor * make_backend_tensor(
        ggml_context * ctx,
        ggml_backend_buffer_type_t buft,
        ggml_backend_buffer_t & buffer,
        ggml_type type,
        int64_t ne0,
        int64_t ne1) {
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    require(tensor != nullptr, "failed to create tensor metadata");

    buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    require(buffer != nullptr, "failed to allocate backend tensor buffer");
    ggml_backend_buffer_clear(buffer, 0);

    return tensor;
}

static void test_backend_materialize(ggml_backend_t backend, ggml_backend_reg_t reg, bool transposed) {
    constexpr uint32_t kv_size = 8;
    constexpr uint32_t n_row_el = 16;
    const ggml_turboquant_codec_params params = {
        /*.group_size    =*/ 8,
        /*.residual_bits =*/ 2,
        /*.qjl           =*/ true,
    };

    auto * fn = (ggml_backend_turboquant_materialize_fn)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_turboquant_materialize");
    require(fn != nullptr, "Vulkan TurboQuant entry point is not exported");

    ggml_init_params init_params = {
        /*.mem_size   =*/ 16 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(init_params);
    require(ctx != nullptr, "failed to create ggml context");

    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = make_backend_tensor(
            ctx,
            ggml_backend_get_default_buffer_type(backend),
            buffer,
            GGML_TYPE_F32,
            transposed ? kv_size : n_row_el,
            transposed ? n_row_el : kv_size);

    const std::vector<uint32_t> row_indices = { 1, 3, 6 };
    const size_t packed_row_size = ggml_turboquant_row_size(n_row_el, params);
    std::vector<uint8_t> packed_rows(row_indices.size() * packed_row_size);
    std::vector<float> expected(kv_size * n_row_el, 0.0f);
    std::vector<float> unpacked_row(n_row_el, 0.0f);

    for (size_t i = 0; i < row_indices.size(); ++i) {
        const auto src_row = make_row(row_indices[i], n_row_el);
        std::vector<uint8_t> packed_row;
        ggml_turboquant_pack_row(src_row.data(), GGML_TYPE_F32, n_row_el, params, packed_row);
        require(packed_row.size() == packed_row_size, "packed row size mismatch");
        std::memcpy(packed_rows.data() + i * packed_row_size, packed_row.data(), packed_row_size);

        ggml_turboquant_unpack_row(
                packed_row.data(),
                GGML_TYPE_F32,
                n_row_el,
                params,
                unpacked_row.data());

        for (uint32_t j = 0; j < n_row_el; ++j) {
            if (transposed) {
                expected[row_indices[i] + j * kv_size] = unpacked_row[j];
            } else {
                expected[row_indices[i] * n_row_el + j] = unpacked_row[j];
            }
        }
    }

    require(fn(
            backend,
            tensor,
            GGML_TYPE_F32,
            n_row_el,
            kv_size,
            transposed,
            params.group_size,
            params.residual_bits,
            params.qjl,
            packed_rows.data(),
            packed_row_size,
            row_indices.data(),
            row_indices.size()), "backend materialize call failed");

    std::vector<float> actual(expected.size(), 0.0f);
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size() * sizeof(float));

    const float err = max_abs_diff(expected, actual);
    require(err < 1e-6f, transposed ? "transposed materialization mismatch" : "materialization mismatch");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void test_backend_compress(ggml_backend_t backend, ggml_backend_reg_t reg, bool transposed) {
    constexpr uint32_t kv_size = 8;
    constexpr uint32_t n_row_el = 16;
    const ggml_turboquant_codec_params params = {
        /*.group_size    =*/ 8,
        /*.residual_bits =*/ 2,
        /*.qjl           =*/ true,
    };

    auto * fn_compress = (ggml_backend_turboquant_compress_fn)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_turboquant_compress");
    auto * fn_get_packed = (ggml_backend_turboquant_get_packed_rows_fn)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_turboquant_get_packed_rows");
    require(fn_compress != nullptr, "Vulkan TurboQuant compress entry point is not exported");
    require(fn_get_packed != nullptr, "Vulkan TurboQuant get_packed_rows entry point is not exported");

    ggml_init_params init_params = {
        /*.mem_size   =*/ 32 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(init_params);
    require(ctx != nullptr, "failed to create ggml context");

    ggml_backend_buffer_t src_buffer = nullptr;
    ggml_tensor * src_tensor = make_backend_tensor(
            ctx,
            ggml_backend_get_default_buffer_type(backend),
            src_buffer,
            GGML_TYPE_F32,
            transposed ? kv_size : n_row_el,
            transposed ? n_row_el : kv_size);

    ggml_backend_buffer_t dst_buffer = nullptr;
    ggml_tensor * dst_tensor = make_backend_tensor(
            ctx,
            ggml_backend_get_default_buffer_type(backend),
            dst_buffer,
            GGML_TYPE_F32,
            transposed ? kv_size : n_row_el,
            transposed ? n_row_el : kv_size);

    // Initialize source tensor with signal
    std::vector<float> src_data(kv_size * n_row_el);
    for (uint32_t r = 0; r < kv_size; ++r) {
        const auto src_row = make_row(r, n_row_el);
        for (uint32_t c = 0; c < n_row_el; ++c) {
            if (transposed) {
                src_data[r + c * kv_size] = src_row[c];
            } else {
                src_data[r * n_row_el + c] = src_row[c];
            }
        }
    }
    ggml_backend_tensor_set(src_tensor, src_data.data(), 0, src_data.size() * sizeof(float));

    const std::vector<uint32_t> row_indices = { 1, 3, 6 };
    const size_t packed_row_size = ggml_turboquant_row_size(n_row_el, params);

    // Compute expected packed rows on CPU
    std::vector<uint8_t> expected_packed_all(kv_size * packed_row_size, 0);
    for (uint32_t r : row_indices) {
        const auto src_row = make_row(r, n_row_el);
        std::vector<uint8_t> packed_row;
        ggml_turboquant_pack_row(src_row.data(), GGML_TYPE_F32, n_row_el, params, packed_row);
        require(packed_row.size() == packed_row_size, "packed row size mismatch");
        std::memcpy(expected_packed_all.data() + r * packed_row_size, packed_row.data(), packed_row_size);
    }

    // Run GPU compression
    require(fn_compress(
            backend,
            src_tensor,
            dst_tensor,
            n_row_el,
            kv_size,
            transposed,
            params.group_size,
            params.residual_bits,
            params.qjl,
            packed_row_size,
            row_indices.data(),
            row_indices.size()), "backend compress call failed");

    // Read back compressed data
    std::vector<uint8_t> actual_packed_all(kv_size * packed_row_size, 0);
    require(fn_get_packed(
            backend,
            dst_tensor,
            actual_packed_all.data(),
            actual_packed_all.size()), "failed to read back packed rows");

    // Verify compressed rows
    for (uint32_t r : row_indices) {
        const uint8_t * exp_ptr = expected_packed_all.data() + r * packed_row_size;
        const uint8_t * act_ptr = actual_packed_all.data() + r * packed_row_size;
        for (size_t b = 0; b < packed_row_size; ++b) {
            if (exp_ptr[b] != act_ptr[b]) {
                std::fprintf(stderr, "test-turboquant-backend: row %u mismatch at byte %zu: expected 0x%02X, got 0x%02X\n",
                             r, b, exp_ptr[b], act_ptr[b]);
                require(false, "compressed byte mismatch");
            }
        }
    }

    ggml_backend_buffer_free(src_buffer);
    ggml_backend_buffer_free(dst_buffer);
    ggml_free(ctx);
}

int main() {
    ggml_backend_dev_t dev = find_vulkan_device();
    if (dev == nullptr) {
        std::printf("test-turboquant-backend: SKIP (no Vulkan backend device available)\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    require(backend != nullptr, "failed to initialize Vulkan backend");

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    require(reg != nullptr, "failed to resolve backend registry");

    test_backend_materialize(backend, reg, false);
    test_backend_materialize(backend, reg, true);

    test_backend_compress(backend, reg, false);
    test_backend_compress(backend, reg, true);

    ggml_backend_free(backend);

    std::printf("test-turboquant-backend: OK\n");
    return 0;
}
