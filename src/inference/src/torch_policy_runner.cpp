#include "torch_policy_runner.hpp"

#include <torch/script.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>

namespace inference {
namespace {

bool tensor_is_valid_policy_output(const torch::Tensor& tensor,
                                   std::size_t expected_count,
                                   std::string& error)
{
    if (tensor.scalar_type() != torch::kFloat32) {
        std::ostringstream oss;
        oss << "TorchScript policy output must be float32, got scalar_type="
            << tensor.scalar_type();
        error = oss.str();
        return false;
    }
    if (tensor.numel() != static_cast<int64_t>(expected_count)) {
        std::ostringstream oss;
        oss << "TorchScript policy output size mismatch, expected "
            << expected_count << " got " << tensor.numel();
        error = oss.str();
        return false;
    }
    if (!tensor.device().is_cpu() || !tensor.is_contiguous()) {
        error = "TorchScript policy output must be CPU contiguous";
        return false;
    }
    if (!torch::isfinite(tensor).all().item<bool>()) {
        error = "TorchScript policy output contains NaN or Inf";
        return false;
    }
    return true;
}

}  // namespace

// LibTorch 类型只放在 .cpp 的 Impl 中，避免公共头文件暴露庞大的 torch 依赖。
struct TorchPolicyRunner::Impl {
    std::unique_ptr<torch::jit::script::Module> module;
};

TorchPolicyRunner::TorchPolicyRunner()
    : impl_(std::make_unique<Impl>())
{
}

TorchPolicyRunner::~TorchPolicyRunner()
{
    unload();
}

/* 加载模型 */
bool TorchPolicyRunner::load(const std::string& model_path)
{
    unload();
    last_error_.clear();

    // LibTorch 的 inter-op 线程数只能在进程早期设置一次；多个 runner/load 不重复设置。
    static std::once_flag torch_thread_init;
    std::call_once(torch_thread_init, [] {
        int num_threads = 1;
        if (const char* env = std::getenv("MYACTUA_TORCH_THREADS")) {
            const int parsed = std::atoi(env);
            if (parsed > 0 && parsed <= 8) {
                num_threads = parsed;
            }
        }
        torch::set_num_threads(num_threads);
        at::set_num_interop_threads(1);
        std::cout << "[TorchPolicyRunner] torch intra-op threads=" << num_threads
                  << " interop threads=1" << std::endl;
    });

    try {
        impl_->module = std::make_unique<torch::jit::script::Module>(
            torch::jit::load(model_path, torch::kCPU));
        impl_->module->eval();
        loaded_ = true;

        // 加载时 dry-run 一次，提前拒绝非 TorchScript 或输入/输出形状不匹配的模型。
        if (!dry_run_and_validate_output()) {
            unload();
            return false;
        }
    } catch (const c10::Error& e) {
        set_error("failed to load TorchScript model: " + std::string(e.what()));
        unload();
        return false;
    } catch (const std::exception& e) {
        set_error("failed to load TorchScript model: " + std::string(e.what()));
        unload();
        return false;
    }

    return true;
}

/* 检查模型与输入类型和维度是否正确 */
bool TorchPolicyRunner::dry_run_and_validate_output()
{
    if (!loaded_ || !impl_->module) {
        set_error("TorchScript policy is not loaded");
        return false;
    }

    try {
        torch::NoGradGuard no_grad;
        // 使用全零观测验证 forward 入口；这里不关心数值，只关心类型和维度。
        torch::Tensor input = torch::zeros(
            {1, static_cast<int64_t>(kInputSize)},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        torch::jit::IValue output_value = impl_->module->forward({input});
        if (!output_value.isTensor()) {
            set_error("TorchScript policy output must be a tensor");
            return false;
        }

        std::string output_error;
        if (!tensor_is_valid_policy_output(output_value.toTensor(),
                                           kOutputSize,
                                           output_error)) {
            set_error(output_error);
            return false;
        }
    } catch (const c10::Error& e) {
        set_error("TorchScript dry-run failed: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        set_error("TorchScript dry-run failed: " + std::string(e.what()));
        return false;
    }

    return true;
}

void TorchPolicyRunner::set_error(const std::string& message)
{
    last_error_ = message;
}

void TorchPolicyRunner::unload()
{
    if (impl_) {
        impl_->module.reset();
    }
    loaded_ = false;
}

bool TorchPolicyRunner::infer(const std::array<float, kInputSize>& observation,
                              std::array<float, kOutputSize>& action)
{
    if (!loaded_ || !impl_->module) {
        set_error("TorchScript policy is not loaded");
        return false;
    }

    try {
        if (!std::all_of(observation.begin(), observation.end(), [](float value) {
                return std::isfinite(value);
            })) {
            set_error("TorchScript policy input contains NaN or Inf");
            return false;
        }
        torch::NoGradGuard no_grad;
        // 模型训练/导出约定为单输入 float32 tensor，形状固定为 [1, kInputSize]。
        torch::Tensor input = torch::from_blob(
            const_cast<float*>(observation.data()),
            {1, static_cast<int64_t>(observation.size())},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        torch::jit::IValue output_value = impl_->module->forward({input});
        if (!output_value.isTensor()) {
            set_error("TorchScript policy output must be a tensor");
            return false;
        }

        torch::Tensor output = output_value.toTensor();
        std::string output_error;
        if (!tensor_is_valid_policy_output(output, action.size(), output_error)) {
            set_error(output_error);
            return false;
        }

        // 输出统一整理为 CPU contiguous float32，便于复制到固定 12 维动作数组。
        output = output.to(torch::kCPU).contiguous();
        const float* output_data = output.data_ptr<float>();
        std::copy(output_data, output_data + action.size(), action.begin());
    } catch (const c10::Error& e) {
        set_error("TorchScript inference failed: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        set_error("TorchScript inference failed: " + std::string(e.what()));
        return false;
    }

    return true;
}


}  // namespace inference
