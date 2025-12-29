#include "drone_test_cpp/model.hpp"

ModelONNX::ModelONNX(const std::string &model_path,
                     int output_size)
: env_(ORT_LOGGING_LEVEL_WARNING, "onnx_runtime"),
  output_size_(output_size)
{
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(
    GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  session_ = Ort::Session(env_, model_path.c_str(), session_options_);

  // Get input / output names
  Ort::AllocatorWithDefaultOptions allocator;

  input_name_ =
    session_.GetInputNameAllocated(0, allocator).get();

  output_name_ =
    session_.GetOutputNameAllocated(0, allocator).get();
}

Eigen::VectorXf ModelONNX::predict(const Eigen::VectorXf& obs)
{
  // 1. Convert double -> float
  Eigen::VectorXf obs_f = obs.cast<float>();

  std::array<int64_t, 2> input_shape{1, obs_f.size()};

  Ort::MemoryInfo mem_info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    mem_info,
    obs_f.data(),
    obs_f.size(),
    input_shape.data(),
    input_shape.size()
  );

  // 2. Names MUST be const char*
  const char* input_names[]  = { input_name_.c_str() };
  const char* output_names[] = { output_name_.c_str() };

  // 3. Correct Run() call
  auto output_tensors = session_.Run(
    Ort::RunOptions{nullptr},
    input_names,
    &input_tensor,
    1,
    output_names,
    1
  );

  // 4. Extract output
  float* out_data =
    output_tensors[0].GetTensorMutableData<float>();

  size_t out_size =
    output_tensors[0].GetTensorTypeAndShapeInfo()
      .GetElementCount();

  return Eigen::Map<Eigen::VectorXf>(out_data, out_size);
}

