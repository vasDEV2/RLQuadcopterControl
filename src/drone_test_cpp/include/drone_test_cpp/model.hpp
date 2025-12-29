#pragma once

#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Dense>
#include <onnxruntime_cxx_api.h>

class ModelONNX
{
public:
  // Constructor
  explicit ModelONNX(const std::string &model_path,
                     int output_size);

  // Inference
  Eigen::VectorXf predict(const Eigen::VectorXf &obs);

private:
  // ONNX Runtime core objects
  Ort::Env env_;
  Ort::Session session_{nullptr};
  Ort::SessionOptions session_options_;

  // IO metadata
  std::string input_name_;
  std::string output_name_;
  int output_size_;
};
