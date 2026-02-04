#pragma once

#include "superres.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace sr {

struct HandleDeleter {
  void operator()(sr_handle handle) const {
    if (handle) {
      sr_destroy(handle);
    }
  }
};

class SuperRes {
public:
  SuperRes() {
    sr_handle h = sr_create();
    if (!h) {
      throw std::runtime_error("Failed to create SuperRes handle.");
    }
    m_handle.reset(h);
  }

  void init(const sr_init_params &params) {
    if (sr_init(m_handle.get(), &params) != 0) {
      throw std::runtime_error("Failed to initialize SuperRes context.");
    }
  }

  void process(const sr_frame &src, sr_frame &dst) {
    if (sr_process(m_handle.get(), &src, &dst) != 0) {
      throw std::runtime_error("Failed to process frame in SuperRes.");
    }
  }

  static void build(const std::string &onnx_path,
                    const std::string &plan_path) {
#if SR_ENABLE_NVONNXPARSER
    if (sr_build(onnx_path.c_str(), plan_path.c_str()) != 0) {
      throw std::runtime_error("Failed to build TensorRT plan from ONNX.");
    }
#endif
  }

  sr_handle get() const { return m_handle.get(); }

private:
  std::unique_ptr<sr_context, HandleDeleter> m_handle;
};

} // namespace sr
