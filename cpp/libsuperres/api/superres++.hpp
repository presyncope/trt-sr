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

  bool is_initialized() const { return m_initialized; }

  void init(const sr_init_params &params) {
    if (m_initialized)
      m_handle.reset(sr_create());

    if (sr_init(m_handle.get(), &params) != 0) {
      throw std::runtime_error("Failed to initialize SuperRes context.");
    }
    m_initialized = true;
  }

  void process(const sr_frame &src, sr_frame &dst) {
    if (sr_process(m_handle.get(), &src, &dst) != 0) {
      throw std::runtime_error("Failed to process frame in SuperRes.");
    }
  }

  static void build(const sr_build_params &params) {
    if (sr_build(&params) != 0) {
      throw std::runtime_error("Failed to build TensorRT plan from ONNX.");
    }
  }

  sr_handle get() const { return m_handle.get(); }

private:
  std::unique_ptr<sr_context, HandleDeleter> m_handle = {};
  bool m_initialized = false;
};

} // namespace sr
