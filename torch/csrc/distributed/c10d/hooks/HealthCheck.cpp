// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/hooks/HealthCheck.hpp>

#include <utility>

namespace c10d {

HealthCheck* HealthCheck::get() {
  // NOLINTNEXTLINE(facebook-hte-InlinedStaticLocalVariableWarning)
  static auto* instance = new HealthCheck();
  return instance;
}

void HealthCheck::setUnhealthy(std::string backend) {
  std::lock_guard<std::mutex> lock(mutex_);
  unhealthy_backends_.insert(std::move(backend));
}

std::vector<std::string> HealthCheck::unhealthyBackends() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {unhealthy_backends_.begin(), unhealthy_backends_.end()};
}

} // namespace c10d
