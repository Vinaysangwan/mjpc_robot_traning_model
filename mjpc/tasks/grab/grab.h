#ifndef MJPC_TASKS_GRAB_GRAB_H_
#define MJPC_TASKS_GRAB_GRAB_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {

// Two-arm robot that squeezes a cuboid moving in a straight line.
class Grab : public Task {
 public:
  Grab() : residual_(this) {}

  std::string Name() const override;
  std::string XmlPath() const override;

  // residual = [tipL - faceL (3), tipR - faceR (3), ctrl - home_ctrl (7)]
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Grab* task) : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  // Moves the cuboid (mocap body) along a straight line as a function of time.
  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_GRAB_GRAB_H_
