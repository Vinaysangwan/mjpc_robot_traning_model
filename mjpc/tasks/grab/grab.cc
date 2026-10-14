#include "mjpc/tasks/grab/grab.h"

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {

std::string Grab::XmlPath() const { return GetModelPath("grab/task.xml"); }
std::string Grab::Name() const { return "Grab"; }

void Grab::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                double* residual) const {
  int counter = 0;

  // 1) left tip -> left face of the cuboid
  double* tipL = SensorByName(model, data, "tipL_pos");
  double* faceL = SensorByName(model, data, "faceL_pos");
  mju_sub3(residual + counter, tipL, faceL);
  counter += 3;

  // 2) right tip -> right face of the cuboid
  double* tipR = SensorByName(model, data, "tipR_pos");
  double* faceR = SensorByName(model, data, "faceR_pos");
  mju_sub3(residual + counter, tipR, faceR);
  counter += 3;

  // 3) effort: command minus current joint angle (= servo torque / kp), so holding still is free.
  //    Joint 0 (BodySync) is a velocity command, so its target is simply 0.
  //    Assumes actuator i drives joint i (true for this model, nq = nv = nu = 7).
  residual[counter] = data->ctrl[0];
  for (int i = 1; i < model->nu; i++) {
    residual[counter + i] = data->ctrl[i] - data->qpos[i];
  }
  counter += model->nu;

  // 4) joint velocities: damps shaking, especially once the cuboid is grabbed
  for (int i = 0; i < model->nv; i++) {
    residual[counter + i] = data->qvel[i];
  }
  counter += model->nv;

  CheckSensorDim(model, counter);
}

// The cuboid is the only mocap body, so it is mocap index 0.
// It travels along a straight line, but its "progress" only advances while the arms have NOT
// grabbed it. Progress lives in d->userdata (needs <size nuserdata="2"/> in task.xml), which is
// copied into planner rollouts, so the planner also sees the cube stop when the arms close.
//   userdata[0] = seconds of travel so far
//   userdata[1] = last d->time this function ran (makes repeated calls harmless)
void Grab::TransitionLocked(mjModel* model, mjData* d) {
  if (model->nmocap < 1 || model->nuserdata < 2) return;

  double dt = d->time - d->userdata[1];
  d->userdata[1] = d->time;

  // grabbed = both tips within a threshold distance of their cuboid faces
  bool grabbed = false;
  if (d->time > 0.1) {  // sensors are not valid before the first step
    double thr = GetNumberOrDefault(0.03, model, "grab_threshold");
    double eL[3], eR[3];
    mju_sub3(eL, SensorByName(model, d, "tipL_pos"),
             SensorByName(model, d, "faceL_pos"));
    mju_sub3(eR, SensorByName(model, d, "tipR_pos"),
             SensorByName(model, d, "faceR_pos"));
    grabbed = mju_norm3(eL) < thr && mju_norm3(eR) < thr;
  }

  if (!grabbed && dt > 0) d->userdata[0] += dt;
  double s = d->userdata[0];

  d->mocap_pos[0] = GetNumberOrDefault(0.0, model, "cuboid_x0") +
                    GetNumberOrDefault(0.0, model, "cuboid_vx") * s;
  d->mocap_pos[1] = GetNumberOrDefault(-0.3, model, "cuboid_y0") +
                    GetNumberOrDefault(0.1, model, "cuboid_vy") * s;
  d->mocap_pos[2] = GetNumberOrDefault(0.3, model, "cuboid_z0") +
                    GetNumberOrDefault(0.0, model, "cuboid_vz") * s;
}

}  // namespace mjpc
