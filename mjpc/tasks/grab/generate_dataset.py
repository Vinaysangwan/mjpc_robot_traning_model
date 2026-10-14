"""Run MJPC as the expert and log (state, action, cuboid) into a .npz training file.

Usage (venv with mujoco + mujoco_mpc active):
  python3 generate_dataset.py --task_xml ~/coding/cpp/projects/mujoco_mpc/mjpc/tasks/grab/task.xml \
      --episodes 20 --seconds 8

The cube motion and the "grabbed -> stop" rule below mirror Grab::TransitionLocked in grab.cc.
Planner weights/noise come from task.xml, so copy your tuned GUI values into it first.
"""
import argparse

import mujoco
import numpy as np
from mujoco_mpc import agent as agent_lib

p = argparse.ArgumentParser()
p.add_argument("--task_xml", required=True)
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--seconds", type=float, default=8.0)
p.add_argument("--planner_steps", type=int, default=5,
               help="planner iterations per control step (more = better actions, slower)")
p.add_argument("--rand_x", type=float, default=0.02, help="+- random shift of cube start x (m)")
p.add_argument("--rand_y", type=float, default=0.05, help="+- random shift of cube start y (m)")
p.add_argument("--rand_speed", type=float, default=0.2, help="+- fractional random cube speed")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="grab_dataset.npz")
args = p.parse_args()

rng = np.random.default_rng(args.seed)
model = mujoco.MjModel.from_xml_path(args.task_xml)
data = mujoco.MjData(model)


def num_adr(name):
    return model.numeric_adr[model.numeric(name).id]


def get_num(name):
    return float(model.numeric_data[num_adr(name)])


def set_num(name, value):
    model.numeric_data[num_adr(name)] = value


# nominal values from task.xml (re-randomised every episode)
NOM = {k: get_num(k) for k in ("cuboid_x0", "cuboid_y0", "cuboid_z0",
                               "cuboid_vx", "cuboid_vy", "cuboid_vz")}
THR = get_num("grab_threshold")


def transition(model, data):
    """Python copy of Grab::TransitionLocked. Returns True if the cube is grabbed."""
    dt = data.time - data.userdata[1]
    data.userdata[1] = data.time

    grabbed = False
    if data.time > 0.1:  # sensors are not valid before the first step
        eL = data.sensor("tipL_pos").data - data.sensor("faceL_pos").data
        eR = data.sensor("tipR_pos").data - data.sensor("faceR_pos").data
        grabbed = np.linalg.norm(eL) < THR and np.linalg.norm(eR) < THR

    if not grabbed and dt > 0:
        data.userdata[0] += dt
    s = data.userdata[0]

    data.mocap_pos[0] = [get_num("cuboid_x0") + get_num("cuboid_vx") * s,
                         get_num("cuboid_y0") + get_num("cuboid_vy") * s,
                         get_num("cuboid_z0") + get_num("cuboid_vz") * s]
    return grabbed


keys = ("episode", "t", "qpos", "qvel", "ctrl", "cuboid_pos", "cuboid_vel",
        "tipL", "tipR", "grabbed")
log = {k: [] for k in keys}
n_steps = int(args.seconds / model.opt.timestep)

for ep in range(args.episodes):
    # randomise the cube for this episode (the planner gets the same values, because the
    # Agent below is created from this modified model)
    set_num("cuboid_x0", NOM["cuboid_x0"] + rng.uniform(-args.rand_x, args.rand_x))
    set_num("cuboid_y0", NOM["cuboid_y0"] + rng.uniform(-args.rand_y, args.rand_y))
    set_num("cuboid_vy", NOM["cuboid_vy"] * (1 + rng.uniform(-args.rand_speed, args.rand_speed)))

    mujoco.mj_resetDataKeyframe(model, data, 0)  # "home" pose; also zeroes userdata
    transition(model, data)                      # place cube at its start
    mujoco.mj_forward(model, data)

    ever_grabbed = False
    with agent_lib.Agent(task_id="Grab", model=model) as agent:
        for _ in range(n_steps):
            agent.set_state(time=data.time, qpos=data.qpos, qvel=data.qvel, act=data.act,
                            mocap_pos=data.mocap_pos, mocap_quat=data.mocap_quat,
                            userdata=data.userdata)
            for _ in range(args.planner_steps):
                agent.planner_step()
            data.ctrl[:] = agent.get_action()

            mujoco.mj_step(model, data)
            grabbed = transition(model, data)
            ever_grabbed |= bool(grabbed)

            log["episode"].append(ep)
            log["t"].append(data.time)
            log["qpos"].append(data.qpos.copy())
            log["qvel"].append(data.qvel.copy())
            log["ctrl"].append(data.ctrl.copy())
            log["cuboid_pos"].append(data.mocap_pos[0].copy())
            log["cuboid_vel"].append([get_num("cuboid_vx"), get_num("cuboid_vy"),
                                      get_num("cuboid_vz")])
            log["tipL"].append(data.sensor("tipL_pos").data.copy())
            log["tipR"].append(data.sensor("tipR_pos").data.copy())
            log["grabbed"].append(bool(grabbed))
    print(f"episode {ep + 1}/{args.episodes}: grabbed={ever_grabbed}")

np.savez(args.out, **{k: np.array(v) for k, v in log.items()})
print("saved", args.out, "with", len(log["t"]), "samples")
