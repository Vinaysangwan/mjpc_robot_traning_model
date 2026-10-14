"""Find the real tip of each wrist link.

Tip = centre of the cluster of mesh vertices farthest from the wrist hinge.
Run from the folder containing task.xml and meshes/:   python3 find_tips.py
Paste the printed pos="..." into the <site name="tipL"/"tipR"> lines in task.xml.
"""
import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("task.xml")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)  # qpos = 0 -> CAD pose; every body frame == world frame here

for side, body, joint in (("L", "Wrist_Left_Link", "WristL_Joint"),
                          ("R", "Wrist_Right_Link", "WristR_Joint")):
    bid = m.body(body).id
    anchor = d.xanchor[m.joint(joint).id]
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[a:a + n].reshape(-1, 3)
            w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T  # world coords
            dist = np.linalg.norm(w - anchor, axis=1)
            tip = w[dist > dist.max() - 0.01].mean(axis=0)
            print(f"tip{side}:  pos=\"{tip[0]:.4f} {tip[1]:.4f} {tip[2]:.4f}\"   "
                  f"(farthest point is {dist.max():.3f} m from the wrist hinge)")
            break
