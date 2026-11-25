"""Test scenes: dig_pit() leaves each machine's tracks on two strips of ground over a pit that the
bucket reaches only near full depth; held() puts a running scene's joints on the nominal oil
where they stand; lock_joints() pins them rigidly."""

import mujoco
import numpy as np

from hma.hydraulics import NOMINAL, Hydraulics
from hma.machine import JOINT_NAMES, ground_axes, machine_prefix, write_ground
from hma.machine.model import GROUND_DEPTH

STRIP_X = 1.7
STRIP_Y = (0.6, 1.5)


def dig_pit(model, n_machines=1):
    xs, ys = ground_axes()
    strips = ((np.abs(ys)[:, None] >= STRIP_Y[0]) & (np.abs(ys)[:, None] <= STRIP_Y[1])
              & (np.abs(xs)[None, :] <= STRIP_X))
    heights = np.where(strips, 0.0, -GROUND_DEPTH)
    for i in range(n_machines):
        write_ground(model, i, heights)
    return model


LAYERS = {}


def held(model, data):
    layer = LAYERS.get(id(data))
    if layer is None or layer.data is not data:
        count = sum(model.body(i).name.endswith("/base") for i in range(model.nbody))
        kept = [getattr(data, k).copy() for k in ("qpos", "qvel", "eq_active", "xfrc_applied")]
        poses = [[data.qpos[model.joint(machine_prefix(i) + j).qposadr[0]] for j in JOINT_NAMES]
                 for i in range(count)]
        layer = Hydraulics(model, data, [NOMINAL] * count)
        for index, pose in enumerate(poses):
            layer.reset(index, NOMINAL, pose)
        data.qpos[:], data.qvel[:], data.eq_active[:], data.xfrc_applied[:] = kept
        mujoco.mj_forward(model, data)
        LAYERS[id(data)] = layer
    return layer


def lock_joints(model, data, index=0, names=JOINT_NAMES):
    for name in names:
        joint = model.joint(machine_prefix(index) + name)
        angle = data.qpos[joint.qposadr[0]]
        model.jnt_range[joint.id] = [angle - 1e-6, angle + 1e-6]
        model.jnt_limited[joint.id] = 1
