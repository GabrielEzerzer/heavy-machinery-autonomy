"""Test scenes: dig_pit() leaves each machine's tracks on two strips of ground over a pit that the
bucket reaches only near full depth; held() puts a running scene's joints on the nominal oil
where they stand; lock_joints() pins them rigidly."""

import mujoco

from hma.hydraulics import NOMINAL, Hydraulics
from hma.machine import JOINT_NAMES, machine_prefix
from hma.machine.ground import Box, Ground, apply_ground

PIT_DEPTH = 6.0
STRIP_X = 1.7
STRIP_Y = (0.6, 1.5)
PIT = Ground(level=-PIT_DEPTH, boxes=tuple(
    Box(0.0, side * (STRIP_Y[0] + STRIP_Y[1]) / 2, 0.0, STRIP_X, (STRIP_Y[1] - STRIP_Y[0]) / 2,
        PIT_DEPTH) for side in (-1, 1)))


def dig_pit(model, n_machines=1):
    for i in range(n_machines):
        apply_ground(model, i, PIT)
    return model


LAYERS = {}
FIELDS = ("qpos", "qvel", "eq_active", "xfrc_applied", "mocap_pos", "mocap_quat", "userdata")


def held(model, data):
    layer = LAYERS.get(id(data))
    if layer is None or layer.data is not data:
        count = sum(model.body(i).name.endswith("/base") for i in range(model.nbody))
        kept = [getattr(data, k).copy() for k in FIELDS]
        poses = [[data.qpos[model.joint(machine_prefix(i) + j).qposadr[0]] for j in JOINT_NAMES]
                 for i in range(count)]
        layer = Hydraulics(model, data, [NOMINAL] * count)
        for index, pose in enumerate(poses):
            layer.reset(index, NOMINAL, pose)
        for name, value in zip(FIELDS, kept, strict=True):
            getattr(data, name)[:] = value
        mujoco.mj_forward(model, data)
        LAYERS[id(data)] = layer
    return layer


def lock_joints(model, data, index=0, names=JOINT_NAMES):
    for name in names:
        joint = model.joint(machine_prefix(index) + name)
        angle = data.qpos[joint.qposadr[0]]
        model.jnt_range[joint.id] = [angle - 1e-6, angle + 1e-6]
        model.jnt_limited[joint.id] = 1
