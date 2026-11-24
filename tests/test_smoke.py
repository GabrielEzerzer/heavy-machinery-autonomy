import mujoco

import hma

PENDULUM = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <body pos="0 0 1">
      <joint type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.02" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_package_imports():
    assert hma.__version__


def test_mujoco_steps_a_model():
    model = mujoco.MjModel.from_xml_string(PENDULUM)
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert abs(data.time - 0.5) < 1e-9
    assert data.qpos[0] != 0.0
