"""Static checks of the cylinder force limits against the ZX120 spec sheet: ISO 6015 bucket and arm
digging forces, hydraulic and tipping lift capacity at the published chart points, and whether the
boom can push the bucket into the ground hard enough to lift the front of the tracks."""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from hma.machine import JOINT_LIMITS, TIP_SITE, load_scene, machine_prefix
from hma.machine.cylinders import MAIN_RELIEF, PORT_RELIEF, joint_torque_limits
from hma.machine.tracks import FRONT_LINE, REAR_LINE, SIDE_LINE

CHART = {
    (4.5, 4.5): ("*3350", "*3350"), (4.5, 6.0): ("3040", "2110"),
    (3.0, 3.0): ("*5460", "*5460"), (3.0, 4.5): ("*4070", "3200"), (3.0, 6.0): ("2980", "2050"),
    (1.5, 3.0): ("*7890", "5380"), (1.5, 4.5): ("4470", "2980"), (1.5, 6.0): ("2880", "1960"),
    (0.0, 3.0): ("*6730", "5070"), (0.0, 4.5): ("4280", "2810"), (0.0, 6.0): ("2790", "1880"),
    (-1.5, 1.5): ("*4630", "*4630"), (-1.5, 3.0): ("8440", "5030"), (-1.5, 4.5): ("4200", "2740"),
    (-1.5, 6.0): ("2760", "1850"),
    (-3.0, 1.5): ("*8540", "*8540"), (-3.0, 3.0): ("*7340", "5120"), (-3.0, 4.5): ("4240", "2780"),
}
OLD_CAPS = {"boom": 330.0e3, "arm": 244.0e3, "bucket": 122.0e3}
HYDRAULIC_SHARE, TIPPING_SHARE = 0.87, 0.75


class Machine:
    def __init__(self) -> None:
        self.model = load_scene(1, weld=True)
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.data = mujoco.MjData(self.model)
        p = machine_prefix(0)
        self.joints = {j: self.model.joint(p + j) for j in ("swing", "boom", "arm", "bucket")}
        self.dofs = {j: int(jt.dofadr[0]) for j, jt in self.joints.items()}
        self.bodies = {b: self.model.body(p + b).id for b in ("base", "arm", "bucket")}
        self.subtree = [b for b in range(self.model.nbody) if self._under(b, self.bodies["base"])]
        self.tip = self.model.site(p + TIP_SITE).id
        self.g = -float(self.model.opt.gravity[2])
        self.bucket_mass = float(self.model.body_mass[self.bodies["bucket"]])

    def _under(self, body: int, root: int) -> bool:
        while body != root and body != 0:
            body = int(self.model.body_parentid[body])
        return body == root

    def pose(self, swing: float, boom: float, arm: float, bucket: float, with_bucket: bool) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        angles = {"swing": swing, "boom": boom, "arm": arm, "bucket": bucket}
        for j, q in angles.items():
            d.qpos[self.joints[j].qposadr[0]] = q
        m.body_mass[self.bodies["bucket"]] = self.bucket_mass if with_bucket else 0.0
        mujoco.mj_forward(m, d)

    def total(self) -> tuple[float, np.ndarray]:
        mass = self.model.body_mass[self.subtree]
        total = float(mass.sum())
        return total, (mass[:, None] * self.data.xipos[self.subtree]).sum(0) / total

    def point_jacobian(self, point: np.ndarray, body: str) -> np.ndarray:
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jac, None, point, self.bodies[body])
        return jac

    def anchor(self, joint: str) -> np.ndarray:
        return self.data.xanchor[self.joints[joint].id].copy()

    def bias(self, joint: str) -> float:
        return float(self.data.qfrc_bias[self.dofs[joint]])

    def done(self) -> None:
        self.model.body_mass[self.bodies["bucket"]] = self.bucket_mass


def caps(joint: str, q: float, pressures: tuple[float, float]) -> tuple[float, float]:
    plus, minus = joint_torque_limits(joint, q, *pressures)
    return float(plus), float(minus)


def force_room(required0: float, per_unit: float, plus: float, minus: float) -> float:
    if per_unit > 0:
        return (plus - required0) / per_unit
    if per_unit < 0:
        return (-minus - required0) / per_unit
    return np.inf


def digging(machine: Machine) -> dict[str, float]:
    q = np.linspace(*JOINT_LIMITS["bucket"], 4001)
    curl, _ = joint_torque_limits("bucket", q, MAIN_RELIEF, MAIN_RELIEF)
    best = int(np.argmax(curl))
    machine.pose(0.0, 0.0, 1.5, q[best], True)
    lip = float(np.linalg.norm(machine.data.site_xpos[machine.tip] - machine.anchor("bucket")))
    arm_lip = float(np.linalg.norm(machine.data.site_xpos[machine.tip] - machine.anchor("arm")))
    qa = np.linspace(*JOINT_LIMITS["arm"], 4001)
    crowd, _ = joint_torque_limits("arm", qa, MAIN_RELIEF, MAIN_RELIEF)
    return {"bucket_moment": float(curl[best]), "bucket_angle": float(q[best]), "lip": lip,
            "bucket_force": float(curl[best]) / lip, "arm_moment": float(crowd.max()),
            "arm_angle": float(qa[np.argmax(crowd)]), "arm_lip": arm_lip,
            "arm_force": float(crowd.max()) / arm_lip}


def reach_load_point(machine: Machine, radius: float, height: float) -> tuple[float, float] | None:
    best = None
    for boom in np.linspace(*JOINT_LIMITS["boom"], 241):
        for arm in np.linspace(*JOINT_LIMITS["arm"], 241):
            machine.pose(0.0, boom, arm, 0.0, False)
            x, _, z = machine.anchor("bucket")
            err = (x - radius) ** 2 + (z - height) ** 2
            if best is None or err < best[0]:
                best = (err, boom, arm)
    _, boom, arm = best
    for _ in range(60):
        machine.pose(0.0, boom, arm, 0.0, False)
        x, _, z = machine.anchor("bucket")
        r = np.array([x - radius, z - height])
        if np.linalg.norm(r) < 1e-6:
            break
        jac = machine.point_jacobian(machine.anchor("bucket"), "arm")
        j2 = jac[np.ix_([0, 2], [machine.dofs["boom"], machine.dofs["arm"]])]
        boom, arm = np.array([boom, arm]) - np.linalg.lstsq(j2, r, rcond=None)[0]
    machine.pose(0.0, boom, arm, 0.0, False)
    x, _, z = machine.anchor("bucket")
    miss = float(np.hypot(x - radius, z - height))
    lo_b, hi_b = JOINT_LIMITS["boom"]
    lo_a, hi_a = JOINT_LIMITS["arm"]
    if miss > 1e-4 or not (lo_b <= boom <= hi_b and lo_a <= arm <= hi_a):
        return None
    return float(boom), float(arm)


def lift(machine: Machine, height: float, radius: float, arm_pressure: float) -> dict | None:
    pose = reach_load_point(machine, radius, height)
    if pose is None:
        return None
    boom, arm = pose
    machine.pose(0.0, boom, arm, 0.0, False)
    point = machine.anchor("bucket")
    jac = machine.point_jacobian(point, "arm")
    room = {}
    for joint, q, pressures in (("boom", boom, (MAIN_RELIEF, MAIN_RELIEF)),
                                ("arm", arm, (arm_pressure, arm_pressure))):
        per_kg = machine.g * jac[2, machine.dofs[joint]]
        room[joint] = force_room(machine.bias(joint), per_kg, *caps(joint, q, pressures))
    hydraulic = min(room.values())
    mass, com = machine.total()
    front = (mass * (FRONT_LINE - com[0]) / (point[0] - FRONT_LINE)
             if point[0] > FRONT_LINE else None)
    machine.pose(np.pi / 2, boom, arm, 0.0, False)
    mass, com = machine.total()
    side_point = machine.anchor("bucket")
    side = mass * (SIDE_LINE - com[1]) / (side_point[1] - SIDE_LINE)
    return {"boom": boom, "arm": arm, "hydraulic": hydraulic,
            "limited_by": min(room, key=room.get), "front_tip": front, "side_tip": side}


def propping(machine: Machine, arm_bucket_pressure: dict[str, tuple[float, float]] | None,
             old: bool = False) -> list[dict]:
    rows = {}
    for boom in np.linspace(*JOINT_LIMITS["boom"], 61):
        for arm in np.linspace(*JOINT_LIMITS["arm"], 61):
            for bucket in np.linspace(*JOINT_LIMITS["bucket"], 31):
                machine.pose(0.0, boom, arm, bucket, True)
                tip = machine.data.site_xpos[machine.tip]
                if abs(tip[2]) > 0.05 or tip[0] < FRONT_LINE + 1.0:
                    continue
                jac = machine.point_jacobian(tip, "bucket")
                room = {}
                for joint, q in (("boom", boom), ("arm", arm), ("bucket", bucket)):
                    if old:
                        plus = minus = OLD_CAPS[joint]
                    else:
                        pressures = ((MAIN_RELIEF, MAIN_RELIEF) if joint == "boom"
                                     else arm_bucket_pressure[joint])
                        plus, minus = caps(joint, q, pressures)
                    room[joint] = force_room(machine.bias(joint), -jac[2, machine.dofs[joint]],
                                             plus, minus)
                force = min(room.values())
                mass, com = machine.total()
                needed = mass * machine.g * (com[0] - REAR_LINE) / (tip[0] - REAR_LINE)
                key = round(float(tip[0]) * 2) / 2
                row = {"reach": key, "force": force, "needed": needed,
                       "limited_by": min(room, key=room.get), "pose": (boom, arm, bucket)}
                if key not in rows or force / needed > rows[key]["force"] / rows[key]["needed"]:
                    rows[key] = row
    return [rows[k] for k in sorted(rows)]


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    machine = Machine()
    dig = digging(machine)
    print("ISO 6015 digging forces at main relief 34.3 MPa")
    print(f"  bucket: {dig['bucket_moment'] / 1e3:.1f} kN m at q={dig['bucket_angle']:.3f}, "
          f"lip radius {dig['lip']:.3f} m -> {dig['bucket_force'] / 1e3:.1f} kN "
          f"(spec sheet ISO 99, SAE 86); radius for 99 kN {dig['bucket_moment'] / 99e3:.3f} m")
    print(f"  arm: {dig['arm_moment'] / 1e3:.1f} kN m at q={dig['arm_angle']:.3f}, "
          f"arm pivot to lip {dig['arm_lip']:.3f} m -> {dig['arm_force'] / 1e3:.1f} kN "
          f"(spec sheet ISO 65, SAE 63); radius for 65 kN {dig['arm_moment'] / 65e3:.3f} m")
    print("\nLift chart, no bucket, rating = min(0.87 hydraulic, 0.75 tipping), kg")
    print("  height radius | chart front side | 0.87 hyd (arm main / port) limited | "
          "0.75 tip front side")
    for (height, radius), (front, side) in CHART.items():
        main_ = lift(machine, height, radius, MAIN_RELIEF)
        port = lift(machine, height, radius, PORT_RELIEF["arm"][0])
        if main_ is None:
            print(f"  {height:5.1f} {radius:5.1f} | {front:>6s} {side:>6s} | out of reach")
            continue
        tip = ("inside tipping line" if main_["front_tip"] is None
               else f"{TIPPING_SHARE * main_['front_tip']:6.0f}")
        print(f"  {height:5.1f} {radius:5.1f} | {front:>6s} {side:>6s} | "
              f"{HYDRAULIC_SHARE * main_['hydraulic']:6.0f} / "
              f"{HYDRAULIC_SHARE * port['hydraulic']:6.0f} {main_['limited_by']:5s} | "
              f"{tip} {TIPPING_SHARE * main_['side_tip']:6.0f}")
    print("\nPropping: tip force the joints can hold against what lifts the front of the tracks")
    cases = {"main relief on arm and bucket": {"arm": (MAIN_RELIEF,) * 2,
                                               "bucket": (MAIN_RELIEF,) * 2},
             "port relief on arm and bucket (unverified)": {"arm": PORT_RELIEF["arm"],
                                                            "bucket": PORT_RELIEF["bucket"]}}
    for name, pressures in [*cases.items(), ("old constant caps", None)]:
        print(f"  {name}")
        for row in propping(machine, pressures, old=pressures is None):
            print(f"    reach {row['reach']:.1f} m: holds {row['force'] / 1e3:6.1f} kN, "
                  f"lift needs {row['needed'] / 1e3:6.1f} kN, "
                  f"ratio {row['force'] / row['needed']:.2f}, limited by {row['limited_by']}, "
                  f"pose {np.round(row['pose'], 2)}")
    machine.done()


if __name__ == "__main__":
    main()
