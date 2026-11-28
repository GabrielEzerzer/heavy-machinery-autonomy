
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from hma.hydraulics import NOMINAL, RANGES, BaseController
from hma.machine import machine_prefix, tracks
from hma.machine.kinematics import BOOM_PIVOT
from hma.machine.model import ground_geoms
from hma.train import evaluate, footing
from hma.train.scene import CONTROL_HZ, HISTORY, PropScene, reward
from hma.train.task import (
    DEV_BASE,
    HELD_OUT_BASE,
    PropPath,
    Settings,
    dev_seeds,
    draw_task,
    reachable,
)

SEED = DEV_BASE + 3
STRIKE_POSE = (-2.36, -0.369, 2.501, 0.521)


def finish(scene, index=0):
    while True:
        _, reward, done, info = scene.step(np.zeros((scene.n, 4)))
        if done[index]:
            return info[index], reward[index]


class Reference(PropScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feet = np.concatenate(footing.front_parts(self.model, 0))
        self.body = self.model.body(machine_prefix(0) + "base").id
        self.touching, self.worst = set(), 0.0

    def reset(self, index, seed):
        obs = super().reset(index, seed)
        self.touching = self._touching()
        self.worst = 0.0
        return obs

    def _touching(self):
        pairs = self.data.contact.geom[:self.data.ncon]
        return {int(g) for g in pairs.ravel() if g in self.feet}

    def _contacts(self):
        touching = self._touching()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.body,
                                 velocity, 0)
        rot = self.data.xmat[self.body].reshape(3, 3)
        for g in touching - self.touching:
            corner = footing.lowest_points(self.model, [g])[0]
            point = velocity[3:] + np.cross(velocity[:3], rot @ corner)
            self.worst = max(self.worst, -point[2])
        self.touching = touching
        return super()._contacts()


def test_landing_is_each_front_parts_speed_just_before_its_own_first_contact():
    scene = Reference(1, "nominal")
    scene.reset(0, DEV_BASE + 4)
    info, _ = finish(scene)
    assert scene.worst > 0.1
    assert info["landing"] == pytest.approx(scene.worst, abs=1e-9)


def test_restoring_never_ends_an_episode_over_a_stale_warning_count():
    scene = PropScene(1, "nominal")
    scene.reset(0, SEED)
    for _ in range(3):
        scene.step(np.zeros((1, 4)))
    saved = scene.snapshot()
    other = PropScene(1, "nominal")
    other.reset(0, SEED + 1)
    other.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 5
    other.restore(saved)
    _, _, done, _ = other.step(np.zeros((1, 4)))
    assert not done[0]


@pytest.mark.parametrize("kind, rung, before", [("acceleration", "nominal", 2),
                                                ("velocity", "nominal", 2),
                                                ("acceleration", "ground", 5)])
def test_a_solver_failure_ends_only_its_own_machine_and_cuts_the_rest_short(kind, rung, before):
    scene = PropScene(3, rung)
    for i in range(3):
        scene.reset(i, SEED + i)
    for _ in range(before):
        obs = scene.step(np.zeros((3, 4)))[0]
    error = scene.err_total / scene.steps_total
    speed = np.linalg.norm(scene._part_velocity(), axis=1).max(1)
    if kind == "acceleration":
        scene.data.xfrc_applied[scene.model.body(machine_prefix(1) + "bucket").id, 2] = 1e30
    else:
        scene.data.qvel[scene._base_v[1]] = np.nan
    after, reward, done, info = scene.step(np.zeros((3, 4)))
    assert done.all() and np.all(reward == 0.0)
    assert np.all(np.isfinite(after)) and np.array_equal(after, obs)
    assert info[1]["cause"] == "unstable" and not info[1]["truncated"]
    for i in (0, 2):
        assert info[i]["cause"] is None and info[i]["truncated"] and info[i]["cut_short"]
        assert not info[i]["passed"]
        assert info[i]["tilt_error"] == error[i]
        assert info[i]["front_speed"] == speed[i]


def test_a_machine_that_goes_non_finite_without_a_warning_ends_alone_and_keeps_a_finite_view():
    scenes = [PropScene(3, "nominal") for _ in range(2)]
    for scene in scenes:
        for i in range(3):
            scene.reset(i, SEED + i)
        for _ in range(3):
            obs = scene.step(np.zeros((3, 4)))[0]
    broken, plain = scenes
    calls = []
    step = broken.layer.step

    def poisoned():
        step()
        calls.append(None)
        if len(calls) == broken.steps:
            broken.layer._pressure[1] = np.nan
            broken.layer._clean[broken.layer._clean_head, :, 1] = np.nan

    broken.layer.step = poisoned
    after, reward, done, info = broken.step(np.zeros((3, 4)))
    expected, _, _, _ = plain.step(np.zeros((3, 4)))
    assert np.all(np.isfinite(after)) and np.array_equal(after[1], obs[1])
    assert done[1] and info[1]["cause"] == "unstable" and reward[1] == 0.0
    assert not done[0] and not done[2]
    assert np.array_equal(after[[0, 2]], expected[[0, 2]])


def test_the_bucket_striking_its_own_machine_ends_the_episode_as_a_strike():
    scene = PropScene(1, "nominal", mode="direct")
    scene.reset(0, SEED)
    scene.layer.reset(0, scene.machines[0], STRIKE_POSE)
    mujoco.mj_forward(scene.model, scene.data)
    _, reward, done, info = scene.step(np.zeros((1, 4)))
    assert done[0] and info[0]["cause"] == "strike" and reward[0] == 0.0


def test_the_pass_mark_at_each_of_its_boundaries():
    s = Settings()
    scene = PropScene(1, "nominal")
    scene.reset(0, SEED)
    scene.hold_steps[0] = 1000
    tiny = 1e-9

    def passes(share=s.hold_share, landing=0.0, carried=1.0, speed=0.0, cause=None,
               bucket=False, front=True):
        scene.in_band[0], scene.landing[0] = round(share * 1000), landing
        return scene.outcome(0, cause, 0.0, carried, speed, bucket, front)["passed"]

    assert passes() and not passes(share=s.hold_share - 0.001)
    assert passes(landing=s.landing - tiny) and not passes(landing=s.landing)
    assert passes(carried=s.carry) and not passes(carried=s.carry - tiny)
    assert passes(speed=s.still - tiny) and not passes(speed=s.still)
    assert not passes(bucket=True) and not passes(front=False)
    assert not passes(cause="tilt")


@pytest.mark.train_extra
def test_the_scheduled_end_is_a_truncation_and_an_early_end_is_terminal():
    from hma.train.vec import SceneVecEnv

    env = SceneVecEnv(2, "nominal", "direct", 0, seeds=[DEV_BASE, DEV_BASE])
    env.reset()
    seen = {}
    while len(seen) < 2:
        actions = env.scene.baseline()
        actions[1] = (0.0, 1.0, 0.0, 0.0)
        env.step_async(actions)
        _, reward, done, infos = env.step_wait()
        for i in np.flatnonzero(done):
            seen.setdefault(int(i), (infos[i], float(reward[i])))
    (end, _), (early, early_reward) = seen[0], seen[1]
    assert end["cause"] is None and end["TimeLimit.truncated"]
    assert end["terminal_observation"].shape == env.observation_space.shape
    assert early["cause"] == "tilt" and not early["TimeLimit.truncated"] and early_reward == 0.0


def test_scoring_refuses_held_out_and_repeated_seeds():
    for seeds in ([DEV_BASE, HELD_OUT_BASE], [DEV_BASE, DEV_BASE]):
        with pytest.raises(ValueError):
            evaluate.score("nominal", "residual", seeds, 2, 1, None)


def test_a_solver_failure_while_scoring_reruns_the_other_seeds_to_the_same_result(monkeypatch):
    seeds = dev_seeds(3)
    clean = evaluate.score("nominal", "residual", seeds, 3, 1, None)

    class Failing(PropScene):
        struck = False

        def step(self, actions):
            if not Failing.struck and self.k[1] == 3:
                Failing.struck = True
                self.data.xfrc_applied[self.model.body(machine_prefix(1) + "bucket").id, 2] = 1e30
            return super().step(actions)

    monkeypatch.setattr(evaluate, "PropScene", Failing)
    failed = evaluate.score("nominal", "residual", seeds, 3, 1, None)
    assert Failing.struck
    assert [r["seed"] for r in failed] == seeds and [r["slot"] for r in failed] == [0, 1, 2]
    assert failed[1]["cause"] == "unstable"
    for k in (0, 2):
        assert failed[k] == clean[k]
    assert all(r["scene_size"] == 3 for r in failed)


def test_every_ground_task_stands_evenly_presses_on_one_flat_surface_and_is_in_reach():
    scene = PropScene(1, "ground")
    side = float(BOOM_PIVOT[1])
    body = scene.model.body(machine_prefix(0) + "base").id
    for seed in dev_seeds(4):
        scene.reset(0, seed)
        task, height = scene.tasks[0], scene.heights[0]
        origin, rot = scene.data.xpos[body], scene.data.xmat[body].reshape(3, 3)
        stance = footing.height_under(scene.model, 0, origin, rot, footing.FOOTPRINT)
        press = footing.height_under(scene.model, 0, origin, rot,
                                     footing.PRESS + [task.prop_x, side])
        assert footing.slope_deg(scene.grounds[0]) <= 10.0, seed
        assert np.abs(stance).max() <= footing.EVEN, seed
        assert np.ptp(press) <= footing.EVEN, seed
        assert height == scene.height_under(0, task.prop_x, side)
        assert reachable(PropPath(task, scene.pivot, height), scene.period), seed
        assert reachable(PropPath(task, scene.pivot), scene.period), seed
        scene.step(np.zeros((1, 4)))
        assert scene.rest_count[0] == 1 and (scene.rest_load[0] > 0.0).all(), seed


def test_a_ground_draw_that_does_not_fit_the_field_is_drawn_again(monkeypatch):
    from hma.train import scene as scene_module

    draw, failed, drawn = scene_module.draw_ground, [], []

    def first_does_not_fit(seed, attempt=0):
        if attempt == 0:
            failed.append(seed)
            raise RuntimeError("a drawn feature does not fit in the field")
        drawn.append(draw(seed, attempt))
        return drawn[-1]

    scene = PropScene(1, "ground")
    monkeypatch.setattr(scene_module, "draw_ground", first_does_not_fit)
    scene.reset(0, SEED)
    assert failed == [SEED]
    assert scene.grounds[0] is drawn[-1]


def test_the_score_does_not_depend_on_the_number_of_processes():
    seeds = dev_seeds(2)
    alone = evaluate.score("nominal", "residual", seeds, 1, 1, None)
    spread = evaluate.score("nominal", "residual", seeds, 1, 2, None)
    assert [r["seed"] for r in alone] == seeds
    assert alone == spread


def test_the_hold_and_its_band_count_only_the_physics_steps_inside_the_hold_span():
    scene = PropScene(1, "nominal")
    scene.reset(0, DEV_BASE)
    info, _ = finish(scene)
    start, end = scene.tasks[0].hold_span
    times = np.arange(1, info["decisions"] * scene.steps + 1) * scene.dt
    assert scene.hold_steps[0] == np.count_nonzero((times >= start) & (times < end))
    assert 0 < scene.hold_steps[0] < len(times)
    assert 0 < scene.in_band[0] <= scene.hold_steps[0]


def test_each_front_load_column_is_the_ground_force_on_that_tracks_own_front_parts():
    scene = PropScene(2, "nominal")
    for i in range(2):
        scene.reset(i, SEED + i)
    scene.step(np.zeros((2, 4)))
    model, data = scene.model, scene.data
    wheels = {str(k) for k, x in enumerate(tracks.WHEEL_X) if x > 0.0}
    belts = {f"belt_{k}" for k, x in enumerate(tracks.BELT_X) if x > 0.0}
    expected = np.zeros((2, 2))
    force = np.zeros(6)
    for i in range(2):
        for k, side in enumerate(tracks.SIDES):
            names = [model.geom(g).name for g in footing.front_parts(model, i)[k]]
            stem = machine_prefix(i) + tracks.TRACK + side + "_"
            assert all(name.startswith(stem) for name in names)
        ground = set(ground_geoms(model, i).tolist())
        for c in range(data.ncon):
            a, b = (int(g) for g in data.contact.geom[c])
            if (a in ground) == (b in ground):
                continue
            name = model.geom(b if a in ground else a).name
            for k, side in enumerate(tracks.SIDES):
                stem = machine_prefix(i) + tracks.TRACK + side + "_"
                part = name[len(stem):] if name.startswith(stem) else None
                if part in wheels or part in belts or part == "front":
                    mujoco.mj_contactForce(model, data, c, force)
                    expected[i, k] += force[0]
    assert np.all(expected > 0.0)
    assert np.array_equal(scene._front_load(), expected)


def test_one_tracks_front_left_unloaded_at_the_end_fails_the_set_down():
    class Unloaded(PropScene):
        def _front_load(self):
            load = super()._front_load()
            if self.k[0] * self.period >= self.tasks[0].length - self.settings.window - 1e-9:
                load[0, 0] = 0.0
            return load

    outcomes = []
    for kind in (PropScene, Unloaded):
        scene = kind(1, "nominal")
        scene.reset(0, DEV_BASE)
        outcomes.append(finish(scene)[0])
    kept, unloaded = outcomes
    assert kept["passed"] and kept["front_carried"] >= Settings().carry
    assert unloaded["front_carried"] == 0.0 and not unloaded["passed"]
    assert unloaded["hold_share"] == kept["hold_share"]


def test_the_reward_stays_in_its_bounds_and_each_end_term_only_lowers_it():
    s = Settings()
    rng = np.random.default_rng(0)
    n = 20_000
    terms = {"error": rng.exponential(0.1, n), "change": rng.uniform(0.0, 1.0, n),
             "roll": rng.exponential(0.1, n), "slide": rng.exponential(0.05, n),
             "landing": rng.exponential(0.3, n), "shortfall": rng.uniform(0.0, 1.0, n),
             "rush": rng.exponential(0.2, n)}
    earned = reward(s, **terms)
    lowest = s.floor - sum(s.weights.values())
    assert lowest == pytest.approx(-0.99)
    assert np.all((earned >= lowest - 1e-12) & (earned <= 1.0))
    assert earned.min() < 0.0
    assert reward(s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) == pytest.approx(1.0)
    for name in ("landing", "shortfall", "rush"):
        cleared = reward(s, **{**terms, name: np.zeros(n)})
        assert np.all(cleared >= earned)
        assert np.all(cleared[terms[name] > 0.0] > earned[terms[name] > 0.0])


class Held(PropPath):
    def __call__(self, t):
        return super().__call__(min(t, self.task.length - self.task.final))


class Pressing(PropScene):
    def _write(self, index, seed):
        super()._write(index, seed)
        self.controllers[index] = BaseController(Held(self.tasks[index], self.pivot), CONTROL_HZ)


def test_skipping_the_lift_at_the_end_earns_less_and_fails_the_set_down():
    results = []
    for kind in (PropScene, Pressing):
        scene = kind(1, "nominal")
        scene.reset(0, DEV_BASE + 11)
        task, settle, earned = scene.tasks[0], [], []
        while True:
            _, r, done, info = scene.step(np.zeros((1, 4)))
            earned.append(r[0])
            settle.append(scene.k[0] * scene.period > task.length - task.final)
            if done[0]:
                results.append((info[0], np.array(earned), np.array(settle)))
                break
    (lifted, lifted_r, at_end), (held, held_r, _) = results
    assert lifted["passed"] and not held["passed"]
    assert held["front_carried"] < Settings().carry <= lifted["front_carried"]
    assert held_r[at_end].sum() < lifted_r[at_end].sum()
    assert held_r.sum() < lifted_r.sum()


def test_the_baseline_lifts_the_tip_back_along_the_path_it_put_it_down_on():
    pivot = footing.rear_pivot()
    task = draw_task(SEED, Settings(), pivot, lambda x, y: 0.0, 0.1)
    path = PropPath(task, pivot)
    down = task.length - task.final
    for u in np.linspace(0.0, task.place, 11):
        up, pitch_up = path(down + u)
        placed, pitch_placed = path(task.place - u)
        assert np.allclose(up, placed, atol=1e-12) and pitch_up == pitch_placed
    assert path(task.length)[0][2] == pytest.approx(task.start_z)


RUSH_SEED = DEV_BASE + 0


def test_a_falling_front_is_charged_and_a_baseline_set_down_or_a_resting_front_is_not(
        monkeypatch):
    import hma.train.scene as scene_module

    charged, real = [], scene_module.reward

    def capture(s, error, change, roll, slide, landing, shortfall, rush):
        charged.append(np.array(rush, float))
        return real(s, error, change, roll, slide, landing, shortfall, rush)

    monkeypatch.setattr(scene_module, "reward", capture)
    scene = PropScene(2, "nominal", mode="direct")
    scene.reset(0, RUSH_SEED)
    scene.reset(1, RUSH_SEED)
    task = scene.tasks[0]
    drop = round(sum(task.hold_span) / 2 / scene.period)
    rest = drop + 30
    released, down = np.zeros(4), []
    infos, k = [None, None], 0
    while None in infos:
        actions = scene.baseline()
        if k >= drop:
            actions[1] = released
            actions[1, 1] = -1.0 if k < drop + 10 else 0.0
        _, _, done, info = scene.step(actions)
        down.append(scene.front_prev[1].any())
        for i in np.flatnonzero(done):
            infos[i] = info[i]
        k += 1
    charged = np.array(charged)
    baseline, dropped = infos
    assert baseline["landing"] < 0.3 and baseline["front_down"]
    assert np.all(charged[:, 0] == 0.0)
    assert dropped["landing"] >= 0.3
    assert charged[drop:rest, 1].max() > Settings().rush_scale
    assert all(down[rest:]) and np.all(charged[rest:, 1] == 0.0)
    assert np.all(charged[:drop, 1] == 0.0)


def test_the_wait_is_in_the_length_and_the_bucket_stays_down_through_it():
    pivot = footing.rear_pivot()
    task = draw_task(SEED, Settings(), pivot, lambda x, y: 0.0, 0.1)
    assert task.wait == Settings().wait > 0.0
    assert Settings().ramp_down[0] <= task.ramp_down <= Settings().ramp_down[1]
    assert task.length == pytest.approx(task.rest + task.ramp_up + task.hold + task.ramp_down
                                        + task.wait + task.final)
    path = PropPath(task, pivot)
    landed = task.rest + task.ramp_up + task.hold + task.ramp_down + 1e-9
    pressed, pitch = path(landed)
    for t in np.linspace(landed, task.length - task.final, 16):
        tip, at = path(float(t))
        assert task.tilt(float(t)) == 0.0
        assert np.allclose(tip, pressed, atol=1e-12) and at == pitch
    assert path(task.length - task.final + task.place / 2)[0][2] > pressed[2] + 0.1


def test_a_full_scale_action_opens_a_valve_past_the_widest_drawn_dead_zone():
    widest = RANGES["deadband"][1]
    assert Settings().scale > widest

    class Widest(PropScene):
        def machine_for(self, index, seed):
            return replace(NOMINAL, deadband=np.full(4, widest))

        def baseline(self):
            return np.zeros((self.n, 4))

    scene = Widest(1, "nominal")
    scene.reset(0, SEED)
    for _ in range(30):
        scene.step(np.ones((1, 4)))
    assert np.allclose(scene.sent, Settings().scale)
    assert np.all(np.abs(scene.layer._spool[0]) > widest)


def test_a_front_bouncing_with_the_bucket_clear_passes_and_a_pressing_bucket_fails():
    class Bouncing(PropScene):
        def _front_load(self):
            left = np.ceil((self.tasks[0].length - self.k[0] * self.period) / self.period - 1e-9)
            self.factor = 0.3 if left % 2 else 1.7
            return super()._front_load() * self.factor

    class Propped(Pressing):
        def _front_load(self):
            rest = self.rest_load / np.maximum(self.rest_count, 1)[:, None]
            return rest if self.touched[0] else super()._front_load()

    def bouncing_share(scene):
        rest = scene.rest_load[0] / scene.rest_count[0]
        return np.min(PropScene._front_load(scene)[0] / rest)

    s = Settings()
    outcomes = []
    for kind in (Bouncing, Propped):
        scene = kind(1, "nominal")
        scene.reset(0, DEV_BASE)
        outcomes.append(finish(scene)[0])
        if kind is Bouncing:
            assert scene.factor * bouncing_share(scene) < s.carry
    bouncing, propped = outcomes
    assert bouncing["passed"] and not bouncing["bucket_down"]
    assert bouncing["front_carried"] >= s.carry
    assert propped["bucket_down"] and propped["front_carried"] >= s.carry
    assert not propped["passed"]


def test_a_landing_is_charged_only_while_it_is_inside_the_observation_history():
    s = Settings()
    free = replace(s, weights={**s.weights, "landing": 0.0})
    scene, uncharged = PropScene(1, "nominal", settings=s), PropScene(1, "nominal", settings=free)
    scene.reset(0, DEV_BASE + 4)
    uncharged.reset(0, DEV_BASE + 4)
    fresh, gap = [], []
    while True:
        _, earned, done, info = scene.step(np.zeros((1, 4)))
        _, kept, _, _ = uncharged.step(np.zeros((1, 4)))
        fresh.append(scene.recent[0, -1])
        gap.append(kept[0] - earned[0])
        if done[0]:
            break
    fresh, gap = np.array(fresh), np.array(gap)
    assert fresh.max() == pytest.approx(info[0]["landing"]) and fresh.max() > 0.1
    charged = np.array([fresh[max(0, k - HISTORY + 1):k + 1].max() for k in range(len(fresh))])
    expected = s.weights["landing"] * np.minimum(charged / s.landing, 1.0)
    assert gap == pytest.approx(expected, abs=1e-12)
    last = np.flatnonzero(fresh)[-1]
    assert last + HISTORY < len(fresh)
    assert np.all(gap[last + HISTORY:] == pytest.approx(0.0, abs=1e-12))


class Traced(PropScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tips, self.touching, self.live = [], [], False

    def step(self, actions):
        self.live = True
        return super().step(actions)

    def _contacts(self):
        found = super()._contacts()
        if self.live:
            self.tips.append(self.data.site_xpos[self._tips[0], :2].copy())
            self.touching.append(bool(found[0][0]))
        return found


def test_the_slide_term_charges_the_tips_path_through_the_contact_hop():
    s = replace(Settings(), slide_scale=10.0)
    free = replace(s, weights={**s.weights, "slide": 0.0})
    traced, uncharged = Traced(1, "nominal", settings=s), PropScene(1, "nominal", settings=free)
    for scene in (traced, uncharged):
        scene.reset(0, DEV_BASE + 7)
    start = traced.data.site_xpos[traced._tips[0], :2].copy()
    gap = []
    while True:
        _, earned, done, _ = traced.step(np.zeros((1, 4)))
        _, kept, _, _ = uncharged.step(np.zeros((1, 4)))
        gap.append(kept[0] - earned[0])
        if done[0]:
            break
    counted = np.sum(gap) / s.weights["slide"] * s.slide_scale
    touching = np.array(traced.touching)
    moved = np.linalg.norm(np.diff(np.vstack([start, traced.tips]), axis=0), axis=1)
    on = np.flatnonzero(touching)
    path = moved[on[0]:on[-1] + 1].sum()
    lost = np.sum(touching[:-1] & ~touching[1:])
    assert lost > 100 and path > 0.5
    assert 0.9 * path <= counted <= path + 0.02
