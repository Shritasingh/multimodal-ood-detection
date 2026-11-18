"""Record the snapshots (sim.plans) with CARLA (running on 127.0.0.1:2000), then the droplet frames.

    .venv/bin/python mvp_recreate/record.py                  # every missing observation
    .venv/bin/python mvp_recreate/record.py --force          # re-record everything
    .venv/bin/python mvp_recreate/record.py --derived-only   # only rebuild the droplet frame (no CARLA)

Per observation, data/<dataset>/<group>/: frames/<idx>.jpg and states/<idx>.json:
  plan        the FramePlan
  history     `history` states, oldest first, each {tick, ego, others}. The ego is parked; the one other agent's states
              assume constant velocity with the heading along its path (the cyclist: along its route at its speed;
              the pedestrian: back from its actual pose at the frame along its heading). On the last state: px
              (instance pixels), pred (predicate), box (2D pixel box of its instance mask)
  agents      id -> {type (cyclist / pedestrian), obj (the semantic object phrase), bp, bb (bounding box extent xyz +
              centre offset xyz, actor frame)}
  concepts    "<obj> <pred>" of every agent with >= visible_px pixels on the frame
  shift       whether the observation is an anomaly
Each droplet frame copies its nominal source's states (so its physics and semantic inputs are identical) and blurs
irregular, asymmetric blobs of the image (like water drops on the lens) over its most detailed spots (they may cover the agent).
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import carla  # noqa: E402

import sim as S  # noqa: E402
from cfg import DATA  # noqa: E402


def out_dir(cache: str, group: str) -> Path:
    return DATA / cache / group


def done(p: S.FramePlan) -> bool:
    d = out_dir(p.cache, p.group)
    return (d / "states" / f"{p.idx:04d}.json").exists() and (d / "frames" / f"{p.idx:04d}.jpg").exists()


def write(p: S.FramePlan, img: np.ndarray, rec: dict) -> None:
    d = out_dir(p.cache, p.group)
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "states").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "frames" / f"{p.idx:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (d / "states" / f"{p.idx:04d}.json").write_text(json.dumps(rec))


class Recorder:
    def __init__(self):
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(120.0)
        world = self.client.get_world()
        if not world.get_map().name.endswith(S.TOWN):
            world = self.client.load_world(S.TOWN)
        self.world, self.m = world, world.get_map()
        self.world = world
        self._clear()                                     # e.g. the ego / camera / agent of a run that was killed
        S.strip_world(world)
        world.set_weather(S.weather())
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, S.FIXED_DELTA
        world.apply_settings(settings)
        self.g = S.Geometry(self.m)
        self.bpl = world.get_blueprint_library()
        e = self.g.ego_tf
        self.ego = self._kinematic(S.EGO_BP, 0)
        self.ego_tf = carla.Transform(carla.Location(e.location.x, e.location.y, e.location.z), carla.Rotation(yaw=e.rotation.yaw))
        self.ego.set_transform(self.ego_tf)
        self.rgb, self.q_rgb = self._camera("sensor.camera.rgb")
        self.inst, self.q_inst = self._camera("sensor.camera.instance_segmentation")
        for _ in range(S._SC.get("camera_warmup_ticks", 0)):   # let the camera's auto-exposure settle: without this the
            f = world.tick()                                     # first ~4 frames of every run come out over-bright
            self._frame(self.q_rgb, f), self._frame(self.q_inst, f)

    def _clear(self, keep: tuple = ()) -> None:
        """Destroy every vehicle / walker / controller / sensor but `keep`, and check they are gone: a leftover agent
        would be in every frame. Raises if any survive 3 tries."""
        w = self.world
        for _ in range(3):
            left = [a for f in ("vehicle.*", "walker.*", "controller.*", "sensor.*") for a in w.get_actors().filter(f) if a.id not in keep]
            if not left:
                return
            for a in left:
                if a.type_id.startswith("sensor.") and a.is_listening:
                    a.stop()
            sync = w.get_settings().synchronous_mode
            self.client.apply_batch_sync([carla.command.DestroyActor(a) for a in left], sync)
            if not sync:
                w.wait_for_tick()
        raise RuntimeError(f"leftover actors survive: {[(a.id, a.type_id) for a in left]}")

    def close(self) -> None:
        for c in (self.rgb, self.inst):
            c.stop()
        self.client.apply_batch_sync([carla.command.DestroyActor(a) for a in (self.rgb, self.inst, self.ego)])
        s = self.world.get_settings()
        s.synchronous_mode = False
        self.world.apply_settings(s)

    def _kinematic(self, bp_name: str, k: int):
        bp = self.bpl.find(bp_name)
        if bp.has_attribute("color"):
            bp.set_attribute("color", "120,120,130")
        a = self.world.spawn_actor(bp, carla.Transform(carla.Location(0.0, 0.0, 200.0 + 8.0 * k)))
        a.set_simulate_physics(False)
        return a

    def _camera(self, kind: str):
        bp = self.bpl.find(kind)
        for k, v in (("image_size_x", S.CAMERA["width"]), ("image_size_y", S.CAMERA["height"]), ("fov", S.CAMERA["fov"])):
            bp.set_attribute(k, str(v))
        cam = self.world.spawn_actor(bp, carla.Transform(carla.Location(x=S.CAMERA["x"], z=S.CAMERA["z"])), attach_to=self.ego)
        q: queue.Queue = queue.Queue()
        cam.listen(q.put)
        return cam, q

    @staticmethod
    def _frame(q: queue.Queue, frame: int):
        while True:
            img = q.get(timeout=20.0)
            if img.frame == frame:
                return img

    def record(self, p: S.FramePlan) -> str:
        """Render one snapshot and write it; returns a warning ('' if none)."""
        self._clear(keep=(self.ego.id, self.rgb.id, self.inst.id))   # only the ego and its cameras before each frame
        g, w = self.g, self.world
        actors, warn = [], ""
        agent = kind = obj = None
        try:
            if p.bike:
                a = self._kinematic(S.BIKE_BP, 1)
                actors.append(a)
                x, y, z, yaw = S.bike_pose(g, p.bike, 0)
                gp = w.ground_projection(carla.Location(x, y, z + 2.0), 6.0)   # on the ground: the curb / sidewalk too
                if gp is not None:
                    z = gp.location.z
                a.set_transform(carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw=yaw)))
                agent, kind, obj = a, "cyclist", "cyclist"
            elif p.ped:
                q = p.ped
                bp = self.bpl.find(q["bp"])
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                sx, sy = S.ped_back(q, S.SETTLE_TICKS * 0.8)      # walkers start a little behind: they accelerate
                a = w.try_spawn_actor(bp, carla.Transform(carla.Location(sx, sy, q["z"] + 1.0), carla.Rotation(yaw=q["yaw"])))
                if a is None:
                    return "walker spawn failed"
                actors.append(a)
                ay = math.radians(q["yaw"])
                ctrl = carla.WalkerControl(carla.Vector3D(math.cos(ay), math.sin(ay), 0.0), q["speed"])
                agent, kind = a, "pedestrian"
                obj = "traffic conductor" if q["bp"] == S.CONDUCTOR_BP else "pedestrian"
            frame = None
            for _ in range(S.SETTLE_TICKS):
                self.ego.set_transform(self.ego_tf)
                if kind == "pedestrian":
                    agent.apply_control(ctrl)
                frame = w.tick()
            img, seg = self._frame(self.q_rgb, frame), self._frame(self.q_inst, frame)

            # history: constant velocity, heading along the path
            el = self.ego_tf.location
            ego = {"id": self.ego.id, "x": round(el.x, 3), "y": round(el.y, 3), "z": round(el.z, 3),
                   "yaw": round(math.radians(self.ego_tf.rotation.yaw), 5), "vx": 0.0, "vy": 0.0}
            hist = []
            if agent is None:          # the empty frame
                hist = [{"tick": S.HISTORY - 1 - k, "ego": ego, "others": []} for k in range(S.HISTORY - 1, -1, -1)]
            if kind == "pedestrian":
                t = agent.get_transform()
                ax, ay_, az = t.location.x, t.location.y, t.location.z
                if math.hypot(ax - q["x"], ay_ - q["y"]) > 1.0:
                    warn = f"walker {math.hypot(ax - q['x'], ay_ - q['y']):.2f} m from its planned spot"
                yaw = q["yaw"] if q["speed"] > 0 else t.rotation.yaw
            for k in range(S.HISTORY - 1, -1, -1) if agent is not None else ():
                if kind == "cyclist":
                    x, y, z, yw = S.bike_pose(g, p.bike, k)
                    sp = p.bike["speed"]
                else:
                    a_ = math.radians(q["yaw"])
                    d = q["speed"] * S.FIXED_DELTA * k
                    x, y, z, yw, sp = ax - d * math.cos(a_), ay_ - d * math.sin(a_), az, yaw, q["speed"]
                o = {"id": agent.id, "x": round(x, 3), "y": round(y, 3), "z": round(z, 3), "yaw": round(math.radians(yw), 5),
                     "vx": round(sp * math.cos(math.radians(yw)), 3), "vy": round(sp * math.sin(math.radians(yw)), 3)}
                hist.append({"tick": S.HISTORY - 1 - k, "ego": ego, "others": [o]})

            arr = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3]
            if agent is None:
                write(p, arr, {"name": f"{p.cache}/{p.group}/{p.idx:04d}", "plan": p.to_dict(), "history": hist, "agents": {},
                               "concepts": [], "shift": p.group != "nominal"})
                return ""
            sa = np.frombuffer(seg.raw_data, np.uint8).reshape(seg.height, seg.width, 4)
            inst = sa[..., 1].astype(np.int32) + sa[..., 0].astype(np.int32) * 256
            mask = inst == (agent.id & 0xFFFF)
            last = hist[-1]["others"][0]
            last["px"] = int(mask.sum())
            last["pred"] = S.predicate(self.m, g.cws, last["x"], last["y"], last["z"])
            if mask.any():
                ys, xs = np.nonzero(mask)
                last["box"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if S.SHIFTED.get(p.group) and last["px"] < S.VISIBLE_PX:
                warn += f" shifted agent only {last['px']} px"
            bb = agent.bounding_box
            agents = {str(agent.id): {"type": kind, "obj": obj, "bp": agent.type_id,
                                      "bb": [round(v, 3) for v in (bb.extent.x, bb.extent.y, bb.extent.z, bb.location.x, bb.location.y, bb.location.z)]}}
            concepts = [f"{obj} {last['pred']}"] if last["px"] >= S.VISIBLE_PX else []
            write(p, arr, {"name": f"{p.cache}/{p.group}/{p.idx:04d}", "plan": p.to_dict(), "history": hist, "agents": agents,
                           "concepts": concepts, "shift": p.group != "nominal"})
            return warn.strip()
        finally:
            self.client.apply_batch_sync([carla.command.DestroyActor(a) for a in actors])
            w.tick()

    def record_walk(self, plans: list[S.FramePlan]) -> list[str]:
        """The walking pedestrian frames of ONE walker in ONE direction, from one continuous run: the walker walks the
        crosswalk slowly (pedestrian.record_speed) from run_up_m before its first spot, and each frame is the tick
        nearest its planned spot. At a slow speed that tick is within record_speed * fixed_delta / 2 of the spot, and the
        equally spaced spots are passed at equal time steps. The physics history still uses the plan's speed (the
        nominal walking speed), back from the walker's actual pose along its heading, so the encoding is the same as
        record(). Returns one warning per plan ('' if none)."""
        self._clear(keep=(self.ego.id, self.rgb.id, self.inst.id))
        w, P = self.world, S._PE
        q0 = plans[0].ped
        u = np.array([math.cos(math.radians(q0["yaw"])), math.sin(math.radians(q0["yaw"]))])
        along = lambda x, y: float(np.array([x, y]) @ u)  # noqa: E731
        plans = sorted(plans, key=lambda p: along(p.ped["x"], p.ped["y"]))
        first, v = plans[0].ped, P["record_speed"]
        sx, sy = first["x"] - u[0] * P["run_up_m"], first["y"] - u[1] * P["run_up_m"]
        bp = self.bpl.find(first["bp"])
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        a = None
        for dz in (1.0, 1.3, 1.6):          # a run-up start on the sidewalk curb collides at the road's z + 1
            a = w.try_spawn_actor(bp, carla.Transform(carla.Location(sx, sy, first["z"] + dz), carla.Rotation(yaw=first["yaw"])))
            if a is not None:
                break
        if a is None:
            return ["walker spawn failed"] * len(plans)
        ctrl = carla.WalkerControl(carla.Vector3D(float(u[0]), float(u[1]), 0.0), v)
        span = along(plans[-1].ped["x"], plans[-1].ped["y"]) - along(sx, sy)
        warns, k, prev = [], 0, None
        try:
            for _ in range(int(3 * (span + 1.0) / v / S.FIXED_DELTA)):
                self.ego.set_transform(self.ego_tf)
                a.apply_control(ctrl)
                frame = w.tick()
                img, seg = self._frame(self.q_rgb, frame), self._frame(self.q_inst, frame)
                t = a.get_transform()
                cur = (along(t.location.x, t.location.y), t,
                       np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy(),
                       np.frombuffer(seg.raw_data, np.uint8).reshape(seg.height, seg.width, 4).copy())
                target = along(plans[k].ped["x"], plans[k].ped["y"])
                if prev is not None and cur[0] >= target:
                    best = cur if abs(cur[0] - target) <= abs(prev[0] - target) else prev
                    warns.append(self._write_walker(plans[k], a, best[1], best[2], best[3]))
                    k += 1
                    if k == len(plans):
                        break
                prev = cur
            warns += ["the run ended before this spot"] * (len(plans) - k)
            return warns
        finally:
            self.client.apply_batch_sync([carla.command.DestroyActor(a)])
            w.tick()

    def _write_walker(self, p: S.FramePlan, agent, t: carla.Transform, arr: np.ndarray, sa: np.ndarray) -> str:
        """Write one walking-pedestrian snapshot (record_walk): the walker's pose t, the camera images at that tick."""
        q, g, warn = p.ped, self.g, ""
        ax, ay_, az = t.location.x, t.location.y, t.location.z
        off = math.hypot(ax - q["x"], ay_ - q["y"])
        if off > 0.1:
            warn = f"walker {off:.2f} m from its planned spot"
        el = self.ego_tf.location
        ego = {"id": self.ego.id, "x": round(el.x, 3), "y": round(el.y, 3), "z": round(el.z, 3),
               "yaw": round(math.radians(self.ego_tf.rotation.yaw), 5), "vx": 0.0, "vy": 0.0}
        a_, hist = math.radians(q["yaw"]), []
        for k in range(S.HISTORY - 1, -1, -1):     # the plan's (nominal) speed, not the slow recording speed
            d = q["speed"] * S.FIXED_DELTA * k
            hist.append({"tick": S.HISTORY - 1 - k, "ego": ego, "others": [
                {"id": agent.id, "x": round(ax - d * math.cos(a_), 3), "y": round(ay_ - d * math.sin(a_), 3), "z": round(az, 3),
                 "yaw": round(a_, 5), "vx": round(q["speed"] * math.cos(a_), 3), "vy": round(q["speed"] * math.sin(a_), 3)}]})
        inst = sa[..., 1].astype(np.int32) + sa[..., 0].astype(np.int32) * 256
        mask = inst == (agent.id & 0xFFFF)
        last = hist[-1]["others"][0]
        last["px"] = int(mask.sum())
        last["pred"] = S.predicate(self.m, g.cws, last["x"], last["y"], last["z"])
        if mask.any():
            ys, xs = np.nonzero(mask)
            last["box"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        bb = agent.bounding_box
        agents = {str(agent.id): {"type": "pedestrian", "obj": "pedestrian", "bp": agent.type_id,
                                  "bb": [round(v, 3) for v in (bb.extent.x, bb.extent.y, bb.extent.z, bb.location.x, bb.location.y, bb.location.z)]}}
        concepts = [f"pedestrian {last['pred']}"] if last["px"] >= S.VISIBLE_PX else []
        write(p, arr, {"name": f"{p.cache}/{p.group}/{p.idx:04d}", "plan": p.to_dict(), "history": hist, "agents": agents,
                       "concepts": concepts, "shift": p.group != "nominal", "recorded_speed": S._PE["record_speed"]})
        return warn


def blob_mask(h: int, w: int, cx: float, cy: float, r: float, rng: np.random.RandomState) -> np.ndarray:
    """(h, w) soft mask of an irregular blob: a circle of radius r whose outline wobbles (harmonics 2-5), stretched along
    a random axis, with a feathered edge."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rot = rng.uniform(0, np.pi)
    u, v = (xx - cx) * np.cos(rot) + (yy - cy) * np.sin(rot), -(xx - cx) * np.sin(rot) + (yy - cy) * np.cos(rot)
    stretch = rng.uniform(1.3, 1.7)
    u, v = u / stretch, v / (1.0 / stretch ** 0.5)
    ang, dist = np.arctan2(v, u), np.hypot(u, v)
    edge = np.full_like(ang, r / stretch ** 0.25)
    for k in range(2, 6):
        edge *= 1 + rng.uniform(0.06, 0.16) * np.cos(k * ang + rng.uniform(0, 2 * np.pi))
    feather = 0.25 * r
    return np.clip((edge - dist) / feather + 0.5, 0, 1).astype(np.float32)


def place_blobs(img: np.ndarray) -> list[dict]:
    """len(radii_px) irregular blobs placed one by one over the most detailed spots of the image (edges: stripes, curbs,
    the horizon, the agent), each fully inside the image and clear of the earlier blobs (centres >= (r_a + r_b) * min_sep
    apart); blob k has seed + k. [{cx, cy, r, seed}]"""
    h, w = img.shape[:2]
    D = S.DROPLET
    detail = np.abs(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F, ksize=3))
    detail = cv2.GaussianBlur(detail, (0, 0), 3)            # suppress pixel noise: count structure (stripes, curbs, edges)
    drops = []
    for k, r in enumerate(D["radii_px"]):
        seed = D["seed"] + k
        shape = blob_mask(h, w, w / 2, h / 2, r, np.random.RandomState(seed))           # this blob at the image centre
        ys, xs = np.nonzero(shape > 0.01)
        hx, hy = int(max(w / 2 - xs.min(), xs.max() - w / 2)) + 2, int(max(h / 2 - ys.min(), ys.max() - h / 2)) + 2
        best = None
        for cx in range(hx, w - hx, 10):                     # fully inside the image
            for cy in range(hy, h - hy, 10):
                if any(math.hypot(cx - d["cx"], cy - d["cy"]) < (r + d["r"]) * D["min_sep"] for d in drops):
                    continue                                 # clear of the earlier blobs
                sc = float(detail[cy - hy:cy + hy, cx - hx:cx + hx].sum())
                if best is None or sc > best[0]:
                    best = (sc, cx, cy)
        _, cx, cy = best
        drops.append({"cx": cx, "cy": cy, "r": r, "seed": seed})
    return drops


def make_droplet(p: S.FramePlan) -> str:
    """Copy the source frame's states and blur the blobs of place_blobs (water drops on the lens) into its image. With
    droplet.blobs_from set, every dataset's droplet frame reuses the blobs placed on THAT dataset's droplet source (the
    same camera, so the same image spots). They may cover the agent; the frame keeps its source's labels."""
    src = out_dir(p.src["cache"], p.src["group"])
    rec = json.loads((src / "states" / f"{p.src['idx']:04d}.json").read_text())
    img = cv2.imread(str(src / "frames" / f"{p.src['idx']:04d}.jpg"))
    h, w = img.shape[:2]
    D = S.DROPLET
    ref = S.DROP_FROM.get(D.get("blobs_from") or p.src["cache"], tuple(p.src.values()))
    drops = place_blobs(cv2.imread(str(out_dir(ref[0], ref[1]) / "frames" / f"{ref[2]:04d}.jpg")))
    mask = np.zeros((h, w), np.float32)
    for d in drops:
        m = blob_mask(h, w, d["cx"], d["cy"], d["r"], np.random.RandomState(d["seed"]))
        mask = np.maximum(mask, m)
        d["px"] = int((m > 0.5).sum())
    blur = cv2.GaussianBlur(img, (0, 0), D["blur_sigma"]).astype(np.float32)
    out = np.clip(img.astype(np.float32) * (1 - mask[..., None]) + blur * mask[..., None], 0, 255).astype(np.uint8)
    box = rec["history"][-1]["others"][0].get("box") if rec["history"][-1]["others"] else None
    on_agent = float(mask[box[1]:box[3] + 1, box[0]:box[2] + 1].mean()) if box else 0.0
    rec.update(name=f"{p.cache}/{p.group}/{p.idx:04d}", plan={**p.to_dict(), "drop": {"blobs": drops, "blobs_from": f"{ref[0]}/{ref[1]}/{ref[2]}", "blur_sigma": D["blur_sigma"],
               "px": int((mask > 0.5).sum()), "agent_box_covered": round(on_agent, 3)}}, shift=True)
    write(p, out, rec)
    return f"{len(drops)} blobs, {int((mask > 0.5).sum())} px; agent box {on_agent:.0%} covered"


def write_geometry(g: S.Geometry) -> None:
    """For the dashboard's top-down view: crosswalks and every lane route through the junction (world x, y), the camera."""
    e = g.ego_tf
    geo = {"crosswalks": [np.round(q, 2).tolist() for q in g.cws],
           "routes": {n: np.round(r.xyz[::4, :2], 2).tolist() for n, r in g.routes.items()},
           "ego": {"x": e.location.x, "y": e.location.y, "yaw": math.radians(e.rotation.yaw)},
           "camera": {"x": S.CAMERA["x"], "fov": S.CAMERA["fov"]}}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "_geometry.json").write_text(json.dumps(geo))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--derived-only", action="store_true", help="only rebuild the frames made from recorded ones (no CARLA)")
    args = ap.parse_args()
    if args.derived_only:
        for p in S.derived_plans():
            print(f"{p.cache}/{p.group}/{p.idx:04d} {p.note}: {make_droplet(p)}", flush=True)
        return
    if args.force:
        for cache in S.CACHES:
            for group in S.CACHES[cache]:
                if out_dir(cache, group).exists():
                    shutil.rmtree(out_dir(cache, group))
    rec = Recorder()
    try:
        write_geometry(rec.g)
        all_plans = S.plans(rec.g)
        walks: dict[tuple, list] = {}            # walking pedestrians: one slow run per walker x direction (record_walk)
        for p in all_plans:
            if p.src is None and not done(p):
                if p.ped and p.ped["speed"] > 0 and S._PE.get("record_speed"):
                    walks.setdefault((p.cache, p.group, p.ped["bp"], p.ped["orient"]), []).append(p)
                    continue
                why = rec.record(p)
                print(f"{p.cache}/{p.group}/{p.idx:04d} {p.note}: {'ok' if not why else why}", flush=True)
        for ps in walks.values():
            for p, why in zip(sorted(ps, key=lambda p: p.ped["t"] * (1 if p.ped["orient"] == "walk_right" else -1)), rec.record_walk(ps)):
                print(f"{p.cache}/{p.group}/{p.idx:04d} {p.note}: {'ok' if not why else why}", flush=True)
    finally:
        rec.close()
    for p in all_plans:
        if p.src is not None:
            print(f"{p.cache}/{p.group}/{p.idx:04d} {p.note}: {make_droplet(p)}", flush=True)


if __name__ == "__main__":
    main()
