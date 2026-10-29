"""Concept vocabulary and ground-truth concept extraction for the semantic24 experiment (AESOP-style concepts).

A concept is an object and where it is: "bus stop on the sidewalk". The closed vocabulary below is shared by the
ground truth, OWL-ViT (which scores every object x predicate phrase) and the closed-vocabulary Qwen prompt.

Ground truth per frame comes from the recorded instance-segmentation image (R = semantic tag, G + 256 B = instance
id), the depth image and the camera pose: every instance with >= MIN_PX pixels gets
  object     from the actor's blueprint (spawned actors), else from the town's environment object whose bounding
             box holds the instance's 3D point (benches, bins, ...), else from the semantic tag (building, tree, ...)
  predicate  from the lane under the instance's ground-contact point (its bottom pixels, back-projected with depth):
             driving lane in a junction -> at the intersection, other road lanes -> on the road, sidewalk -> on the
             sidewalk, anything else / off the map -> beside the road. Scenery (SCENERY) is always beside the road;
             furniture, poles and signs in the kerb-side shoulder lane are on the sidewalk.
             Nothing in Town01/02 is "in the sky".
Instances no rule names keep object None ("unlabeled"); the Labels tab shows them so they can be named by hand.
Hand corrections live in data/semantic24/gt_overrides.json and are applied by effective_concepts(). Qwen's ground truth
starts from those concepts at each Qwen frame and has its own corrections (gt_qwen_overrides.json, qwen_concepts()).
"""
from __future__ import annotations

import math

import numpy as np

# --- vocabulary ---------------------------------------------------------------------------------------------

OBJECT_FAMILIES = {
    "vehicles": ["car", "van", "truck", "bus", "motorcycle", "bicycle"],
    "people": ["pedestrian", "cyclist"],
    "traffic control": ["traffic light", "speed limit sign", "pole", "street lamp"],
    "street furniture": ["trash can", "garbage bag", "recycling bin", "mailbox", "bus stop", "cardboard box", "bench", "plant pot",
                         "fire hydrant", "billboard", "kiosk", "bike rack", "chair", "table", "umbrella"],
    "barriers": ["fence", "wall", "guard rail"],
    "built / nature": ["building", "bridge", "tree", "grass", "water"],
    "anomaly": ["hay bale", "ambulance", "police car", "warning sign"],
}
OBJECTS = [o for objs in OBJECT_FAMILIES.values() for o in objs]
PREDICATES = ["on the road", "on the sidewalk", "at the intersection", "beside the road", "in the sky"]


def concept(obj: str, pred: str) -> str:
    return f"{obj} {pred}"


CONCEPTS = [concept(o, p) for o in OBJECTS for p in PREDICATES]

# AESOP prompt: robot description, monitoring task, current task, observed concepts, output instruction
DESCRIPTION = ("I am the runtime monitor for an autonomous car driving through a town. "
               "My job is to notice observations that differ from the car's normal driving experience. "
               "The car is currently following its route along the town's streets. "
               "The car's front camera observes: {concepts}. "
               "Describe whether this observation is normal or anomalous.")


def describe(concepts: list[str]) -> str:
    """The text embedded for one frame: sorted, de-duplicated concepts in the AESOP template."""
    items = sorted(set(concepts))
    return DESCRIPTION.format(concepts=", ".join(f"a {c}" for c in items) if items else "nothing in particular")


# --- blueprint / tag / environment-object names -> objects ------------------------------------------------

# spawned props (static.prop.<name>) per vocabulary word; also what run_semantic24.py places
PROPS = {
    "trash can": ["trashcan01", "trashcan02", "trashcan03", "trashcan04", "trashcan05", "bin"],
    "garbage bag": ["trashbag"],
    "recycling bin": ["glasscontainer", "clothcontainer"],
    "mailbox": ["mailbox"],
    "bus stop": ["busstop", "busstoplb"],
    "hay bale": ["haybale", "haybalelb"],
    "warning sign": ["warningaccident"],
}
PROP_OBJECT = {f"static.prop.{p}": obj for obj, props in PROPS.items() for p in props}
SPECIAL_VEHICLES = {"vehicle.ford.ambulance": "ambulance",
                    "vehicle.dodge.charger_police": "police car", "vehicle.dodge.charger_police_2020": "police car"}
VEHICLE_BASE = {"car": "car", "van": "van", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle", "bicycle": "cyclist"}

# CARLA 0.9.16 CityObjectLabel values
TAGS = {0: "none", 1: "roads", 2: "sidewalks", 3: "buildings", 4: "walls", 5: "fences", 6: "poles", 7: "traffic light",
        8: "traffic signs", 9: "vegetation", 10: "terrain", 11: "sky", 12: "pedestrians", 13: "rider", 14: "car", 15: "truck",
        16: "bus", 17: "train", 18: "motorcycle", 19: "bicycle", 20: "static", 21: "dynamic", 22: "other", 23: "water",
        24: "road lines", 25: "ground", 26: "bridge", 27: "rail track", 28: "guard rail"}
SURFACE_TAGS = {0, 1, 2, 11, 24, 25, 27}   # locations, not objects
TAG_OBJECT = {3: "building", 4: "wall", 5: "fence", 6: "pole", 7: "traffic light", 8: "speed limit sign", 9: "tree", 10: "grass",
              12: "pedestrian", 13: "cyclist", 14: "car", 15: "truck", 16: "bus", 18: "motorcycle", 19: "bicycle", 23: "water",
              26: "bridge", 28: "guard rail"}
ENV_TAGS = {6, 20, 21, 22}   # tags whose object is better named by the town's environment-object name
# environment-object name fragment (lower case) -> object; first match wins
ENV_NAME_OBJECT = [("citylamp", "street lamp"), ("artificiallight", "street lamp"), ("bench", "bench"), ("plantpot", "plant pot"),
                   ("treepot", "plant pot"), ("metromap", "billboard"), ("smallkiosk", "kiosk"), ("map_table", "kiosk"),
                   ("chair", "chair"), ("umbrella", "umbrella"), ("parasol", "umbrella"), ("table", "table"),
                   ("plant", "plant pot"), ("trashcan", "trash can"), ("promenadetrashca", "trash can"), ("bin", "trash can"),
                   ("cloth_container", "recycling bin"), ("container", "recycling bin"), ("creased", "cardboard box"),
                   ("box", "cardboard box"), ("garbage", "garbage bag"), ("trashbag", "garbage bag"), ("mailbox", "mailbox"), ("bus_stop", "bus stop"),
                   ("busstop", "bus stop"), ("firehydrant", "fire hydrant"), ("billboard", "billboard"),
                   ("streetcounter", "kiosk"), ("bikeparking", "bike rack")]
# scenery: always "beside the road" (where its base meets a sidewalk polygon is arbitrary); everything else is placed by the map
SCENERY = {"building", "wall", "fence", "tree", "grass", "water", "bridge", "guard rail"}
# things that move on the road; anything else standing in the narrow kerb-side Shoulder lane is on the sidewalk's edge
MOVERS = set(OBJECT_FAMILIES["vehicles"] + OBJECT_FAMILIES["people"]) | {"ambulance", "police car"}

MIN_PX = 250           # pixels for an instance to count as visible (about 16 x 16 of the 800 x 600 frame)
ENV_MATCH_M = 1.0      # how far outside an environment object's box its instance point may fall


def actor_object(type_id: str, base_type: str = "") -> str | None:
    """Vocabulary object of a spawned actor (its blueprint id and base_type attribute)."""
    if type_id in SPECIAL_VEHICLES:
        return SPECIAL_VEHICLES[type_id]
    if type_id in PROP_OBJECT:
        return PROP_OBJECT[type_id]
    if type_id.startswith("walker."):
        return "pedestrian"
    if type_id.startswith("vehicle."):
        return VEHICLE_BASE.get(base_type.lower(), "car")
    return None


def env_name_object(name: str) -> str | None:
    n = name.lower()
    return next((obj for frag, obj in ENV_NAME_OBJECT if frag in n), None)


# --- geometry -----------------------------------------------------------------------------------------------

def decode_depth(bgra: np.ndarray) -> np.ndarray:
    """CARLA depth image (B, G, R, A) -> metres along the camera's forward axis."""
    b, g, r = (bgra[..., i].astype(np.float64) for i in range(3))
    return (r + g * 256 + b * 65536) / (256**3 - 1) * 1000.0


def rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """UE4 rotation (degrees) as a matrix taking camera-local (forward, right, up) to world."""
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
                     [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
                     [sp, -cp * sr, cp * cr]])


def backproject(u: np.ndarray, v: np.ndarray, depth: np.ndarray, cam: dict, fov: float) -> np.ndarray:
    """World points (N, 3) of pixels (u, v) at their depth; cam = {x, y, z, pitch, yaw, roll} of the camera."""
    h, w = depth.shape
    f = w / (2 * math.tan(math.radians(fov) / 2))
    d = depth[v, u]
    local = np.stack([d, (u - w / 2) * d / f, -(v - h / 2) * d / f], axis=1)
    return local @ rotation_matrix(cam["pitch"], cam["yaw"], cam["roll"]).T + np.array([cam["x"], cam["y"], cam["z"]])


def predicate_at(cmap, x: float, y: float, z: float) -> tuple[str, str]:
    """(predicate, lane type) of a world point from the map's lanes."""
    import carla

    wp = cmap.get_waypoint(carla.Location(x, y, z), project_to_road=False, lane_type=carla.LaneType.Any)
    if wp is None:
        return "beside the road", "none"
    lt = str(wp.lane_type)
    if lt in ("Driving", "Bidirectional", "Shoulder", "Parking", "Biking", "Stop"):
        return ("at the intersection" if wp.is_junction else "on the road"), lt
    if lt == "Sidewalk":
        return "on the sidewalk", lt
    return "beside the road", lt


class EnvIndex:
    """The town's environment objects (from world.get_environment_objects, saved per town) for point lookups."""

    def __init__(self, objects: list[dict]):
        self.objects = [o for o in objects if o["type"] in ("Static", "Dynamic", "Other", "Poles")]
        self.centre = np.array([o["center"] for o in self.objects]) if self.objects else np.zeros((0, 3))
        self.extent = np.array([o["extent"] for o in self.objects]) if self.objects else np.zeros((0, 3))
        self.yaw = np.radians([o["yaw"] for o in self.objects]) if self.objects else np.zeros(0)

    def lookup(self, p: np.ndarray) -> dict | None:
        """The environment object whose (yaw-rotated) box holds p, within ENV_MATCH_M; the smallest if several."""
        if not len(self.objects):
            return None
        d = p[None, :2] - self.centre[:, :2]
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        lx, ly = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
        dz = p[2] - self.centre[:, 2]
        inside = ((np.abs(lx) <= self.extent[:, 0] + ENV_MATCH_M) & (np.abs(ly) <= self.extent[:, 1] + ENV_MATCH_M)
                  & (np.abs(dz) <= self.extent[:, 2] + ENV_MATCH_M))
        idx = np.flatnonzero(inside)
        if not len(idx):
            return None
        return self.objects[idx[np.argmin(self.extent[idx].prod(1))]]


def frame_instances(inst_bgra: np.ndarray, depth: np.ndarray, cam: dict, fov: float, cmap, env: EnvIndex, actors: dict) -> list[dict]:
    """Every visible instance of one frame with its object, predicate, pixel count and box.
    actors: {instance id (actor id & 0xFFFF): {"type_id", "base_type"}} of the run's spawned actors."""
    tag = inst_bgra[..., 2].astype(np.int64)
    iid = inst_bgra[..., 1].astype(np.int64) + inst_bgra[..., 0].astype(np.int64) * 256
    key = (tag << 16 | iid).ravel()
    order = np.argsort(key, kind="stable")
    keys, starts, counts = np.unique(key[order], return_index=True, return_counts=True)
    w = inst_bgra.shape[1]
    out = []
    for k, s, n in zip(keys.tolist(), starts.tolist(), counts.tolist()):
        t, i = k >> 16, k & 0xFFFF
        if n < MIN_PX or t in SURFACE_TAGS:
            continue
        pix = order[s : s + n]
        v, u = pix // w, pix % w
        bottom = v >= np.percentile(v, 95)   # ground-contact pixels: the instance's lowest 5% of rows
        ub, vb = u[bottom], v[bottom]
        mid = np.argsort(ub)[len(ub) // 2]
        ground = backproject(ub[mid : mid + 1], vb[mid : mid + 1], depth, cam, fov)[0]
        pred, lane = predicate_at(cmap, *ground)
        rec = {"id": i, "tag": TAGS.get(t, str(t)), "px": n, "box": [int(u.min()), int(v.min()), int(u.max()) + 1, int(v.max()) + 1],
               "xyz": [round(float(c), 2) for c in ground], "lane": lane, "predicate": pred}
        if i in actors:
            rec["object"], rec["src"] = actor_object(actors[i]["type_id"], actors[i].get("base_type", "")), f"actor:{actors[i]['type_id']}"
        elif t in ENV_TAGS:
            centre = backproject(u[n // 2 : n // 2 + 1], v[n // 2 : n // 2 + 1], depth, cam, fov)[0]
            hit = env.lookup(ground) or env.lookup(centre)
            rec["object"] = env_name_object(hit["name"]) if hit else None
            if t == 6 and rec["object"] != "street lamp":   # pole pixels: a lamp by name, else a pole (never a nearby prop)
                rec["object"] = "pole"
            rec["src"] = f"env:{hit['name']}" if hit else f"tag:{rec['tag']}"
        else:
            rec["object"], rec["src"] = TAG_OBJECT.get(t), f"tag:{rec['tag']}"
        if rec["object"] in SCENERY:
            rec["predicate"] = "beside the road"
        elif lane == "Shoulder" and rec["object"] and rec["object"] not in MOVERS:
            rec["predicate"] = "on the sidewalk"
        out.append(rec)
    return out


# --- hand corrections ----------------------------------------------------------------------------------------

def apply_overrides(instances: list[dict], t: int, run_overrides: dict) -> list[dict]:
    """Instances of tick t after the run's track-level then frame-level corrections; dropped ones are removed.
    run_overrides = {"track": {id: edit}, "frame": {t: {id: edit, "add": ["object|predicate", ...]}}},
    edit = {"object": str, "predicate": str, "drop": bool} (any subset)."""
    track = run_overrides.get("track", {})
    frame = run_overrides.get("frame", {}).get(str(t), {})
    out = []
    for rec in instances:
        r = dict(rec)
        for edit in (track.get(str(rec["id"])), frame.get(str(rec["id"]))):
            if edit:
                r.update({k: v for k, v in edit.items() if k in ("object", "predicate", "drop")})
                r["edited"] = True
        if not r.get("drop"):
            out.append(r)
    for item in frame.get("add", []):
        obj, _, pred = item.partition("|")
        out.append({"id": None, "object": obj, "predicate": pred, "src": "added", "edited": True})
    return out


def effective_concepts(instances: list[dict], t: int, run_overrides: dict) -> list[str]:
    """Sorted, de-duplicated concepts of tick t (named instances only) after the corrections."""
    return sorted({concept(r["object"], r["predicate"]) for r in apply_overrides(instances, t, run_overrides) if r.get("object")})


def qwen_concepts(base: list[str], t: int, run_qwen_overrides: dict) -> list[str]:
    """Qwen's ground truth at tick t: the (corrected) ground-truth concepts, then Qwen's own corrections,
    run_qwen_overrides = {t: {"remove": [concept, ...], "add": [concept, ...]}}."""
    edit = run_qwen_overrides.get(str(t), {})
    return sorted((set(base) - set(edit.get("remove", []))) | set(edit.get("add", [])))
