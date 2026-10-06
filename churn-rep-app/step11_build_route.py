"""
STEP 11 -- Mock "today's route" for one sales rep (input for the mobile app).

ASSUMPTION (stated in the app): a route planner already exists. This step only imitates
its output so the app has something realistic to show:

  * the app shows every active account (all regions)
  * the rep's territory for today is Sao Paulo: the visit list = the VISITS highest-priority
    accounts within RADIUS_KM (by road) of the depot; further accounts are on other days
  * stop order: nearest-neighbour from the depot, improved with 2-opt (shortest loop)
  * distances: straight line x ROAD_FACTOR; drive time at CITY_SPEED (no clock times:
    the rep gets the stores and the order, not a timetable)
  * calls: the next "Save now" accounts in SP that are not on the route

Output: outputs/mobile_data.json
Run (after step 8):   python step11_build_route.py
"""
import json
import math
import os
import pandas as pd

import config

VISITS = 6
RADIUS_KM = 14
ROAD_FACTOR = 1.35          # roads are longer than a straight line
CITY_SPEED = 24             # km/h, Sao Paulo traffic
CALLS = 5
DEPOT = {"name": "Sao Paulo distribution centre", "lat": -23.5329, "lng": -46.6395}   # assumed start point
TODAY = "Mon 3 Sep 2018"


def km(a, b):
    r = 6371
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dp, dl = p2 - p1, math.radians(b["lng"] - a["lng"])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h)) * ROAD_FACTOR


def tour_length(pts):
    return sum(km(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def plan(stops):
    """Nearest neighbour from the depot, then 2-opt, returning to the depot."""
    left, path = stops[:], [DEPOT]
    while left:
        nxt = min(left, key=lambda s: km(path[-1], s))
        path.append(nxt); left.remove(nxt)
    path.append(DEPOT)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                new = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
                if tour_length(new) < tour_length(path) - 1e-9:
                    path, improved = new, True
    return path


def build():
    from step10_export_app_data import export
    _, accounts = export(n_active=None, n_winback=0, filename="all_active.json")
    with open(os.path.join(config.OUTPUT_DIR, "all_active.json"), encoding="utf-8") as fh:
        meta = json.load(fh)["meta"]
    accounts.sort(key=lambda a: a["rank"])
    sp = [a for a in accounts if a["state"] == "SP" and a["lat"] is not None]

    visit_list = [a for a in sp if km(DEPOT, a) <= RADIUS_KM][:VISITS]
    path = plan([{"id": a["id"], "lat": a["lat"], "lng": a["lng"]} for a in visit_list])

    stops, total_km, total_drive = [], 0.0, 0
    for i in range(1, len(path) - 1):
        d = km(path[i - 1], path[i]); drive = max(5, round(d / CITY_SPEED * 60))
        total_km += d; total_drive += drive
        stops.append({"id": path[i]["id"], "stop": i, "drive_min": drive, "km": round(d, 1)})
    back = km(path[-2], DEPOT); back_min = max(5, round(back / CITY_SPEED * 60))
    total_km += back; total_drive += back_min

    on_route = {s["id"] for s in stops}
    calls = [a["id"] for a in sp if a["id"] not in on_route and a["segment"].startswith("1")][:CALLS]

    out = {
        "meta": {**meta, "today": TODAY, "territory": "Sao Paulo",
                 "route_note": "Route order comes from an assumed route planner (mocked here)."},
        "depot": DEPOT,
        "route": {"stops": stops, "calls": calls, "total_km": round(total_km, 1), "total_drive_min": total_drive,
                  "return_km": round(back, 1), "return_min": back_min},
        "accounts": accounts,
    }
    path_out = os.path.join(config.OUTPUT_DIR, "mobile_data.json")
    with open(path_out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    return path_out, out


if __name__ == "__main__":
    p, out = build()
    r = out["route"]
    print(f"{len(out['accounts']):,} active accounts -> {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
    print(f"route: {len(r['stops'])} stops, {r['total_km']} km, {r['total_drive_min']} min driving")
    for s in r["stops"]:
        print(" ", s)
    print("  calls:", r["calls"])
