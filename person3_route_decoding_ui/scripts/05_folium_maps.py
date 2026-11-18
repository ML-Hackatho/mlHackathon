"""Folium maps for the report: GPS vs reference route vs classifier-only vs HMM route.

Student-testing trips only (the reference route is drawn here for analysis; it is never sent to the UI).
Usage: python scripts/05_folium_maps.py [trip_id ...]      (default: all 10 student-test trips)
Outputs: maps/decoded_<trip>.html
"""
import os
import sys

import folium
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.inference import RoutePredictor  # noqa: E402
from pathsnap.io import list_trips  # noqa: E402
from pathsnap.team import load_with_route  # noqa: E402
from pathsnap.metrics import collapse  # noqa: E402

MODEL = os.environ.get("PATHSNAP_MODEL", "mlp")


def arcs_to_lines(trip, arcs):
    return [[[trip.nodes[u, 1], trip.nodes[u, 0]], [trip.nodes[v, 1], trip.nodes[v, 0]]]
            for u, v in trip.arcs[arcs]]


def make_map(trip_id, predictor):
    d = os.path.join(C.STUDENT_TEST_DIR, trip_id)
    out, _ = predictor.predict(d)
    trip = load_with_route(d, "student_test")[1]     # with route, for the analysis map only
    gps = [[la, lo] for lo, la in trip.track[:, :2]]
    m = folium.Map(location=np.mean(gps, axis=0).tolist(), zoom_start=14, tiles="OpenStreetMap")

    net = folium.FeatureGroup("Road network (candidate area)", show=True)
    for seg in out["network_segments"]:
        folium.PolyLine(seg, color="#9aa5b1", weight=2, opacity=0.6).add_to(net)
    net.add_to(m)

    ref = folium.FeatureGroup("Reference route (ground truth)", show=True)
    for seg in arcs_to_lines(trip, trip.route):
        folium.PolyLine(seg, color="#2ca02c", weight=9, opacity=0.45).add_to(ref)
    ref.add_to(m)

    clf = folium.FeatureGroup("Classifier-only Top-1 arcs", show=False)
    for seg in arcs_to_lines(trip, collapse(np.asarray(out["classifier_only_arcs_per_point"]))):
        folium.PolyLine(seg, color="#9467bd", weight=4, opacity=0.8).add_to(clf)
    clf.add_to(m)

    hmm = folium.FeatureGroup("Predicted route (HMM/Viterbi)", show=True)
    folium.PolyLine(out["predicted_route_coords"], color="#d62728", weight=4, opacity=0.9).add_to(hmm)
    hmm.add_to(m)

    g = folium.FeatureGroup("GPS observations", show=True)
    step = max(1, len(gps) // 2500)
    for p in gps[::step]:
        folium.CircleMarker(p, radius=2, color="#1f77b4", fill=True, fill_opacity=0.8, weight=0).add_to(g)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(g)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(g)
    g.add_to(m)

    legend = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
         padding: 10px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px; line-height: 20px;">
      <b>Trip %s</b><br>
      <span style="color:#1f77b4">&#9679;</span> GPS observations<br>
      <span style="color:#2ca02c">&#9644;&#9644;</span> Reference route<br>
      <span style="color:#d62728">&#9644;&#9644;</span> HMM/Viterbi predicted route<br>
      <span style="color:#9467bd">&#9644;&#9644;</span> Classifier-only (toggle)<br>
      <span style="color:#9aa5b1">&#9644;&#9644;</span> Road network
    </div>""" % trip_id
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([np.min(gps, axis=0).tolist(), np.max(gps, axis=0).tolist()])
    path = os.path.join(C.MAPS, f"decoded_{trip_id}.html")
    m.save(path)
    print("saved", path)


if __name__ == "__main__":
    pred = RoutePredictor(os.path.join(C.MODELS, f"{MODEL}.joblib"), os.path.join(C.MODELS, "decoder_config.json"))
    for tid in (sys.argv[1:] or list_trips(C.STUDENT_TEST_DIR)):
        make_map(tid, pred)
