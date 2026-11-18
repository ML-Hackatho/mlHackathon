"""Create a layered Folium map for an XGBoost student-test prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import folium
import numpy as np

from pathsnap_person1.io import load_trip
from pathsnap_person2.inference import RoutePredictor
from pathsnap_person2.decoder import collapse_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--trip-id", default="00000000")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.project_root.resolve()
    data_root = (args.data_root or root / "data" / "Student folder").resolve()
    trip_dir = data_root / "student_test" / args.trip_id
    model_path = root / "outputs/person2/models/final_model.joblib"
    decoder_path = root / "outputs/person2/models/decoder_config.json"

    # Prediction reads .track/.nodes/.arcs only. The reference route is loaded
    # afterward for the report visualization, never for inference.
    prediction, _ = RoutePredictor(model_path, decoder_path).predict(
        trip_dir, partition="student_test", include_network=True
    )
    trip = load_trip(trip_dir, "student_test", include_route=True)
    if trip.route is None:
        raise ValueError(f"Trip {args.trip_id} has no reference route for map layers")

    gps = prediction["gps"]
    map_view = folium.Map(location=np.mean(np.asarray(gps), axis=0).tolist(), zoom_start=13,
                          tiles="OpenStreetMap", control_scale=True)

    network = folium.FeatureGroup(name="Road network (candidate area)", show=True)
    for segment in prediction["network_segments"]:
        folium.PolyLine(segment, color="#97a3af", weight=2, opacity=0.45).add_to(network)
    network.add_to(map_view)

    reference = folium.FeatureGroup(name="Reference route (analysis only)", show=True)
    for arc_id in trip.route:
        start, end = trip.arcs[int(arc_id)]
        coords = [[float(trip.nodes[start, 1]), float(trip.nodes[start, 0])],
                  [float(trip.nodes[end, 1]), float(trip.nodes[end, 0])]]
        folium.PolyLine(coords, color="#1b9e77", weight=7, opacity=0.55).add_to(reference)
    reference.add_to(map_view)

    classifier = folium.FeatureGroup(name="Classifier-only Top-1 route", show=False)
    classifier_arcs = collapse_sequence(np.asarray(prediction["classifier_only_arcs_per_point"], dtype=np.int64))
    for arc_id in classifier_arcs:
        if int(arc_id) < 0:
            continue
        start, end = trip.arcs[int(arc_id)]
        coords = [[float(trip.nodes[start, 1]), float(trip.nodes[start, 0])],
                  [float(trip.nodes[end, 1]), float(trip.nodes[end, 0])]]
        folium.PolyLine(coords, color="#9467bd", weight=4, opacity=0.7).add_to(classifier)
    classifier.add_to(map_view)

    predicted = folium.FeatureGroup(name="Predicted route (XGBoost + HMM/Viterbi)", show=True)
    folium.PolyLine(prediction["predicted_route_coords"], color="#e34a33", weight=4, opacity=0.95).add_to(predicted)
    predicted.add_to(map_view)

    observations = folium.FeatureGroup(name="GPS observations", show=True)
    stride = max(1, len(gps) // 2200)
    for point in gps[::stride]:
        folium.CircleMarker(point, radius=2, color="#3182bd", fill=True,
                            fill_color="#3182bd", fill_opacity=0.8, weight=0).add_to(observations)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(observations)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(observations)
    observations.add_to(map_view)

    legend = f"""
    <div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999; background: white;
         padding: 9px 12px; border: 1px solid #999; border-radius: 6px; font-size: 12px; line-height: 18px;">
      <b>Student-test trip {args.trip_id}</b><br>
      <span style="color:#3182bd">&#9679;</span> GPS observations<br>
      <span style="color:#1b9e77">&#9644;&#9644;</span> Reference route (report analysis only)<br>
      <span style="color:#e34a33">&#9644;&#9644;</span> XGBoost + HMM/Viterbi prediction<br>
      <span style="color:#9467bd">&#9644;&#9644;</span> Classifier-only (toggle)<br>
      <span style="color:#97a3af">&#9644;&#9644;</span> Candidate road network
    </div>"""
    map_view.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(map_view)
    map_view.fit_bounds([np.min(np.asarray(gps), axis=0).tolist(), np.max(np.asarray(gps), axis=0).tolist()])

    output = args.output or root / "reports/technical_report_assets/folium_xgboost_trip_00000000.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    map_view.save(output)
    print(output)


if __name__ == "__main__":
    main()
