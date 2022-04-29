import json
from glob import glob
from pathlib import Path

from flask import Blueprint, render_template

dataset_report = Blueprint("dataset_report", __name__, template_folder="templates", static_folder="static")


@dataset_report.route("/")
def index():
    args = {"report_list": [Path(f).name for f in glob(str(Path(__file__).parent / "static/reports/*")) if Path(f).is_dir()]}
    return render_template("dataset_report/index.j2", **args)


@dataset_report.route("/report/<string:report_name>", methods=["GET"])
def report(report_name):
    meta_data = json.load(open(Path(__file__).parent / "static/reports" / report_name / "meta.json"))
    args = {
        "intertopic_distance_map": {
            "name": f"reports/{report_name}/topic_model/intertopic_distance_map/report.html",
            "width": meta_data["topic_model"]["intertopic_distance_map"]["width"] + 50,
            "height": meta_data["topic_model"]["intertopic_distance_map"]["height"] + 50,
        },
        "hierarchical_clustering": {
            "name": f"reports/{report_name}/topic_model/hierarchical_clustering/report.html",
            "width": meta_data["topic_model"]["hierarchical_clustering"]["width"] + 50,
            "height": meta_data["topic_model"]["hierarchical_clustering"]["height"] + 50,
        },
        "topic_word_score": {
            "name": f"reports/{report_name}/topic_model/barchart/report.html",
            "width": meta_data["topic_model"]["barchart"]["width"] + 50,
            "height": meta_data["topic_model"]["barchart"]["height"] + 50,
        },
        "similarity_matrix": {
            "name": f"reports/{report_name}/topic_model/similarity_matrix/report.html",
            "width": meta_data["topic_model"]["similarity_matrix"]["width"] + 50,
            "height": meta_data["topic_model"]["similarity_matrix"]["height"] + 50,
        },
        "topics_per_class": {
            "name": f"reports/{report_name}/topic_model/topics_per_class/report.html",
            "width": meta_data["topic_model"]["topics_per_class"]["width"] + 50,
            "height": meta_data["topic_model"]["topics_per_class"]["height"] + 50,
        },
        "topics": [{"index": idx, "topic": topic["name"]} for idx, topic in meta_data["topic_model"]["topics"].items()],
        "topic_prob_dist": [
            {
                "title": m["title"],
                "url": f"/dataset_report/static/reports/{report_name}/topic_model/topic_model_prob_dist/{m['htmlfile']}",
                "topics": m["topics"],
                "width": m["width"],
                "height": m["height"],
            }
            for m in meta_data["topic_model"]["topic_prob_dist"]
        ],
    }
    return render_template("dataset_report/report.j2", **args)
