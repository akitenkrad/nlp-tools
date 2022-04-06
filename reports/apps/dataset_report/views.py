import json
from glob import glob
from pathlib import Path

from flask import Blueprint, render_template

dataset_report = Blueprint(
    'dataset_report',
    __name__,
    template_folder='templates',
    static_folder='static'
)


@dataset_report.route('/')
def index():
    report_list = [Path(f).name for f in glob(str(Path(__file__).parent / 'static/reports'))]
    return render_template('dataset_report/index.html', report_list=report_list)


@dataset_report.route('/report/<string:report_name>', methods=['GET'])
def report(report_name):
    meta_json = json.load(open(Path(__file__).parent / 'static/reports' / report_name / 'meta.json'))
    args = {
        'intertopic_distance_map': f'reports/{report_name}/topic_model/intertopic_distance_map/report.html',
        'hierarchical_clustering': f'reports/{report_name}/topic_model/hierarchical_clustering/report.html',
        'topic_word_score': f'reports/{report_name}/topic_model/barchart/report.html',
        'similarity_matrix': f'reports/{report_name}/topic_model/similarity_matrix/report.html',
        'topics_per_class': f'reports/{report_name}/topic_model/topics_per_class/report.html',
        'topic_prob_dist_options': [{'text': m['text'], 'value': m['value']} for m in meta_json['topic_prob_dist']]
    }
    return render_template('dataset_report/report.html', **args)
