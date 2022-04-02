from flask import Blueprint, render_template

dataset_report = Blueprint(
    'dataset_report',
    __name__,
    template_folder='templates',
    static_folder='static'
)


@dataset_report.route('/')
def index():
    return render_template('dataset_report/index.html')
