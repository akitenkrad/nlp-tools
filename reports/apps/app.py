from flask import Flask


def create_app():
    app = Flask(__name__)

    from apps.dataset_report import views
    app.register_blueprint(views.dataset_report, url_prefix='/dataset_report')

    return app
