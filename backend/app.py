import logging

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from backend.config import Settings
from backend.routes.dataset import create_dataset_blueprint
from backend.routes.generative import create_generative_blueprint
from backend.routes.llm import create_llm_blueprint
from backend.routes.health import health_blueprint
from backend.routes.knowledge import create_knowledge_blueprint
from backend.routes.ml import create_ml_blueprint
from backend.routes.validation import create_validation_blueprint


def create_app(settings: Settings | None = None) -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    try:
        app_settings = settings or Settings.from_environment()
    except ValueError as error:
        raise RuntimeError(f"Invalid application configuration: {error}") from error

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    CORS(app, resources={r"/api/*": {"origins": list(app_settings.cors_origins)}})
    app.register_blueprint(health_blueprint, url_prefix="/api")
    app.register_blueprint(
        create_dataset_blueprint(app_settings), url_prefix="/api/dataset"
    )
    app.register_blueprint(
        create_llm_blueprint(app_settings), url_prefix="/api/llm"
    )
    app.register_blueprint(create_ml_blueprint(app_settings), url_prefix="/api/ml")
    app.register_blueprint(
        create_knowledge_blueprint(app_settings), url_prefix="/api/knowledge"
    )
    app.register_blueprint(
        create_generative_blueprint(app_settings), url_prefix="/api/generative"
    )
    app.register_blueprint(
        create_validation_blueprint(app_settings), url_prefix="/api/validation"
    )

    @app.errorhandler(404)
    def handle_not_found(_error):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(_error):
        return jsonify({"error": "Method not allowed for this URL."}), 405

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(_error):
        return jsonify({"error": "Request exceeds the 10 MiB limit."}), 413

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        app.logger.exception("Unhandled request error")
        return jsonify({"error": "An unexpected server error occurred."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
