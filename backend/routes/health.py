from flask import Blueprint, jsonify


health_blueprint = Blueprint("health", __name__)


@health_blueprint.get("/health")
def health():
    return jsonify({"status": "ok"})


@health_blueprint.get("/ready")
def readiness():
    return jsonify({"status": "ready"})
