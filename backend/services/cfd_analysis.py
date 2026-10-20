"""Read OpenFOAM field metadata and calculate exact partition-wide statistics."""

import math
import re
from pathlib import Path
from typing import Any


NUMBER = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
FIELD_CLASS = re.compile(r"\bclass\s+(vol|surface|point)(Scalar|Vector|SymmTensor|Tensor)Field\b")
DIMENSIONS = re.compile(r"\bdimensions\s*\[([^\]]+)\]")
INTERNAL_NONUNIFORM = re.compile(
    r"\binternalField\s+nonuniform\s+List<([^>]+)>\s+(\d+)\s*\(",
    re.DOTALL,
)
INTERNAL_EMPTY_NONUNIFORM = re.compile(
    r"\binternalField\s+nonuniform\s+0\s*\(\s*\)\s*;",
    re.DOTALL,
)
INTERNAL_UNIFORM = re.compile(r"\binternalField\s+uniform\s+([^;]+);", re.DOTALL)
PROCESSOR_DIRECTORY = re.compile(r"processor\d+$")
SUPPORTED_CLASSES = {
    "volScalarField",
    "volVectorField",
    "volSymmTensorField",
    "volTensorField",
    "surfaceScalarField",
}
COMPONENTS_BY_VALUE_TYPE = {
    "scalar": 1,
    "vector": 3,
    "symmTensor": 6,
    "tensor": 9,
}
NORMALIZED_VALUE_TYPES = {
    "Scalar": "scalar",
    "Vector": "vector",
    "SymmTensor": "symmTensor",
    "Tensor": "tensor",
}
KNOWN_UNITS = {
    "0 0 0 1 0 0 0": "K",
    "1 -1 -2 0 0 0 0": "Pa",
    "0 1 -1 0 0 0 0": "m/s",
    "1 0 -3 -1 0 0 0": "W/(m^2 K)",
    "1 1 -3 0 0 0 0": "W",
}


class CfdDataError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _processor_directories(case_path: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in case_path.iterdir()
            if path.is_dir() and PROCESSOR_DIRECTORY.fullmatch(path.name)
        ),
        key=lambda path: int(path.name.removeprefix("processor")),
    )


def _common_time_name(processors: list[Path]) -> str:
    common: set[str] | None = None
    for processor in processors:
        times = {
            child.name
            for child in processor.iterdir()
            if child.is_dir() and _is_numeric_time(child.name)
        }
        common = times if common is None else common & times
    if not common:
        raise CfdDataError(
            "No common numeric solution time was found across the processor partitions.",
            422,
        )
    return max(common, key=float)


def _is_numeric_time(name: str) -> bool:
    try:
        return math.isfinite(float(name))
    except ValueError:
        return False


def _field_header(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="ascii") as stream:
            header = stream.read(65536)
    except (OSError, UnicodeError):
        return None
    field_class = FIELD_CLASS.search(header)
    if not field_class:
        return None
    value_type = NORMALIZED_VALUE_TYPES[field_class.group(2)]
    dimensions = DIMENSIONS.search(header)
    dimension_values = " ".join(dimensions.group(1).split()) if dimensions else None
    return {
        "field_class": f"{field_class.group(1)}{field_class.group(2)}Field",
        "value_type": value_type,
        "dimensions": dimension_values,
        "units": KNOWN_UNITS.get(dimension_values or ""),
        "association": {
            "vol": "cell",
            "surface": "face",
            "point": "point",
        }[field_class.group(1)],
    }


def _fields_by_partition(
    processors: list[Path], time_name: str
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    discovered: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for processor in processors:
        time_path = processor / time_name
        region_folders = sorted(path for path in time_path.iterdir() if path.is_dir())
        candidates = [
            (region.name, field_path)
            for region in region_folders
            for field_path in sorted(region.iterdir())
            if field_path.is_file()
        ]
        candidates.extend(
            ("default", path)
            for path in sorted(time_path.iterdir())
            if path.is_file()
        )
        for region, path in candidates:
            header = _field_header(path)
            if header is None:
                continue
            entry = {
                **header,
                "processor": processor.name,
                "path": path,
            }
            discovered.setdefault((region, path.name), []).append(entry)
    return discovered


def _field_metadata(
    case_path: Path,
) -> tuple[str, list[Path], dict[tuple[str, str], list[dict[str, Any]]]]:
    if not case_path.is_dir():
        raise CfdDataError("Selected CFD case directory is unavailable.", 404)
    processors = _processor_directories(case_path)
    if not processors:
        raise CfdDataError(
            "This case is not stored as processor-decomposed OpenFOAM output, which is the supported input format.",
            422,
        )
    time_name = _common_time_name(processors)
    discovered = _fields_by_partition(processors, time_name)
    complete_fields = {
        key: records
        for key, records in discovered.items()
        if len(records) == len(processors)
        and len({(item["field_class"], item["dimensions"]) for item in records}) == 1
    }
    if not complete_fields:
        raise CfdDataError(
            f"No complete, consistent OpenFOAM fields were found at solution time {time_name}.",
            422,
        )
    return time_name, processors, complete_fields


def get_case_field_options(case_path: Path) -> dict[str, Any]:
    time_name, processors, fields = _field_metadata(case_path)
    options = []
    for (region, name), records in sorted(fields.items()):
        first = records[0]
        field_class = first["field_class"]
        options.append(
            {
                "name": name,
                "region": region,
                "field_class": field_class,
                "value_type": first["value_type"],
                "association": first["association"],
                "dimensions": first["dimensions"],
                "units": first["units"],
                "components": _component_options(first["value_type"]),
                "processor_count": len(records),
                "statistics_supported": field_class in SUPPORTED_CLASSES,
            }
        )
    return {
        "time": time_name,
        "processor_count": len(processors),
        "fields": options,
        "visualization": {
            "status": "boundary_surface_on_request",
            "reason": (
                "A boundary surface is extracted from the selected OpenFOAM region "
                "on request. Field mapping is validated during rendering; this is not "
                "a volumetric rendering."
            ),
        },
    }


def _component_options(value_type: str) -> list[str]:
    if value_type == "scalar":
        return ["value"]
    if value_type == "vector":
        return ["magnitude", "x", "y", "z"]
    return ["magnitude"]


def _number_width(value_type: str) -> int:
    return COMPONENTS_BY_VALUE_TYPE[value_type]


def _selected_value(values: tuple[float, ...], value_type: str, component: str) -> float:
    if value_type == "scalar":
        if component != "value":
            raise CfdDataError("Scalar fields support only the value component.")
        return values[0]
    if value_type == "vector":
        if component in {"x", "y", "z"}:
            return values[{"x": 0, "y": 1, "z": 2}[component]]
        if component == "magnitude":
            return math.sqrt(sum(value * value for value in values))
        raise CfdDataError("Vector fields support magnitude, x, y, or z.")
    if component != "magnitude":
        raise CfdDataError("Tensor fields support the Frobenius magnitude only.")
    if value_type == "symmTensor":
        return math.sqrt(
            values[0] ** 2
            + 2 * values[1] ** 2
            + 2 * values[2] ** 2
            + values[3] ** 2
            + 2 * values[4] ** 2
            + values[5] ** 2
        )
    return math.sqrt(sum(value * value for value in values))


def _field_values(path: Path, value_type: str) -> tuple[str, int, list[float]]:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise CfdDataError(f"Could not read OpenFOAM field file {path.name}: {error}", 422) from error
    empty_nonuniform = INTERNAL_EMPTY_NONUNIFORM.search(text)
    if empty_nonuniform:
        return "nonuniform", 0, []

    nonuniform = INTERNAL_NONUNIFORM.search(text)
    if nonuniform:
        listed_type = nonuniform.group(1)
        if listed_type != value_type:
            raise CfdDataError(
                f"Field file {path.name} declares inconsistent value type {listed_type}.",
                422,
            )
        value_count = int(nonuniform.group(2))
        body = text[nonuniform.end() :]
        numbers = iter(NUMBER.finditer(body))
        values = []
        expected_numbers = value_count * _number_width(value_type)
        for _ in range(expected_numbers):
            match = next(numbers, None)
            if match is None:
                raise CfdDataError(
                    f"Field file {path.name} is truncated before all internal values were read.",
                    422,
                )
            value = float(match.group())
            if not math.isfinite(value):
                raise CfdDataError(
                    f"Field file {path.name} contains a non-finite value.", 422
                )
            values.append(value)
        return "nonuniform", value_count, values

    uniform = INTERNAL_UNIFORM.search(text)
    if not uniform:
        raise CfdDataError(
            f"Field file {path.name} has no supported internalField declaration.", 422
        )
    values = [float(value) for value in NUMBER.findall(uniform.group(1))]
    width = _number_width(value_type)
    if len(values) != width or not all(math.isfinite(value) for value in values):
        raise CfdDataError(
            f"Field file {path.name} has a malformed uniform internal value.", 422
        )
    return "uniform", 1, values


def calculate_field_statistics(
    case_path: Path,
    *,
    case_id: str,
    reynolds_number: int | None,
    mesh_resolution: str | None,
    region: str,
    field_name: str,
    component: str = "value",
) -> dict[str, Any]:
    time_name, processors, fields = _field_metadata(case_path)
    records = fields.get((region, field_name))
    if records is None:
        raise CfdDataError(
            f"Field '{field_name}' is not available in region '{region}' at solution time {time_name}.",
            404,
        )
    field_class = records[0]["field_class"]
    value_type = records[0]["value_type"]
    if field_class not in SUPPORTED_CLASSES:
        raise CfdDataError(
            f"Statistics are not supported for OpenFOAM field type {field_class}.",
            422,
        )
    if component not in _component_options(value_type):
        raise CfdDataError(
            f"Component '{component}' is not supported for {value_type} fields.", 400
        )

    modes: set[str] = set()
    minimum = math.inf
    maximum = -math.inf
    total = 0.0
    count = 0
    uniform_values: list[float] = []
    for record in records:
        mode, _value_count, values = _field_values(record["path"], value_type)
        modes.add(mode)
        if mode == "uniform":
            uniform_values.append(_selected_value(tuple(values), value_type, component))
            continue
        width = _number_width(value_type)
        for offset in range(0, len(values), width):
            selected = _selected_value(
                tuple(values[offset : offset + width]), value_type, component
            )
            minimum = min(minimum, selected)
            maximum = max(maximum, selected)
            total += selected
            count += 1

    if len(modes) != 1:
        raise CfdDataError(
            "The selected field mixes uniform and nonuniform partition data; a weighted global statistic cannot be computed safely.",
            422,
        )
    if modes == {"uniform"}:
        if max(uniform_values) != min(uniform_values):
            raise CfdDataError(
                "Uniform field values differ between partitions; cell counts are unavailable for a valid weighted mean.",
                422,
            )
        minimum = maximum = uniform_values[0]
        total = uniform_values[0]
        count = 1
        statistic_scope = "uniform field definition; mesh-element count is not inferred"
    else:
        statistic_scope = (
            "all internal cell values across processor partitions"
            if records[0]["association"] == "cell"
            else "all internal face values across processor partitions"
        )
    if count == 0:
        raise CfdDataError("No internal field values were available for statistics.", 422)

    first = records[0]
    return {
        "case_id": case_id,
        "reynolds_number": reynolds_number,
        "mesh_resolution": mesh_resolution,
        "region": region,
        "field": field_name,
        "field_class": field_class,
        "component": component,
        "association": first["association"],
        "dimensions": first["dimensions"],
        "units": first["units"],
        "time": time_name,
        "processor_count": len(processors),
        "sample_count": count if modes == {"nonuniform"} else None,
        "statistic_scope": statistic_scope,
        "minimum": minimum,
        "maximum": maximum,
        "mean": total / count,
        "visualization": {
            "status": "boundary_surface_on_request",
            "reason": (
                "A field-mapped boundary surface can be requested from the selected "
                "OpenFOAM region. Internal solver statistics are not boundary-surface "
                "values, and this is not a volumetric rendering."
            ),
        },
    }


def compare_mesh_cases(
    *,
    case_id: str,
    reference_case_id: str,
    case_metadata: dict[str, Any],
    reference_metadata: dict[str, Any],
) -> dict[str, str]:
    if case_id == reference_case_id:
        raise CfdDataError("Choose two different mesh cases for comparison.")
    current_re = case_metadata.get("reynolds_number")
    reference_re = reference_metadata.get("reynolds_number")
    if current_re is None or reference_re is None or current_re != reference_re:
        raise CfdDataError(
            "Mesh comparison requires cases with the same verified Reynolds number."
        )
    current_mesh = case_metadata.get("mesh_resolution")
    reference_mesh = reference_metadata.get("mesh_resolution")
    if not current_mesh or not reference_mesh or current_mesh == reference_mesh:
        raise CfdDataError(
            "Mesh comparison requires two different verified mesh resolutions."
        )
    return {
        "status": "unavailable",
        "reason": (
            "No spatial interpolation between the processor-decomposed meshes has been "
            "validated. Comparing values by array index would be scientifically invalid, "
            "so no cross-mesh result was calculated."
        ),
    }
