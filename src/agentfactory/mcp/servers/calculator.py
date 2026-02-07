"""
Calculator MCP tool server.

Provides: calculate (math expressions) and convert_units.

Run as:
    python -m agentfactory.mcp.servers.calculator
"""

from __future__ import annotations

import math
import logging
from typing import Any

from agentfactory.mcp.server import StdioToolServer, ToolHandler

logger = logging.getLogger(__name__)

# Safe math namespace
SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "int": int, "float": float,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "pi": math.pi, "e": math.e, "inf": math.inf,
    "ceil": math.ceil, "floor": math.floor,
}


class CalculateTool(ToolHandler):
    name = "calculate"
    description = "Evaluate a mathematical expression safely. Supports standard math functions."
    parameters = {
        "expression": {
            "type": "string",
            "description": "Math expression (e.g., 'sqrt(2) * pi / 3')",
        },
    }

    def handle(self, params: dict[str, Any]) -> dict:
        expression = params.get("expression", "")
        if not expression:
            return {"error": "No expression provided"}

        try:
            result = eval(expression, {"__builtins__": {}}, SAFE_MATH)
            return {"expression": expression, "result": float(result)}
        except Exception as e:
            return {"expression": expression, "error": str(e)}


UNIT_CONVERSIONS = {
    ("km", "miles"): 0.621371,
    ("miles", "km"): 1.60934,
    ("kg", "lb"): 2.20462,
    ("lb", "kg"): 0.453592,
    ("m", "ft"): 3.28084,
    ("ft", "m"): 0.3048,
    ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
}


class ConvertUnitsTool(ToolHandler):
    name = "convert_units"
    description = "Convert between common units (length, weight, temperature)."
    parameters = {
        "value": {"type": "number", "description": "The value to convert"},
        "from_unit": {"type": "string", "description": "Source unit"},
        "to_unit": {"type": "string", "description": "Target unit"},
    }

    def handle(self, params: dict[str, Any]) -> dict:
        value = params.get("value")
        from_unit = params.get("from_unit", "").lower()
        to_unit = params.get("to_unit", "").lower()

        key = (from_unit, to_unit)
        conversion = UNIT_CONVERSIONS.get(key)

        if conversion is None:
            supported = [f"{f}->{t}" for f, t in UNIT_CONVERSIONS.keys()]
            return {"error": f"Unsupported conversion: {from_unit} -> {to_unit}. Supported: {supported}"}

        if callable(conversion):
            result = conversion(value)
        else:
            result = value * conversion

        return {"value": value, "from": from_unit, "to": to_unit, "result": round(result, 6)}


def main():
    server = StdioToolServer()
    server.register(CalculateTool())
    server.register(ConvertUnitsTool())
    server.run()


if __name__ == "__main__":
    main()
