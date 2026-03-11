"""TTIR conversion exceptions.

Defined in a separate module to allow import without pypto dependency.
"""


class ConversionError(Exception):
    """Raised when TTIR to PyPTO conversion fails."""

    def __init__(
        self,
        message: str,
        op_name: str | None = None,
        span: object = None,
    ):
        self.op_name = op_name
        self.span = span

        location = "<unknown>"
        if span is not None and hasattr(span, "filename"):
            try:
                loc_str = str(span)
                if loc_str and loc_str != "unknown":
                    location = f"{span.filename}:{span.line}:{span.column}"
            except Exception:
                pass

        full_message = f"[{location}] {message}"
        if op_name:
            full_message += f"\n  Operation: {op_name}"

        super().__init__(full_message)


class UnsupportedOpError(ConversionError):
    """Raised when encountering an unsupported TTIR operation."""

    def __init__(self, op_name: str, span: object = None):
        message = f"Unsupported operation: {op_name}"
        suggestion = (
            f"Operation '{op_name}' is not yet supported. "
            "Please check the supported operations list."
        )
        super().__init__(message, op_name=op_name, span=span)
        self.suggestion = suggestion
