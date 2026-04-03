"""No-op Chroma telemetry client to avoid posthog/runtime telemetry issues."""

from overrides import override
from chromadb.config import System
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent


class NoOpTelemetry(ProductTelemetryClient):
    def __init__(self, system: System):
        super().__init__(system)

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return
