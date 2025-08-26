import logging
from llama_index.core.callbacks import CallbackManager
from llama_index.callbacks.arize_phoenix import ArizePhoenixCallback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Phoenix tracing for LlamaIndex workflows
phoenix_callback = ArizePhoenixCallback()
callback_manager = CallbackManager([phoenix_callback])

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}

    def track_query_time(self, query_type: str, duration: float):
        if query_type not in self.metrics:
            self.metrics[query_type] = []
        self.metrics[query_type].append(duration)

    def get_average_response_time(self, query_type: str) -> float:
        if not self.metrics.get(query_type):
            return 0.0
        return sum(self.metrics.get(query_type, [])) / len(self.metrics.get(query_type, []))