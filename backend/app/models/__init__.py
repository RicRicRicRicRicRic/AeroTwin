"""ML model wrappers (material segmentation, element detection, crack detection).

Model weights live in ``backend/app/models/weights/``; missing weights must
surface as HTTP 503 responses, never as silent failures.
"""
