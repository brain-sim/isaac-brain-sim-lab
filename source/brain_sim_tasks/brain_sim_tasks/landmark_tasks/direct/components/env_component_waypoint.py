class EnvComponentWaypoint:
    """Component responsible for waypoint generation and management."""

    def __init__(self, env):
        self.env = env
        self.waypoints = None
