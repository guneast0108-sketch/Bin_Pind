class PindError(Exception):
    """Base domain exception."""


class VideoNotFound(PindError):
    pass


class PlaceNotFound(PindError):
    pass


class CostLimitExceeded(PindError):
    def __init__(self, video_id: str, estimated: float, limit: float) -> None:
        super().__init__(f"video={video_id} estimated={estimated:.3f} limit={limit:.3f}")
        self.video_id = video_id
        self.estimated = estimated
        self.limit = limit


class DownloadFailed(PindError):
    pass


class WebhookAuthError(PindError):
    pass
