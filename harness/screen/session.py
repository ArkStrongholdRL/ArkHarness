"""Keep resource registration separate from one task's recognition context."""

from contextlib import contextmanager
from threading import Lock

from maa.custom_recognition import CustomRecognition


class _TaskRecognizer(CustomRecognition):
    def __init__(self):
        super().__init__()
        self.active = None

    def analyze(self, context, argv):
        recognition = self.active
        return recognition.analyze(context, argv) if recognition is not None else None


class SessionRecognizers:
    def __init__(self, resource):
        self._recognizers = {name: _TaskRecognizer() for name in ("ArkNavigation", "ArkDeploy", "ArkReadScreen")}
        self._task_lock = Lock()
        for name, recognizer in self._recognizers.items():
            if not resource.register_custom_recognition(name, recognizer):
                raise RuntimeError(f"Could not register {name}")

    @contextmanager
    def bind(self, name, recognition):
        recognizer = self._recognizers[name]
        if not self._task_lock.acquire(blocking=False):
            raise RuntimeError("A screen recognition task is already active in this session")
        try:
            recognizer.active = recognition
            yield
        finally:
            recognizer.active = None
            self._task_lock.release()


@contextmanager
def task_recognition(tasker, name, recognition):
    """Attach a recognizer until its synchronous MFW task has finished."""
    session = getattr(tasker, "screen_recognizers", None)
    if session is not None:
        with session.bind(name, recognition):
            yield
        return

    # Keep compatibility with callers that supply their own Tasker and resource.
    resource = tasker.resource
    if not resource.register_custom_recognition(name, recognition):
        raise RuntimeError(f"Could not register {name}")
    try:
        yield
    finally:
        resource.unregister_custom_recognition(name)
