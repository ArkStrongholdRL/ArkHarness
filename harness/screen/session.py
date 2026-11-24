"""Keep resource registration separate from one task's recognition context."""

from contextlib import contextmanager
from threading import Lock

from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition


READY_ACTION = "ArkRememberAndClickReady"
SCREEN_RECOGNIZERS = (
    "ArkBoardRead",
    "ArkBriefing",
    "ArkCommand",
    "ArkCommander",
    "ArkDeploy",
    "ArkEquip",
    "ArkEquipChoice",
    "ArkEquipmentRead",
    "ArkHand",
    "ArkHandSale",
    "ArkNavigation",
    "ArkReadScreen",
    "ArkSell",
    "ArkShopBuy",
    "ArkShopControl",
    "ArkShopView",
    "ArkWithdraw",
)


class _TaskRecognizer(CustomRecognition):
    def __init__(self):
        super().__init__()
        self.active = None

    def analyze(self, context, argv):
        recognition = self.active
        return recognition.analyze(context, argv) if recognition is not None else None


class _TaskAction(CustomAction):
    def __init__(self):
        super().__init__()
        self.active = None

    def run(self, context, argv):
        action = self.active
        return action.run(context, argv) if action is not None else False


class SessionRecognizers:
    def __init__(self, resource):
        self._recognizers = {name: _TaskRecognizer() for name in SCREEN_RECOGNIZERS}
        self._task_lock = Lock()
        for name, recognizer in self._recognizers.items():
            if not resource.register_custom_recognition(name, recognizer):
                raise RuntimeError(f"Could not register {name}")
        self._ready = _TaskAction()
        if not resource.register_custom_action(READY_ACTION, self._ready):
            raise RuntimeError(f"Could not register {READY_ACTION}")

    @contextmanager
    def bind(self, name, recognition, ready_action=None):
        recognizer = self._recognizers[name]
        if not self._task_lock.acquire(blocking=False):
            raise RuntimeError("A screen recognition task is already active in this session")
        try:
            recognizer.active = recognition
            self._ready.active = ready_action
            yield
        finally:
            self._ready.active = None
            recognizer.active = None
            self._task_lock.release()


@contextmanager
def task_recognition(tasker, name, recognition, *, ready_action=None):
    """Attach callbacks until their synchronous MFW task has finished."""
    session = getattr(tasker, "screen_recognizers", None)
    if session is not None:
        with session.bind(name, recognition, ready_action):
            yield
        return

    # Keep compatibility with callers that supply their own Tasker and resource.
    resource = tasker.resource
    if not resource.register_custom_recognition(name, recognition):
        raise RuntimeError(f"Could not register {name}")
    try:
        if ready_action is None:
            yield
        else:
            if not resource.register_custom_action(READY_ACTION, ready_action):
                raise RuntimeError(f"Could not register {READY_ACTION}")
            try:
                yield
            finally:
                resource.unregister_custom_action(READY_ACTION)
    finally:
        resource.unregister_custom_recognition(name)
