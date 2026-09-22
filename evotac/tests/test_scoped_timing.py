from evotac.perf.scoped_timing import ScopedTiming


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 0.1
        return self.value


def test_scoped_timing_reports_inclusive_and_exclusive_phases():
    clock = Clock()
    timing = ScopedTiming(clock)
    with timing.measure("outer"):
        with timing.measure("inner"):
            pass
    report = timing.report()["setup"]
    assert report["outer"]["calls"] == 1
    assert report["inner"]["calls"] == 1
    assert report["outer"]["inclusive_seconds"] > report["outer"]["exclusive_seconds"]


def test_scoped_timing_wrap_restores_instance_method():
    clock = Clock()
    timing = ScopedTiming(clock)

    class Worker:
        def run(self, value):
            return value + 1

    worker = Worker()
    timing.wrap(worker, "run", "work")
    assert worker.run(3) == 4
    assert timing.report()["setup"]["work"]["calls"] == 1
    timing.close()
    assert worker.run(3) == 4
    assert "run" not in vars(worker)
