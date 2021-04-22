import numpy as np

class State:
    def __init__(self, name):
        self.name = name
        self.state = "running"
        self.idx = None
        self.CLASS = ["ONE", "TWO", "THREE", "FOUR", "OK", "MENU",
                 "POINTING", "LEFT", "RIGHT", "CIRCLE", "V",
                 "CROSS", "GRAB", "PINCH", "TAP", "DENY", "KNOB", "EXPAND", "NO_GESTURE"]

    def update(self, window, idx): pass
    def get_start(self): pass
    def get_end(self): pass

    def reset(self):
        self.state = "running"
        self.idx = None

class State0(State):
    def __init__(self, name): super().__init__(name)

    def update(self, window, idx):
        n_predicted = len(np.where(window != self.CLASS.index("NO_GESTURE"))[0])
        if n_predicted > 0:
            self.state = "completed"
            self.idx = idx
        return self.state

    def get_start(self):
        if self.state != "completed":
            return None
        else:
            return self.idx

    def reset(self):
        self.state = "running"
        self.idx = None

class State1(State):
    def __init__(self, name, threshold_window = 10, threshold_count = None):
        super().__init__(name)
        self.start = None
        self.end = None
        self.end_count = 0
        self.window_count = 1
        self.count = 1
        self.threshold_window = threshold_window
        self.threshold_count = self.threshold_window // 2 if threshold_count is None else threshold_count

    def update(self, window, idx):
        n_predicted = len(np.where(window != self.CLASS.index("NO_GESTURE"))[0])
        if n_predicted > 0:
            self.count += 1
            self.start = idx
            self.end = None
            self.end_count = 0
        else:
            self.end = idx
            self.end_count += 1
        self.window_count += 1
        if self.window_count == self.threshold_window:
            if self.count >= self.threshold_count:
                self.state = "completed"
            else:
                self.state = "failed"
        return self.state

    def get_end(self):
        if self.state != "completed":
            return None
        else:
            return self.end, self.end_count

    def get_start(self):
        if self.state != "completed":
            return None
        else:
            return self.start

    def reset(self):
        self.start = None
        self.end = None
        self.end_count = 0
        self.window_count = 1
        self.count = 1
        self.idx = None
        self.state = "running"

class State2(State):
    def __init__(self, name):
        super().__init__(name)

    def update(self, window, idx):
        # If the window contains only no gesture
        if len(np.where(window == self.CLASS.index("NO_GESTURE"))[0]) == len(window):
            self.state = "completed"
            self.idx = idx - 1
        return self.state

    def get_end(self):
        if self.state != "completed":
            return None
        else:
            return self.idx

    def reset(self):
        self.state = "running"
        self.idx = None

class State3(State):
    def __init__(self, name, threshold):
        super().__init__(name)
        self.count = 1
        self.threshold = threshold

    def update(self, window, idx):
        if len(np.where(window == self.CLASS.index("NO_GESTURE"))[0]) == len(window):
            self.count += 1
        else:
            return "failed"
        if self.count >= self.threshold:
            return "completed"
        else:
            return "running"

    def set_end_count(self, end, count):
        self.idx = end
        self.count = count

    def reset(self):
        self.count = 1
        self.state = "running"
        self.idx = None

class StateMachine:
    def __init__(self, threshold_window_S1 = 10, threshold_count_S1 = None, threshold_S3 = 10):
        self.states = [State0("S0"), State1("S1", threshold_window_S1, threshold_count_S1), State2("S2"), State3("S3", threshold_S3)]
        self.active = 0
        self.start = None
        self.end = None
        self.done = False

    def update(self, window, idx):
        if self.done:
            return True
        state = self.states[self.active].update(window, idx)
        retval = False

        if self.active == 0:
            if state == "completed":
                self.start = self.states[self.active].get_start()
        elif self.active == 1:
            if state == "completed":
                self.start = self.states[self.active].get_start()
                if self.states[self.active].get_end()[0] is not None:
                    self.end, end_count = self.states[self.active].get_end()
                    self.active += 1
                    self.states[self.active + 1].set_end_count(self.end, end_count)
            elif state == "failed":
                self.start = None
                self.end = None
                self.states[self.active].reset()
                self.active = 0
                self.states[self.active].reset()
        elif self.active == 2:
            if state == "completed":
                self.end = self.states[self.active].get_end()
                self.states[self.active + 1].set_end_count(self.end, 1)
        elif self.active == 3:
            # If the last state return True, I've finally complete the search for start_end, and can reset and restart
            if state == "completed":
                if self.end - self.start > 3:
                    retval = True
                    self.done = True
                else:
                    self.reset()
                    return retval
            elif state == "failed":
                self.states[self.active].reset()
                self.active -= 1
                self.states[self.active].reset()
                self.end = None

        if state == "completed":
            self.active += 1
        return retval

    def get_start(self): return self.start
    def get_end(self): return self.end

    def reset(self):
        self.active = 0
        self.start = None
        self.end = None
        self.done = False
        for s in self.states: s.reset()