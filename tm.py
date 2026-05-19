from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional
import numpy as np


class TMState(Enum):
    INIT = auto()
    SCAN = auto()
    REWIND = auto()
    DONE = auto()

    def label(self):
        return self.name


SYM_0 = 0
SYM_1 = 1
SYM_B = 2
SYM_F = 3
SYM_LEFT = 4
SYM_BLANK = 5

SYM_CHAR = {
    SYM_0: '0',
    SYM_1: '1',
    SYM_B: 'B',
    SYM_F: 'F',
    SYM_LEFT: '#',
    SYM_BLANK: '_',
}

DELTA: Dict[Tuple[TMState, int], Tuple[TMState, int, int]] = {
    (TMState.INIT, SYM_LEFT): (TMState.SCAN, SYM_LEFT, +1),

    (TMState.SCAN, SYM_0): (TMState.SCAN, SYM_0, +1),
    (TMState.SCAN, SYM_1): (TMState.SCAN, SYM_1, +1),
    (TMState.SCAN, SYM_B): (TMState.SCAN, SYM_F, +1),
    (TMState.SCAN, SYM_BLANK): (TMState.REWIND, SYM_BLANK, -1),

    (TMState.REWIND, SYM_0): (TMState.REWIND, SYM_0, -1),
    (TMState.REWIND, SYM_1): (TMState.REWIND, SYM_1, -1),
    (TMState.REWIND, SYM_F): (TMState.REWIND, SYM_F, -1),
    (TMState.REWIND, SYM_LEFT): (TMState.DONE, SYM_LEFT, 0),
}


@dataclass
class TMStep:
    state: TMState
    next_state: TMState
    head: int
    next_head: int
    read_sym: int
    write_sym: int
    direction: int
    tape: List[int]
    feature_idx: int
    accumulator: float
    output_features: List[float]

    def transition_label(self):
        d = {-1: 'L', 0: 'S', +1: 'R'}[self.direction]
        return f"{SYM_CHAR[self.read_sym]} -> {SYM_CHAR[self.write_sym]},{d}"


REGION_SIZE = 98
N_REGIONS = 8


class GranularTM:
    def __init__(self, image_bits, target_features):
        bits = [int(b) for b in image_bits]
        assert len(bits) == 784

        tape = [SYM_LEFT]
        for region in range(N_REGIONS):
            for i in range(REGION_SIZE):
                idx = region * REGION_SIZE + i
                tape.append(SYM_1 if bits[idx] == 1 else SYM_0)
            tape.append(SYM_B)
        tape.append(SYM_BLANK)

        self.tape = tape
        self.head = 0
        self.state = TMState.INIT
        self.feature_idx = 0
        self.accumulator = 0.0
        self.output_features = []
        self.target_features = np.asarray(target_features).flatten()
        self.history = []

    def step(self):
        if self.state == TMState.DONE:
            return None

        if 0 <= self.head < len(self.tape):
            read_sym = self.tape[self.head]
        else:
            read_sym = SYM_BLANK

        key = (self.state, read_sym)
        if key not in DELTA:
            self.state = TMState.DONE
            return None

        next_state, write_sym, direction = DELTA[key]

        prev_state = self.state
        if prev_state == TMState.SCAN:
            if read_sym == SYM_0 or read_sym == SYM_1:
                self.accumulator += read_sym
            elif read_sym == SYM_B:
                if self.feature_idx < len(self.target_features):
                    self.output_features.append(float(self.target_features[self.feature_idx]))
                else:
                    self.output_features.append(0.0)
                self.feature_idx += 1
                self.accumulator = 0.0

        self.tape[self.head] = write_sym
        next_head = self.head + direction

        snap = TMStep(
            state=prev_state,
            next_state=next_state,
            head=self.head,
            next_head=next_head,
            read_sym=read_sym,
            write_sym=write_sym,
            direction=direction,
            tape=self.tape[:],
            feature_idx=self.feature_idx,
            accumulator=self.accumulator,
            output_features=self.output_features[:],
        )
        self.history.append(snap)

        self.state = next_state
        self.head = next_head
        return snap

    def run_to_done(self):
        n = 0
        while self.state != TMState.DONE and n < 10000:
            self.step()
            n += 1
        return self.history
