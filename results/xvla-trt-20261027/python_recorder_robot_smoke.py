import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from lerobot_vla import record_episodes as recorder

# Only the gamepad is scripted; real cameras, IMUs, 100 Hz controller, dataset
# writer and save/discard/resume paths run. All commanded valves are zero.
class NeutralPad:
    def __init__(self): self.n=0
    def open(self): return True
    def is_live(self): return True
    def close(self): pass
    def poll(self):
        self.n+=1
        if self.n>=230: raise KeyboardInterrupt
        mask=(1 if self.n in [1,61,91,151,211] else 2 if self.n==121 else 0)
        return {k:0.0 for k in ['right_rl','right_ud','left_rl','left_ud','right_paddle','left_paddle']},mask

root=Path('results/dual-ir-check-20261027/robot-neutral-test-dataset').resolve()
with patch.object(recorder,'LocalGamepadInput',NeutralPad):
    for resume in [False,True]:
        sys.argv=['record_episodes','--root',str(root),'--repo-id','test/robot_neutral_diagnostic',
                  '--task','diagnostic neutral commands; not training data']+(['--resume'] if resume else [])
        assert recorder.main()==0
print('ROBOT RECORDER SAVE/DISCARD/RESUME PASSED')
