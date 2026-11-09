# Valtra T171 pallet demo (X-VLA, Jetson)

Nopee vibe-koodattu pohja Valtraan. Tästä pitää vielä kattoo:

-CAN prio, menin kolmosella enkä testannut toimiiko suoraan laskemalla arvoa. En usko että koneen MCU kattoo tota yhtään (eli ei varmaan toimi)

-pätkiminen (MCU tappelee joikkarin/jetsonin komentojen välillä) korjattiin ampumalla komentoja sisään 100hz. Huomio että väännetty pythonilla, eli viestit ei mene tasaisin väliajoin sisään (korkea jitter)


Ekaa kertaa kun ajat tän jetsonilla, niin odota rauhassa 10min. Malli rakennetaan splitatuista ONNX-palikoista yhdeksi TensorRuntime(TRT) kokonaisuudeksi, tässä menee hetki. Moottorit tallennetaan niin jatkossa lähtee nopeeta käyntiin.


Jetson on semi tarkka siitä että se saa laadukasta virtaa sisään. Huono piuha tai muuttuva jännite saattaa jumittaa koko kortin, GPU hävii näkyvistä kokonaan ("brown-out"). Heittää jotain satunnaisia TRT virheitä.

Omat mallit voit kouluttaa ihan normaalisti LeRobotilla, sun pitää vaan exporttaa `jetson-orin-nano-vla` -repon työkalulla. Claude jeesaa ja sillai.

noni palaillaa mo

---


X-VLA runtime for a Valtra T171 front loader on a Jetson Orin Nano: one USB camera -> TensorRT policy -> ISOBUS
valve commands. Self-contained (`lerobot_vla/` is the minimal TensorRT runtime, `modules/gamepad.py` the pad
reader). The model `bundle/` (10k-step full fine-tune on `AleksiL/valtra_t171_pallet`: one camera, zero state,
actions /100 in training, x100 here) is not in git; put it in `./bundle`.

```bash
python -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
P=.venv/bin/python

$P valtra_run.py check                               # engines (first run ~5 min) + one chunk per task
$P valtra_run.py camera --camera /dev/video0         # camera.jpg vs reference_front.jpg
$P valtra_run.py listen --can slcan:/dev/ttyACM0     # tractor's own valve commands, signs
$P valtra_run.py run --camera /dev/video0 --can slcan:/dev/ttyACM0          # dry run
$P valtra_run.py run --camera /dev/video0 --can slcan:/dev/ttyACM0 --live   # drives
```

In `run`:

- Gamepad (Xbox, local): starts in MANUAL, sticks centred = 0. **A** toggles policy <-> manual,
  **D-pad left/right** cycles the task, right stick up/down = boom, left/right = bucket
  (left = +, curl in). Full stick = 50 % (`--manual-flow`). A lost pad sends 0.
- Keyboard: `1` (lower the pallet) / `2` (lift the pallet) + Enter starts the policy,
  empty Enter = STOP. Ctrl+C leaves both valves blocked.

## At the tractor

1. Camera: `v4l2-ctl --list-devices`, then `camera`; aim like `reference_front.jpg`
   (fork carriage at the bottom, looking forward/down).
2. CAN adapter (250 kbit/s, normal mode, not listen-only). This kernel has no USB-CAN
   drivers, so pick the userspace backend:
   - serial (CANable slcan, CANUSB, /dev/ttyACM*): `--can slcan:/dev/ttyACM0`
   - candleLight / gs_usb firmware: `--can gs_usb:0` (may need sudo or a udev rule for USB access)
   - Orin's own CAN (needs a transceiver): `sudo ip link set can0 up type can bitrate 250000`,
     then `--can socketcan:can0`
3. `listen` while moving the loader by hand: boom up must show `+`, bucket curl-in `+`.
   If a sign is reversed, add `--boom-raise retract` / `--bucket-curl retract` to every command.
4. Dry `run`, press `2` / `1`, check the printed % look sane, then `--live`.

## Things to know

- Frames: ISO 11783-7 Auxiliary Valve Command, boom = valve 7 `0CFE3728`, bucket = valve 8
  `0CFE3828` (source address 0x28 as in the recording, `--sa` changes it). Byte 1 flow
  0.4 %/bit, byte 3 = 0xC0 + state (0 block, 1 extend, 2 retract), rest 00, sent at 30 Hz;
  same layout as the Valtra terminal (idle `00 00 C0 00 00 00 00 00`).
- `--live` refuses to start if another node already sends valve 7/8 commands (the tractor
  terminal did in the recording). Two senders fight; `--allow-other-sender` overrides.
- Clamp +-50 % (`--max-flow`), deadband 1 % (`--deadband`). Stale frame (>0.3 s), inference
  stall or an exhausted chunk sends 0.
- The model only knows the loader valves; driving is yours. No held-out score exists: this is
  the first test of the policy.
