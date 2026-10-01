"""ROS-free browser capture tool for SO-101 hand-eye calibration.

Open the printed URL in a browser, use the joint sliders to place the arm,
and press ``Take sample`` when the ChArUco board is visible.  The tool writes
the same ``records.json`` format consumed by ``calibrate_handeye.py``.

This intentionally uses joint sliders rather than Cartesian IK.  It keeps the
hardware path small and explicit while still allowing remote, repeatable,
well-separated calibration poses without ROS.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

JOINTS = [
    ("shoulder_pan.pos", "Shoulder pan", -110.0, 110.0),
    ("shoulder_lift.pos", "Shoulder lift", -100.0, 100.0),
    ("elbow_flex.pos", "Elbow", -100.0, 90.0),
    ("wrist_flex.pos", "Wrist pitch", -95.0, 95.0),
    ("wrist_roll.pos", "Wrist roll", -160.0, 160.0),
    ("gripper.pos", "Gripper", 0.0, 100.0),
]


def page() -> bytes:
    rows = "\n".join(
        f'''<label>{label}: <output id="out-{i}"></output>
        <input id="joint-{i}" type="range" min="{lo}" max="{hi}" step="0.1" value="0">
        </label>'''
        for i, (_, label, lo, hi) in enumerate(JOINTS)
    )
    return f'''<!doctype html>
<html><head><meta charset="utf-8"><title>SO-101 hand-eye capture</title>
<style>
body {{ font:16px sans-serif; max-width:1100px; margin:20px auto; background:#202124; color:#eee }}
main {{ display:grid; grid-template-columns:420px 1fr; gap:24px }}
label {{ display:block; margin:15px 0 }} input {{ width:290px; vertical-align:middle }}
button {{ font-size:17px; margin:6px; padding:10px 14px }}
#frame {{ width:640px; max-width:100%; background:#111 }} #status {{ white-space:pre-wrap; color:#9f9 }}
.danger {{ background:#a22; color:#fff }} .safe {{ background:#174; color:#fff }}
</style></head><body>
<h1>SO-101 hand-eye capture</h1>
<p>Support the arm before enabling torque. Move only after checking the live frame.</p>
<main><section>
<div id="sliders">{rows}</div>
<button class="safe" onclick="enableTorque()">Enable torque</button>
<button onclick="moveArm()">Move to sliders</button>
<button onclick="sample()">Take sample</button>
<button class="danger" onclick="stopArm()">Disable torque / stop</button>
<p id="status">Loading...</p>
</section><section><img id="frame" src="/frame.jpg"><p>Green text means the board has enough detected corners.</p></section></main>
<script>
const joints = {json.dumps([name for name, *_ in JOINTS])};
function values() {{ return joints.map((_,i)=>+document.querySelector('#joint-'+i).value); }}
function setValues(v) {{ v.forEach((x,i)=>{{let e=document.querySelector('#joint-'+i); e.value=x; document.querySelector('#out-'+i).value=(+x).toFixed(1);}}); }}
document.querySelectorAll('input[type=range]').forEach(e=>e.oninput=()=>{{e.previousElementSibling.value=(+e.value).toFixed(1);}});
async function post(path, body={{}}) {{ let r=await fetch(path,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}}); let j=await r.json(); document.querySelector('#status').textContent=j.message||JSON.stringify(j); if(j.values) setValues(j.values); }}
function enableTorque() {{ if(confirm('Support the arm, then enable torque?')) post('/torque',{{enabled:true}}); }}
function stopArm() {{ post('/torque',{{enabled:false}}); }}
function moveArm() {{ post('/move',{{values:values()}}); }}
function sample() {{ post('/sample',{{}}); }}
async function refresh() {{ try {{ let j=await (await fetch('/state')).json(); document.querySelector('#status').textContent=j.message; if(j.values) setValues(j.values); }} catch(e) {{}} }}
setInterval(()=>document.querySelector('#frame').src='/frame.jpg?t='+Date.now(),700);
setInterval(refresh,1000); refresh();
</script></body></html>'''.encode()


class CaptureState:
    def __init__(self, args, interface, camera, board):
        self.args = args
        self.interface = interface
        self.camera = camera
        self.board = board
        self.lock = threading.RLock()
        self.torque = False
        self.records: list[dict] = []
        self.capture_dir = args.capture_dir or (REPO_ROOT / "outputs/handeye" / args.camera)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.record_path = self.capture_dir / "records.json"
        self.values = self._read_current()
        self.message = "Torque is OFF. Set sliders, support the arm, then enable torque."

    def _read_current(self):
        observation = self.interface.robot.get_observation()
        return [float(observation[name]) for name, *_ in JOINTS]

    def _action(self, values):
        return {name: float(value) for (name, *_), value in zip(JOINTS, values)}

    def json(self):
        return {"values": self.values, "torque": self.torque, "message": self.message,
                "samples": len(self.records)}

    def enable_torque(self, enabled):
        with self.lock:
            if enabled:
                self.interface.robot.send_action(self._action(self.values))
                self.interface.robot.bus.enable_torque()
                self.interface.robot.send_action(self._action(self.values))
                self.torque = True
                self.message = "Torque ON. Move to a slider pose, then press Move."
            else:
                self.interface.robot.bus.disable_torque()
                self.torque = False
                self.message = "Torque OFF. Support the arm before moving it by hand."

    def move(self, values):
        if len(values) != len(JOINTS):
            raise ValueError("expected six joint values")
        with self.lock:
            if not self.torque:
                raise RuntimeError("enable torque first")
            self.values = [float(v) for v in values]
            self.interface.robot.send_action(self._action(self.values))
            self.message = "Command sent. Wait for the arm to settle before sampling."

    def frame(self):
        with self.lock:
            frame = self.camera.read_fresh(raw=True)
            detection = __import__("so101.charuco", fromlist=["detect_board"]).detect_board(
                frame, self.board
            )
            import cv2

            cv2.putText(frame, f"ChArUco corners: {detection.count}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 0) if detection.usable() else (0, 180, 255), 2,
                        cv2.LINE_AA)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError("could not encode camera frame")
            return encoded.tobytes()

    def sample(self):
        import cv2
        from so101.charuco import detect_board

        with self.lock:
            if not self.torque:
                raise RuntimeError("enable torque first")
            frame = self.camera.read_fresh(raw=True)
            detection = detect_board(frame, self.board)
            if not detection.usable():
                raise RuntimeError(f"only {detection.count} ChArUco corners detected")
            image_path = self.capture_dir / f"pose_{len(self.records):03d}.png"
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"could not write {image_path}")
            observation = self.interface.robot.get_observation()
            values = [float(observation[name]) for name, *_ in JOINTS]
            self.values = values
            import torch

            self.records.append({
                "index": len(self.records),
                "image": image_path.name,
                "joint_positions_rad": self.interface.get_mapped_actions_vectorized(
                    torch.tensor(values, dtype=torch.float32)
                ).tolist(),
                "raw_values": values,
            })
            self.record_path.write_text(
                json.dumps({"camera": self.args.camera, "records": self.records}, indent=2),
                encoding="utf-8",
            )
            self.message = f"Saved {image_path.name}; {detection.count} corners; {len(self.records)} samples total."


class Handler(BaseHTTPRequestHandler):
    state: CaptureState

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path == "/":
                self._send(200, "text/html; charset=utf-8", page())
            elif path == "/state":
                self._send(200, "application/json", json.dumps(self.state.json()).encode())
            elif path == "/frame.jpg":
                self._send(200, "image/jpeg", self.state.frame())
            else:
                self._send(404, "text/plain", b"not found")
        except Exception as error:  # noqa: BLE001
            self._send(500, "application/json", json.dumps({"message": str(error)}).encode())

    def do_POST(self):  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            path = urlsplit(self.path).path
            if path == "/torque":
                self.state.enable_torque(bool(body.get("enabled")))
            elif path == "/move":
                self.state.move(body.get("values", []))
            elif path == "/sample":
                self.state.sample()
            else:
                raise ValueError("not found")
            self._send(200, "application/json", json.dumps(self.state.json()).encode())
        except Exception as error:  # noqa: BLE001
            self._send(400, "application/json", json.dumps({"message": str(error)}).encode())

    def log_message(self, format, *args):
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=["front", "wrist"], default="front")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--robot-id", default="my_follower")
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--capture-dir", type=Path, default=None)
    args = parser.parse_args()

    import logging
    from so101.charuco import charuco_board, gripper_board
    from so101.real.cameras import DEFAULT_SPECS, Camera
    from so101.real.interface import LeRobotSO101Interface

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    interface = LeRobotSO101Interface(device="cpu", port=args.port, id=args.robot_id,
                                      cameras={}, fps=30, kind="follower")
    interface.init_device()
    interface.connect()
    interface.robot.bus.disable_torque()
    board = gripper_board() if args.camera == "front" else charuco_board()
    server = None
    try:
        with Camera(DEFAULT_SPECS[args.camera], calibration=None, rectify=False) as camera:
            state = CaptureState(args, interface, camera, board)
            Handler.state = state
            server = ThreadingHTTPServer(("0.0.0.0", args.http_port), Handler)
            print(f"Open http://localhost:{args.http_port} in a browser")
            print(f"Samples will be written to {state.capture_dir}")
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web capture.")
    finally:
        if server is not None:
            server.server_close()
        interface.robot.bus.disable_torque()
        interface.robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
