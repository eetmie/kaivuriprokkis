/**
 * Browser-side excavator controller.
 *
 * Renders the real URDF with Three.js (WebGL, GPU-accelerated) so CPU load
 * stays negligible. Joint angles come from server telemetry over SSE; keyboard
 * events POST the keyset to the server, which owns the IK tick loop.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import URDFLoader from '/vendor/urdf-loader/URDFLoader.js';

// The URDF is authored Z-up; tell Three.js to treat Z as the world up vector.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

// TODO: read from server config once exposed over SSE.
// Matches robot.frame.origin_height_m in control_config.yaml.
// IK world origin is this many metres below the slew joint (URDF world origin),
// so all IK coordinates must be shifted down by this amount before rendering.
const IK_ORIGIN_Z_OFFSET = 0.075;

const viewport = document.getElementById('viewport');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1f24);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 50);
camera.position.set(1.4, -1.4, 1.1);
camera.up.set(0, 0, 1);
camera.lookAt(0, 0, 0.15);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
viewport.appendChild(renderer.domElement);

const ambient = new THREE.AmbientLight(0xffffff, 0.55);
scene.add(ambient);
const sun = new THREE.DirectionalLight(0xffffff, 0.8);
sun.position.set(1.5, -1.2, 2.5);
scene.add(sun);
const fill = new THREE.DirectionalLight(0xcfd6de, 0.25);
fill.position.set(-1.0, 1.5, 0.5);
scene.add(fill);

// Ground plane — a faint checker for depth perception
const grid = new THREE.GridHelper(4, 20, 0x304050, 0x243040);
grid.rotation.x = Math.PI / 2;
grid.position.z = -0.075;
scene.add(grid);

// World-origin axis helper (X red / Y green / Z blue)
const axes = new THREE.AxesHelper(0.15);
scene.add(axes);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.target.set(0.45, 0, -0.05);
controls.update();

// Render-on-demand: only redraw when something changes.
let needsRender = true;
const markDirty = () => { needsRender = true; };
controls.addEventListener('change', markDirty);

// ----- Target marker -------------------------------------------------------
const target = makeTargetMarker();
scene.add(target.group);

function makeTargetMarker() {
    const group = new THREE.Group();
    const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.012, 16, 12),
        new THREE.MeshBasicMaterial({ color: 0xff4444 })
    );
    group.add(sphere);
    function set(x, y, z) {
        group.position.set(x, y, z - IK_ORIGIN_Z_OFFSET);
    }
    return { group, set };
}

// ----- URDF load -----------------------------------------------------------
let robot = null;
const JOINT_MAP = {
    slew: 'revolute_carriage',
    boom: 'revolute_lift',
    arm: 'revolute_tilt',
    bucket: 'revolute_tool',
};

const loadingOverlay = document.createElement('div');
loadingOverlay.id = 'loading-overlay';
loadingOverlay.textContent = 'Loading robot meshes…';
viewport.appendChild(loadingOverlay);

async function loadRobot() {
    const res = await fetch('/robot/robot.urdf');
    let xml = await res.text();
    const initialJointPositions = parseInitialJointPositions(xml);
    // URDF references meshes as `file:./meshes/X.obj` — strip the scheme so
    // the browser can fetch them relative to workingPath.
    xml = xml.replace(/filename="file:/g, 'filename="');

    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    loader.loadMeshCb = (path, mgr, done) => loadObjMesh(path, mgr, done);
    const parsed = loader.parse(xml, '/robot/');

    await new Promise((resolve, reject) => {
        manager.onLoad = resolve;
        manager.onError = reject;
    });

    applyInitialJointPositions(parsed, initialJointPositions);

    parsed.traverse(c => {
        if (c.isMesh) {
            c.castShadow = false;
            c.receiveShadow = false;
            // Some excavator meshes come in with non-standard face orientation;
            // double-side keeps them visible from any angle.
            if (c.material) {
                const mats = Array.isArray(c.material) ? c.material : [c.material];
                mats.forEach(m => { m.side = THREE.DoubleSide; });
            }
        }
    });

    return parsed;
}

function parseInitialJointPositions(xml) {
    const doc = new DOMParser().parseFromString(xml, 'application/xml');
    const values = {};
    doc.querySelectorAll('ros2_control joint').forEach((joint) => {
        const name = joint.getAttribute('name');
        const param = joint.querySelector('state_interface[name="position"] param[name="initial_value"]');
        if (!name || !param) return;
        const value = Number.parseFloat(param.textContent || '');
        if (Number.isFinite(value)) {
            values[name] = value;
        }
    });
    return values;
}

function applyInitialJointPositions(loadedRobot, initialJointPositions) {
    Object.entries(initialJointPositions).forEach(([jointName, value]) => {
        const joint = loadedRobot.joints?.[jointName];
        if (joint) {
            joint.setJointValue(value);
        }
    });
}

function loadObjMesh(path, manager, done) {
    const cleanPath = path.replace(/\bfile:/g, '');
    const slashIdx = cleanPath.lastIndexOf('/');
    const dir = cleanPath.substring(0, slashIdx + 1);
    const file = cleanPath.substring(slashIdx + 1);
    const mtlFile = file.replace(/\.obj$/i, '.mtl');

    const mtl = new MTLLoader(manager);
    mtl.setPath(dir);
    mtl.load(
        mtlFile,
        (materials) => {
            materials.preload();
            const obj = new OBJLoader(manager);
            obj.setMaterials(materials);
            obj.setPath(dir);
            obj.load(file, (group) => done(group), null, (err) => done(null, err));
        },
        null,
        () => {
            // Missing or broken MTL — still render the OBJ with default material.
            const obj = new OBJLoader(manager);
            obj.setPath(dir);
            obj.load(file, (group) => done(group), null, (err) => done(null, err));
        },
    );
}

loadRobot()
    .then((r) => {
        robot = r;
        scene.add(robot);
        loadingOverlay.classList.add('hidden');
        setTimeout(() => loadingOverlay.remove(), 500);
        markDirty();
    })
    .catch((err) => {
        console.error('Failed to load robot:', err);
        loadingOverlay.textContent = 'Failed to load robot. Check console.';
    });

// ----- Resize --------------------------------------------------------------
function resize() {
    const w = viewport.clientWidth;
    const h = viewport.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
    markDirty();
}
new ResizeObserver(resize).observe(viewport);
resize();

// ----- Render loop ---------------------------------------------------------
function tick() {
    controls.update();
    if (needsRender) {
        renderer.render(scene, camera);
        needsRender = false;
    }
    requestAnimationFrame(tick);
}
tick();

// ----- Server state via SSE ------------------------------------------------
const els = {
    status: document.getElementById('status'),
    hz: document.getElementById('hz'),
    recv: document.getElementById('recv'),
    tx: document.getElementById('tx'),
    ty: document.getElementById('ty'),
    tz: document.getElementById('tz'),
    trot: document.getElementById('trot'),
    mx: document.getElementById('mx'),
    my: document.getElementById('my'),
    mz: document.getElementById('mz'),
    mrot: document.getElementById('mrot'),
    hint: document.getElementById('hint'),
    host: document.getElementById('host'),
    port: document.getElementById('port'),
    rate: document.getElementById('rate'),
    stepPos: document.getElementById('step-pos'),
    stepRot: document.getElementById('step-rot'),
    btnConnect: document.getElementById('btn-connect'),
    btnStream: document.getElementById('btn-stream'),
    btnPause: document.getElementById('btn-pause'),
    btnResume: document.getElementById('btn-resume'),
    btnReload: document.getElementById('btn-reload'),
    btnDirect: document.getElementById('btn-direct'),
    btnIkSpace: document.getElementById('btn-ik-space'),
    btnReset: document.getElementById('btn-reset'),
};

let lastState = null;

const fmt = (v, dp = 3, width = 6) => {
    const s = (v >= 0 ? '+' : '') + v.toFixed(dp);
    return s.padStart(width, ' ');
};

const HINTS = {
    cartesian: 'IK Cartesian: A/D=X, W/S=Z, Q/E=Y, R/F=Rot · Tab = toggle · Direct: A/D=Slew, W/S=Boom, Q/E=Arm, R/F=Bucket',
    radial: 'IK Radial: A/D=Slew, W/S=Radius, Q/E=Z, R/F=Rot · Tab = toggle · Direct: A/D=Slew, W/S=Boom, Q/E=Arm, R/F=Bucket',
};

function applyState(state) {
    lastState = state;

    els.status.textContent = `Status: ${state.status}`;
    els.hz.textContent = `Send Hz: ${state.send_hz.toFixed(1)}`;
    els.recv.textContent = `Recv: ${state.recv}`;

    els.tx.textContent = fmt(state.target.x);
    els.ty.textContent = fmt(state.target.y);
    els.tz.textContent = fmt(state.target.z);
    els.trot.textContent = fmt(state.target.rot_y, 1, 5);
    els.mx.textContent = fmt(state.measured.x);
    els.my.textContent = fmt(state.measured.y);
    els.mz.textContent = fmt(state.measured.z);
    els.mrot.textContent = fmt(state.measured.rot_y, 1, 5);

    target.set(state.target.x, state.target.y, state.target.z);

    if (robot && state.joint_angles_deg && state.measured.valid) {
        const [slew, boom, arm, bucket] = state.joint_angles_deg;
        setJoint('slew', slew);
        setJoint('boom', boom);
        setJoint('arm', arm);
        setJoint('bucket', bucket);
    }

    // Buttons
    const connected = state.connected;
    const streaming = state.streaming;
    els.btnConnect.textContent = connected ? 'Disconnect' : 'Connect';
    els.btnStream.disabled = !connected;
    els.btnStream.textContent = streaming ? 'Stop Streaming' : 'Start Streaming';
    els.btnPause.disabled = !streaming;
    els.btnResume.disabled = !streaming;
    els.btnReload.disabled = !streaming;
    els.btnDirect.disabled = !streaming;
    els.btnDirect.textContent = state.direct_mode ? 'IK Mode' : 'Direct Mode';
    els.btnDirect.classList.toggle('active', state.direct_mode);
    els.btnIkSpace.textContent = `IK Space: ${state.ik_space === 'radial' ? 'Radial' : 'Cartesian'}`;
    els.btnIkSpace.classList.toggle('active', state.ik_space === 'radial');
    els.hint.textContent = HINTS[state.ik_space] || HINTS.cartesian;

    // Host / port / rate / step inputs are user-owned: the server never
    // writes back to them. Initial values come from the HTML defaults.

    markDirty();
}

function setJoint(name, degrees) {
    if (!robot) return;
    const jointName = JOINT_MAP[name];
    const joint = robot.joints?.[jointName];
    if (!joint) return;
    joint.setJointValue(degrees * Math.PI / 180);
}

function connectSSE() {
    const es = new EventSource('/events');
    es.onmessage = (ev) => {
        try {
            applyState(JSON.parse(ev.data));
        } catch (e) {
            console.warn('Bad SSE message:', e);
        }
    };
    es.onerror = () => {
        // EventSource auto-reconnects; nothing to do.
    };
}
connectSSE();

// ----- Actions -------------------------------------------------------------
async function postJSON(path, body) {
    try {
        const res = await fetch(path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body || {}),
        });
        return await res.json();
    } catch (e) {
        console.error('POST failed', path, e);
        return { ok: false, error: String(e) };
    }
}

const postAction = (action, extra = {}) => postJSON('/action', { action, ...extra });

els.btnConnect.addEventListener('click', async () => {
    if (lastState?.connected) {
        await postAction('disconnect');
        return;
    }
    const host = els.host.value.trim();
    const port = parseInt(els.port.value, 10);
    const rate = parseFloat(els.rate.value);
    if (!host) {
        alert('Enter a host IP first.');
        els.host.focus();
        return;
    }
    if (!Number.isFinite(port) || !Number.isFinite(rate)) {
        alert('Port and rate must be numbers.');
        return;
    }
    const res = await postAction('connect', { host, port, rate_hz: rate });
    if (!res.ok) alert(res.msg || 'Connection failed');
});
els.btnStream.addEventListener('click', async () => {
    const action = lastState?.streaming ? 'stop_stream' : 'start_stream';
    const res = await postAction(action);
    if (!res.ok && res.msg) alert(res.msg);
});
els.btnPause.addEventListener('click', () => postAction('pause'));
els.btnResume.addEventListener('click', () => postAction('resume'));
els.btnReload.addEventListener('click', () => postAction('reload'));
els.btnDirect.addEventListener('click', () => postAction('toggle_direct'));
els.btnIkSpace.addEventListener('click', () => postAction('toggle_ik_space'));
els.btnReset.addEventListener('click', () => postAction('reset'));

function pushConfig() {
    postJSON('/config', {
        step_pos: parseFloat(els.stepPos.value),
        step_rot: parseFloat(els.stepRot.value),
    });
}
els.stepPos.addEventListener('change', pushConfig);
els.stepRot.addEventListener('change', pushConfig);

// ----- Keyboard → server ---------------------------------------------------
const keysDown = new Set();
const MOVE_KEYS = new Set(['w', 'a', 's', 'd', 'q', 'e', 'r', 'f']);

function isEditing() {
    const el = document.activeElement;
    return el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA');
}

function postKeys() {
    postJSON('/input', { keys: Array.from(keysDown) });
}

window.addEventListener('keydown', (e) => {
    if (e.repeat) return;
    if (isEditing()) return;
    const k = e.key.toLowerCase();
    if (k === 'tab') {
        e.preventDefault();
        postAction('toggle_ik_space');
        return;
    }
    if (e.key === 'Shift') keysDown.add('shift');
    else if (MOVE_KEYS.has(k)) keysDown.add(k);
    else return;
    e.preventDefault();
    postKeys();
});

window.addEventListener('keyup', (e) => {
    if (isEditing()) return;
    const k = e.key.toLowerCase();
    if (e.key === 'Shift') keysDown.delete('shift');
    else if (MOVE_KEYS.has(k)) keysDown.delete(k);
    else return;
    postKeys();
});

window.addEventListener('blur', () => {
    if (keysDown.size) {
        keysDown.clear();
        postKeys();
    }
});
