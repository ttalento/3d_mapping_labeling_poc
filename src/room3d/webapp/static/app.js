import * as THREE from "three";

/* ------------------------------------------------------------------ state */

const S = {
  room: null,
  summary: null,
  objects: [],
  observations: [],
  obsByFrame: new Map(),
  frames: [],
  trajectory: null,
  selected: null,
  imageHW: [0, 0],
  runId: null,
  runTimer: null,
  runLines: 0,
};

const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* not json */ }
    throw new Error(detail);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
};

const status = (msg, isError = false) => {
  const el = $("status");
  el.textContent = msg ?? "";
  el.classList.toggle("err", isError);
};

/* Stable colour per object index so the sidebar swatch, the 2D box, the 3D
   wireframe and the floor-plan footprint all agree without passing colours
   around. */
const hue = (i) => (i * 47) % 360;
const colorOf = (i) => `hsl(${hue(i)} 78% 62%)`;
const colorHex = (i) => new THREE.Color(`hsl(${hue(i)}, 78%, 62%)`);

/* ------------------------------------------------------------------ rooms */

async function loadRooms() {
  const { rooms } = await api("/api/rooms");
  const sel = $("room-select");
  sel.innerHTML = "";

  if (!rooms.length) {
    sel.innerHTML = "<option>(no rooms in out/)</option>";
    status("No rooms found. Run the pipeline first.", true);
    return;
  }
  for (const r of rooms) {
    const o = document.createElement("option");
    o.value = r.name;
    o.textContent = `${r.name}  (${r.n_objects ?? 0} objects)`;
    sel.appendChild(o);
  }
  sel.value = rooms[0].name;
  await selectRoom(rooms[0].name);
}

async function selectRoom(name) {
  S.room = name;
  S.selected = null;
  status("loading…");

  try {
    S.summary = await api(`/api/rooms/${name}`);
  } catch (e) {
    status(e.message, true);
    return;
  }

  const t = S.summary.trajectory;
  $("room-stats").innerHTML = [
    `<span>frames <b>${S.summary.n_frames ?? 0}</b></span>`,
    `<span>points <b>${((S.summary.n_points ?? 0) / 1000).toFixed(0)}k</b></span>`,
    `<span>objects <b>${S.summary.n_objects ?? 0}</b></span>`,
    t ? `<span>camera span <b>${t.span_max} m</b></span>` : "",
  ].join("");

  S.objects = [];
  S.observations = [];
  S.obsByFrame = new Map();

  try {
    S.objects = (await api(`/api/rooms/${name}/objects`)).objects ?? [];
  } catch { /* room may be unlabeled */ }

  try {
    const doc = await api(`/api/rooms/${name}/observations`);
    S.observations = doc.observations ?? [];
    S.imageHW = doc.image_hw ?? [0, 0];
    for (const o of S.observations) {
      if (!S.obsByFrame.has(o.frame_idx)) S.obsByFrame.set(o.frame_idx, []);
      S.obsByFrame.get(o.frame_idx).push(o);
    }
    $("frames-note").textContent = `${S.observations.length} detections`;
  } catch (e) {
    $("frames-note").textContent = "no observations.json — boxes unavailable";
  }

  try {
    S.frames = (await api(`/api/rooms/${name}/frames`)).frames ?? [];
  } catch { S.frames = []; }

  try {
    S.trajectory = await api(`/api/rooms/${name}/trajectory`);
  } catch { S.trajectory = null; }

  renderObjectList();
  renderFrames();
  renderTrajectory();
  cloud.invalidate();
  plan.invalidate();
  refreshActiveTab();
  status("");
}

/* ------------------------------------------------------------- object list */

function visibleObjects() {
  const q = $("object-filter").value.trim().toLowerCase();
  const minConf = parseFloat($("filter-conf").value);
  const minObs = parseInt($("filter-obs").value, 10);

  return S.objects
    .map((o, i) => ({ o, i }))
    .filter(({ o }) =>
      o.confidence >= minConf &&
      o.n_observations >= minObs &&
      (!q || o.label.toLowerCase().includes(q) ||
        (o.aliases ?? []).some((a) => a.toLowerCase().includes(q))));
}

function renderObjectList() {
  const list = $("object-list");
  const shown = visibleObjects();
  $("object-count").textContent = `${shown.length}/${S.objects.length}`;
  list.innerHTML = "";

  if (!shown.length) {
    list.innerHTML = `<li class="empty">No objects match.</li>`;
    return;
  }

  for (const { o, i } of shown) {
    const li = document.createElement("li");
    li.className = S.selected === o.id ? "selected" : "";
    li.onclick = () => selectObject(S.selected === o.id ? null : o.id);
    li.innerHTML = `
      <span class="swatch" style="background:${colorOf(i)}"></span>
      <span class="obj-label">${o.label}${
        o.aliases?.length ? ` <span class="obj-alias">${o.aliases.join(", ")}</span>` : ""
      }</span>
      <span class="obj-meta">${o.confidence.toFixed(2)} · ${o.n_observations}×</span>`;
    list.appendChild(li);
  }
}

function selectObject(id) {
  S.selected = id;
  renderObjectList();
  renderFrames();
  cloud.highlight();
  plan.highlight();
}

const objectIndex = (id) => S.objects.findIndex((o) => o.id === id);

/* ---------------------------------------------------------------- frames */

function renderFrames() {
  const grid = $("frame-grid");
  if (!S.frames.length) {
    grid.innerHTML = `<div class="empty">No frames for this room.</div>`;
    return;
  }

  const showBoxes = $("show-boxes").checked;
  const dim = $("dim-unselected").checked;
  const [h, w] = S.imageHW;
  grid.innerHTML = "";

  for (const f of S.frames) {
    const obs = S.obsByFrame.get(f.index) ?? [];
    const mine = obs.filter((o) => o.object_id === S.selected);

    const card = document.createElement("div");
    card.className = "frame-card" + (S.selected && mine.length ? " has-selection" : "");

    let svg = "";
    if (showBoxes && obs.length && w && h) {
      const parts = obs.map((o) => {
        if (!o.box_px) return "";
        const [x0, y0, x1, y1] = o.box_px;
        const idx = objectIndex(o.object_id);
        const c = idx >= 0 ? colorOf(idx) : "#888";
        const faded = dim && S.selected && o.object_id !== S.selected ? " dim" : "";
        return `<rect class="box${faded}" x="${x0}" y="${y0}" width="${x1 - x0}" height="${
          y1 - y0}" stroke="${c}"></rect>
          <text class="box-label${faded}" x="${x0 + 3}" y="${y0 + 13}">${o.label}</text>`;
      });
      svg = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${parts.join("")}</svg>`;
    }

    card.innerHTML = `
      <div class="frame-head"><span>frame ${f.index}</span><span>${obs.length} det</span></div>
      <div class="frame-holder">
        <img loading="lazy" src="/api/rooms/${S.room}/frames/${f.index}" alt="frame ${f.index}">
        ${svg}
      </div>`;
    grid.appendChild(card);
  }
}

/* ------------------------------------------------------- minimal 3D orbit */

function makeViewport(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x0f1114);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 500);

  const target = new THREE.Vector3();
  const spherical = new THREE.Spherical(4, Math.PI / 2.4, 0.7);
  let dragging = null, lastX = 0, lastY = 0;

  function apply() {
    camera.position.setFromSpherical(spherical).add(target);
    camera.lookAt(target);
  }

  const el = renderer.domElement;
  el.style.touchAction = "none";
  el.addEventListener("pointerdown", (e) => {
    dragging = e.button === 2 || e.shiftKey ? "pan" : "orbit";
    lastX = e.clientX; lastY = e.clientY;
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener("pointerup", (e) => {
    dragging = null;
    el.releasePointerCapture(e.pointerId);
  });
  el.addEventListener("contextmenu", (e) => e.preventDefault());
  el.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;

    if (dragging === "orbit") {
      spherical.theta -= dx * 0.005;
      spherical.phi = Math.max(0.02, Math.min(Math.PI - 0.02, spherical.phi - dy * 0.005));
    } else {
      // Pan in the camera's own plane, scaled by distance so it feels the same
      // whether you are across the room or inside a sofa.
      const k = spherical.radius * 0.0016;
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
      target.addScaledVector(right, -dx * k).addScaledVector(up, dy * k);
    }
    apply();
  });
  el.addEventListener("wheel", (e) => {
    e.preventDefault();
    spherical.radius = Math.max(0.05, Math.min(200, spherical.radius * (1 + Math.sign(e.deltaY) * 0.1)));
    apply();
  }, { passive: false });

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);

  function frame(center, radius) {
    target.copy(center);
    spherical.radius = Math.max(radius * 2.2, 0.4);
    apply();
  }

  apply(); resize();
  renderer.setAnimationLoop(() => renderer.render(scene, camera));

  return { scene, camera, renderer, frame, resize, target, spherical, apply };
}

function boxWireframe(obb, color) {
  const g = new THREE.BoxGeometry(...obb.extent.map((e) => Math.max(e, 0.01)));
  const wire = new THREE.LineSegments(
    new THREE.EdgesGeometry(g),
    new THREE.LineBasicMaterial({ color })
  );
  const m = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(obb.R[0][0], obb.R[1][0], obb.R[2][0]),
    new THREE.Vector3(obb.R[0][1], obb.R[1][1], obb.R[2][1]),
    new THREE.Vector3(obb.R[0][2], obb.R[1][2], obb.R[2][2])
  );
  wire.quaternion.setFromRotationMatrix(m);
  wire.position.set(...obb.center);
  return wire;
}

/* ----------------------------------------------------------- cloud viewer */

const cloud = (() => {
  let vp = null, points = null, boxGroup = null, camGroup = null;
  let dirty = true, loadedFor = null, bounds = null;

  const invalidate = () => { dirty = true; };

  async function show() {
    if (!vp) vp = makeViewport($("cloud-canvas"));
    vp.resize();
    if (!dirty || !S.room) return;
    dirty = false;

    const maxPoints = parseInt($("max-points").value, 10);
    const key = `${S.room}:${maxPoints}`;
    if (loadedFor !== key) {
      await loadPoints(maxPoints);
      loadedFor = key;
    }
    drawBoxes();
    drawCameras();
  }

  async function loadPoints(maxPoints) {
    $("cloud-note").textContent = "loading cloud…";
    let buf;
    try {
      buf = await (await fetch(`/api/rooms/${S.room}/cloud?max_points=${maxPoints}`)).arrayBuffer();
    } catch (e) {
      $("cloud-note").textContent = "failed to load cloud";
      return;
    }

    const n = new DataView(buf).getUint32(0, true);
    const xyz = new Float32Array(buf, 4, n * 3);
    const rgb = new Uint8Array(buf, 4 + n * 12, n * 3);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(rgb, 3, true));
    geo.computeBoundingSphere();

    if (points) { vp.scene.remove(points); points.geometry.dispose(); points.material.dispose(); }
    points = new THREE.Points(
      geo,
      new THREE.PointsMaterial({
        size: parseFloat($("point-size").value),
        vertexColors: true,
        sizeAttenuation: true,
      })
    );
    vp.scene.add(points);

    bounds = geo.boundingSphere;
    vp.frame(bounds.center, bounds.radius);
    $("cloud-note").textContent = `${(n / 1000).toFixed(0)}k points`;
  }

  function drawBoxes() {
    // The sidebar filters call this on every keystroke, which happens long
    // before the Cloud tab is first opened and the viewport exists.
    if (!vp) return;
    if (boxGroup) vp.scene.remove(boxGroup);
    boxGroup = new THREE.Group();
    if ($("show-boxes-3d").checked) {
      const shown = new Set(visibleObjects().map(({ o }) => o.id));
      S.objects.forEach((o, i) => {
        if (!shown.has(o.id)) return;
        const wire = boxWireframe(o.obb, colorHex(i));
        wire.userData.id = o.id;
        boxGroup.add(wire);
      });
    }
    vp.scene.add(boxGroup);
    highlight();
  }

  function drawCameras() {
    if (!vp) return;
    if (camGroup) vp.scene.remove(camGroup);
    camGroup = new THREE.Group();
    if ($("show-cams").checked && S.trajectory?.poses?.length) {
      const pts = S.trajectory.poses.map((p) => new THREE.Vector3(...p.position));
      camGroup.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: 0xffb454 })
      ));
      for (const p of pts) {
        const s = new THREE.Mesh(
          new THREE.SphereGeometry(0.02, 8, 8),
          new THREE.MeshBasicMaterial({ color: 0xffb454 })
        );
        s.position.copy(p);
        camGroup.add(s);
      }
    }
    vp.scene.add(camGroup);
  }

  function highlight() {
    if (!vp || !boxGroup) return;
    for (const wire of boxGroup.children) {
      const isSel = S.selected && wire.userData.id === S.selected;
      wire.material.opacity = !S.selected || isSel ? 1 : 0.18;
      wire.material.transparent = true;
      wire.material.depthTest = !isSel;   // selected box shows through the cloud
    }
    if (S.selected) {
      const o = S.objects.find((x) => x.id === S.selected);
      if (o && vp) {
        vp.frame(new THREE.Vector3(...o.centroid),
          Math.max(0.35, Math.hypot(...o.obb.extent) * 0.6));
      }
    }
  }

  function recentre() {
    if (vp && bounds) vp.frame(bounds.center, bounds.radius);
  }

  return { show, invalidate, highlight, drawBoxes, recentre,
    setPointSize: (v) => { if (points) points.material.size = v; } };
})();

/* ------------------------------------------------------------- floor plan */

const plan = (() => {
  let dirty = true, meta = null;
  const invalidate = () => { dirty = true; };

  async function show() {
    if (!dirty || !S.room) return;
    dirty = false;

    const wrap = $("plan-wrap");
    const up = $("plan-up").value;
    const ceil = $("plan-ceiling").value;
    wrap.innerHTML = `<div class="empty">rendering…</div>`;

    try {
      meta = await api(`/api/rooms/${S.room}/floorplan/meta${up ? `?up=${up}` : ""}`);
    } catch (e) {
      wrap.innerHTML = `<div class="empty">${e.message}</div>`;
      return;
    }

    const est = meta.up_estimate;
    $("plan-note").textContent = up
      ? `up = ${up} (manual)`
      : `up = ${est.axis} (${est.source}, confidence ${est.confidence})`;

    const src = `/api/rooms/${S.room}/floorplan.png?size=900&drop_ceiling=${ceil}${
      up ? `&up=${up}` : ""}`;

    // A plan can only be drawn against a coordinate axis, so an unlevelled room
    // gets a sheared projection no matter how good the up estimate is. Say
    // which of the two problems this is, because the fixes differ: levelling is
    // a one-line command, a bad estimate is not.
    // The warning lives outside .plan-wrap: the SVG overlay is positioned
    // against that container, so anything else inside it shifts the boxes off
    // the image.
    let warning = "";
    if (!up && !est.levelled && est.snap_deg > 3) {
      warning = `<div class="notice"><b>This room is not levelled.</b> Up points
        ${est.snap_deg}° off the nearest axis, so the plan below is a sheared
        projection. Fix it once with
        <code>uv run room3d level --room ${S.room}</code> — it is a rigid
        transform, so no geometry changes.</div>`;
    } else if (!up && est.confidence < 0.3) {
      warning = `<div class="notice"><b>Up axis is a guess.</b> Confidence
        ${est.confidence} (${est.source}) — ${
        est.notes[0] || "neither the camera poses nor a floor plane pinned it down"
        }. Pick an axis manually above.</div>`;
    }
    $("plan-warning").innerHTML = warning;

    wrap.innerHTML =
      `<img src="${src}" alt="floor plan">
       <svg viewBox="0 0 ${meta.transform.width} ${meta.transform.height}"
            preserveAspectRatio="none"></svg>`;
    drawFootprints();
  }

  function drawFootprints() {
    const svg = $("plan-wrap").querySelector("svg");
    if (!svg || !meta) return;
    if (!$("plan-footprints").checked) { svg.innerHTML = ""; return; }

    const shown = new Set(visibleObjects().map(({ o }) => o.id));
    svg.innerHTML = meta.footprints
      .filter((f) => shown.has(f.id))
      .map((f) => {
        const sel = S.selected === f.id;
        const dim = S.selected && !sel ? " dim" : "";
        const pts = f.hull.map((p) => p.join(",")).join(" ");
        return `<polygon class="foot${sel ? " selected" : ""}${dim}" points="${pts}"
                  data-id="${f.id}"><title>${f.label}</title></polygon>`;
      })
      .join("");

    for (const el of svg.querySelectorAll("polygon")) {
      el.onclick = () => selectObject(S.selected === el.dataset.id ? null : el.dataset.id);
    }
  }

  return { show, invalidate, highlight: drawFootprints, redraw: drawFootprints };
})();

/* ------------------------------------------------------------- trajectory */

let trajVp = null;

function renderTrajectory() {
  const t = S.trajectory;
  if (!t) {
    $("traj-stats").innerHTML = `<div class="empty">No trajectory for this room.</div>`;
    $("traj-table").innerHTML = "";
    $("traj-warning").innerHTML = "";
    return;
  }

  const s = t.stats;
  // A walkthrough needs parallax. Under ~1 m of camera travel, depth is poorly
  // constrained and fusion cannot match an object between views -- which shows
  // up as many single-observation duplicates rather than as an obvious error.
  const tooStill = s.span_max < 1.0;

  $("traj-stats").innerHTML = [
    ["poses", s.n_poses],
    ["camera span (max)", `${s.span_max} m`],
    ["path length", `${s.path_length} m`],
    ["mean step", `${s.mean_step} m`],
  ].map(([k, v], i) =>
    `<div class="stat${i === 1 && tooStill ? " alert" : ""}">
       <div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

  $("traj-warning").innerHTML = tooStill
    ? `<div class="notice"><b>The camera barely moved.</b> Span is
       ${s.span_max} m across ${s.n_poses} frames, so there is little parallax.
       Depth is poorly constrained and fusion cannot match an object between
       views — which is what produces many single-observation duplicates.
       <br>Walk <em>through</em> the room rather than panning from one spot.</div>`
    : "";

  $("traj-table").innerHTML =
    `<thead><tr><th>frame</th><th>x</th><th>y</th><th>z</th><th>step (m)</th></tr></thead>
     <tbody>${t.poses.map((p) => `
       <tr class="${p.step > 0 && p.step < 0.05 ? "tiny" : ""}">
         <td>${p.index}</td>
         ${p.position.map((v) => `<td>${v.toFixed(3)}</td>`).join("")}
         <td>${p.step.toFixed(3)}</td></tr>`).join("")}</tbody>`;

  drawTrajectory3D();
}

function drawTrajectory3D() {
  const t = S.trajectory;
  if (!t?.poses?.length) return;
  if (!trajVp) trajVp = makeViewport($("traj-canvas"));
  trajVp.resize();

  trajVp.scene.clear();
  const pts = t.poses.map((p) => new THREE.Vector3(...p.position));

  trajVp.scene.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0xffb454 })
  ));

  pts.forEach((p, i) => {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(0.015, 10, 10),
      new THREE.MeshBasicMaterial({ color: i === 0 ? 0x5ad19b : 0xffb454 })
    );
    m.position.copy(p);
    trajVp.scene.add(m);
  });

  // A 1 m reference grid, so "0.6 m of travel" is a size you can see rather
  // than a number you have to trust.
  const grid = new THREE.GridHelper(4, 4, 0x2c313a, 0x21252c);
  const centre = pts.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(pts.length);
  grid.position.set(centre.x, Math.min(...pts.map((p) => p.y)) - 0.05, centre.z);
  trajVp.scene.add(grid);

  const box = new THREE.Box3().setFromPoints(pts);
  trajVp.frame(box.getCenter(new THREE.Vector3()),
    Math.max(box.getSize(new THREE.Vector3()).length() * 0.5, 0.6));
}

/* -------------------------------------------------------------- re-fusion */

async function doRefuse(save = false) {
  if (!S.room) return;
  const body = {
    radius_floor: parseFloat($("rf").value),
    radius_scale: parseFloat($("rs").value),
    min_obb_iou: parseFloat($("iou").value),
    save,
  };

  $("btn-refuse").disabled = $("btn-save").disabled = true;
  try {
    const res = await api(`/api/rooms/${S.room}/refuse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    S.objects = res.objects;
    S.selected = null;
    $("refuse-result").textContent =
      `${res.previous_count} → ${res.objects.length} objects from ${res.n_observations} observations` +
      (save ? " · saved (backup: objects.prev.json)" : "");

    // The floor and the size term compete; whichever wins is what actually
    // decides the merge. Saying which one is in charge is the difference
    // between a slider that works and a slider that appears not to.
    const r = res.radius;
    if (r) {
      $("radius-note").textContent =
        `Effective radius ${r.effective_radius} m (driven by ${r.driven_by}; ` +
        `mean object diagonal ${r.mean_obb_diagonal} m). Non-overlapping boxes ` +
        `merge within ${r.merge_distance_no_overlap} m.`;
    }

    renderObjectList();
    renderFrames();
    cloud.drawBoxes();
    plan.invalidate();
    if ($("tabs").querySelector(".active").dataset.tab === "plan") plan.show();
  } catch (e) {
    $("refuse-result").textContent = e.message;
  } finally {
    $("btn-refuse").disabled = $("btn-save").disabled = false;
  }
}

/* -------------------------------------------------------------------- run */

async function loadVideos() {
  try {
    const { videos } = await api("/api/videos");
    const sel = $("run-video");
    sel.innerHTML = videos.length
      ? videos.map((v) => `<option value="${v.path}">${v.path} (${v.mb} MB)</option>`).join("")
      : `<option value="">(no videos found)</option>`;
  } catch { /* non-fatal */ }
}

function estimateRun() {
  const n = parseInt($("run-frames").value, 10) || 8;
  // swin-3, symmetrised: 2 * sum(min(3, n-1-i)); ~13.5 s/pair measured on CPU.
  let pairs = 0;
  for (let i = 0; i < n; i++) pairs += Math.min(3, n - 1 - i);
  pairs *= 2;
  const mins = (pairs * 13.5) / 60 + n * 0.5;
  $("run-estimate").textContent =
    `${n} frames → ~${pairs} pairs → roughly ${Math.round(mins)} min on this CPU.`;
}

async function startRun() {
  const body = {
    source: $("run-video").value,
    room: $("run-room").value.trim(),
    n_frames: parseInt($("run-frames").value, 10),
    direct: $("run-direct").checked,
    allow_mixed: $("run-mixed").checked,
  };
  if (!body.source || !body.room) {
    $("run-log").textContent = "Pick a video and enter a room name.";
    return;
  }

  try {
    const run = await api("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    S.runId = run.id;
    S.runLines = 0;
    $("run-log").textContent = "";
    pollRun();
  } catch (e) {
    $("run-log").textContent = `Could not start: ${e.message}`;
  }
}

async function pollRun() {
  clearTimeout(S.runTimer);
  if (!S.runId) return;

  try {
    const snap = await api(`/api/runs/${S.runId}?since=${S.runLines}`);
    if (snap.lines.length) {
      $("run-log").textContent += snap.lines.join("\n") + "\n";
      $("run-log").scrollTop = $("run-log").scrollHeight;
      S.runLines = snap.total_lines;
    }

    const stages = ["extract", "reconstruct", "detect", "project", "cluster", "label"];
    $("run-progress").innerHTML = stages.map((s, i) => {
      let cls = "stage";
      if (snap.status === "failed" && i === snap.stage_index) cls += " failed";
      else if (i < snap.stage_index) cls += " done";
      else if (i === snap.stage_index) cls += snap.status === "done" ? " done" : " current";
      return `<div class="${cls}">${s}</div>`;
    }).join("");

    status(`run ${snap.status} · ${snap.stage} · ${snap.elapsed}s`);

    if (snap.status === "running") {
      S.runTimer = setTimeout(pollRun, 1200);
    } else {
      S.runId = null;
      status(`run ${snap.status}`);
      await loadRooms();
    }
  } catch (e) {
    status(e.message, true);
    S.runId = null;
  }
}

/* ------------------------------------------------------------------- tabs */

function refreshActiveTab() {
  const tab = $("tabs").querySelector(".active")?.dataset.tab;
  if (tab === "cloud") cloud.show();
  if (tab === "plan") plan.show();
  if (tab === "traj") { drawTrajectory3D(); trajVp?.resize(); }
}

function initTabs() {
  for (const btn of $("tabs").querySelectorAll("button")) {
    btn.onclick = () => {
      for (const b of $("tabs").querySelectorAll("button")) b.classList.remove("active");
      for (const p of document.querySelectorAll(".panel")) p.classList.remove("active");
      btn.classList.add("active");
      document.querySelector(`[data-panel="${btn.dataset.tab}"]`).classList.add("active");
      refreshActiveTab();
    };
  }
}

/* ------------------------------------------------------------------- wire */

function init() {
  initTabs();

  $("room-select").onchange = (e) => selectRoom(e.target.value);
  $("clear-selection").onclick = () => selectObject(null);
  $("object-filter").oninput = () => { renderObjectList(); cloud.drawBoxes(); plan.redraw(); };

  const bindFilter = (id, out, fmt) => {
    $(id).oninput = () => {
      $(out).textContent = fmt($(id).value);
      renderObjectList(); cloud.drawBoxes(); plan.redraw();
    };
  };
  bindFilter("filter-conf", "out-conf", (v) => (+v).toFixed(2));
  bindFilter("filter-obs", "out-obs", (v) => v);

  for (const [id, out, fmt] of [
    ["rf", "out-rf", (v) => (+v).toFixed(2)],
    ["rs", "out-rs", (v) => (+v).toFixed(2)],
    ["iou", "out-iou", (v) => (+v).toFixed(2)],
  ]) $(id).oninput = () => ($(out).textContent = fmt($(id).value));

  $("btn-refuse").onclick = () => doRefuse(false);
  $("btn-save").onclick = () => doRefuse(true);
  $("btn-reset").onclick = () => {
    $("rf").value = 0.3; $("rs").value = 0.5; $("iou").value = 0.1;
    $("out-rf").textContent = "0.30"; $("out-rs").textContent = "0.50";
    $("out-iou").textContent = "0.10";
    doRefuse(false);
  };

  $("show-boxes").onchange = renderFrames;
  $("dim-unselected").onchange = renderFrames;

  $("max-points").oninput = () => {
    $("out-maxpts").textContent = `${(+$("max-points").value / 1000).toFixed(0)}k`;
  };
  $("max-points").onchange = () => { cloud.invalidate(); cloud.show(); };
  $("point-size").oninput = () => cloud.setPointSize(parseFloat($("point-size").value));
  $("show-boxes-3d").onchange = () => cloud.drawBoxes();
  $("show-cams").onchange = () => { cloud.invalidate(); cloud.show(); };
  $("btn-recentre").onclick = () => cloud.recentre();

  $("plan-up").onchange = () => { plan.invalidate(); plan.show(); };
  $("plan-ceiling").oninput = () => {
    $("out-ceil").textContent = `${Math.round($("plan-ceiling").value * 100)}%`;
  };
  $("plan-ceiling").onchange = () => { plan.invalidate(); plan.show(); };
  $("plan-footprints").onchange = () => plan.redraw();

  $("btn-run").onclick = startRun;
  $("btn-cancel").onclick = async () => {
    if (S.runId) { await api(`/api/runs/${S.runId}/cancel`, { method: "POST" }).catch(() => {}); }
  };
  $("run-frames").oninput = estimateRun;

  estimateRun();
  loadVideos();
  loadRooms();
}

init();
