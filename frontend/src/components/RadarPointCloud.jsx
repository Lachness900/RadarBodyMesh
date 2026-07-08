import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const POINT_MODES = [
  {
    key: "projected_radar",
    label: "Projected",
    description: "Filtered and flattened",
  },
  {
    key: "filtered_radar",
    label: "Filtered",
    description: "Filtered xyz",
  },
  {
    key: "raw_radar",
    label: "Raw",
    description: "Raw xyz",
  },
];

// Match point_visualizer/visualizer.py: -12m..+12m with 1m tick marks.
const AXIS_SIZE = 12;
const AXIS_TICK_SPACING = 1;
const RADAR_VIEW_DISTANCE = 5;
const RADAR_VIEW_TARGET = new THREE.Vector3(0, 0, 1);

/**
 * Reset to a stable radar view. The camera looks along the positive X
 * direction while focusing on a fixed point 5m above the origin.
 */
function resetCameraToRadarView(camera, controls) {
  camera.up.set(0, 0, 1);
  camera.position.set(-RADAR_VIEW_DISTANCE, 0, RADAR_VIEW_TARGET.z);
  camera.lookAt(RADAR_VIEW_TARGET);
  camera.near = 0.01;
  camera.far = 1000;
  camera.updateProjectionMatrix();
  controls.target.copy(RADAR_VIEW_TARGET);
  controls.update();
  controls.saveState();
}

function makeTextSprite(text, position, color = "#4b5961", size = 0.34) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.font = "700 42px Inter, Arial, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillStyle = color;
  context.fillText(text, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.position.copy(position);
  sprite.scale.set(size * 2, size, 1);
  return sprite;
}

function makeLineSegments(points, colors) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(points.flat(), 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors.flat(), 3));
  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.7,
  });
  return new THREE.LineSegments(geometry, material);
}

function addMetricAxes(scene) {
  const red = [0.9, 0.22, 0.22];
  const green = [0.25, 0.72, 0.25];
  const blue = [0.24, 0.45, 1.0];
  const tickColor = [0.38, 0.46, 0.52];
  const points = [
    [-AXIS_SIZE, 0, 0],
    [AXIS_SIZE, 0, 0],
    [0, -AXIS_SIZE, 0],
    [0, AXIS_SIZE, 0],
    [0, 0, -AXIS_SIZE],
    [0, 0, AXIS_SIZE],
  ];
  const colors = [red, red, green, green, blue, blue];
  const tickLength = AXIS_SIZE * 0.03;

  for (let value = -AXIS_SIZE; value <= AXIS_SIZE; value += AXIS_TICK_SPACING) {
    if (value === 0) continue;
    points.push([value, -tickLength, 0], [value, tickLength, 0]);
    colors.push(tickColor, tickColor);
    points.push([-tickLength, value, 0], [tickLength, value, 0]);
    colors.push(tickColor, tickColor);
    points.push([-tickLength, 0, value], [tickLength, 0, value]);
    colors.push(tickColor, tickColor);
  }

  scene.add(makeLineSegments(points, colors));
  for (let value = -AXIS_SIZE; value <= AXIS_SIZE; value += AXIS_TICK_SPACING) {
    if (value === 0) continue;
    scene.add(
      makeTextSprite(String(value), new THREE.Vector3(value, -tickLength * 3, 0), "#6b747c", 0.24),
    );
    scene.add(
      makeTextSprite(String(value), new THREE.Vector3(-tickLength * 3, value, 0), "#6b747c", 0.24),
    );
    scene.add(
      makeTextSprite(String(value), new THREE.Vector3(-tickLength * 3, 0, value), "#6b747c", 0.24),
    );
  }
  scene.add(makeTextSprite("X (m)", new THREE.Vector3(AXIS_SIZE + 0.75, 0, 0), "#d94b4b", 0.36));
  scene.add(makeTextSprite("Y (m)", new THREE.Vector3(0, AXIS_SIZE + 0.75, 0), "#339448", 0.36));
  scene.add(makeTextSprite("Z (m)", new THREE.Vector3(0, 0, AXIS_SIZE + 0.75), "#3c63d8", 0.36));
}

/**
 * Convert backend JSON points into a Three.js BufferGeometry.
 * Positions use the original x/y/z values directly so the frontend does not
 * manually project or flatten the radar point cloud.
 */
function pointsToGeometry(points) {
  const validPoints = (points || []).filter(
    (point) =>
      Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z),
  );
  const positions = new Float32Array(validPoints.length * 3);
  const colors = new Float32Array(validPoints.length * 3);
  const baseColor = new THREE.Color("#0f8b8d");
  const brightColor = new THREE.Color("#e4572e");

  validPoints.forEach((point, index) => {
    positions[index * 3] = point.x;
    positions[index * 3 + 1] = point.y;
    positions[index * 3 + 2] = point.z;

    const intensity = Math.max(0, Math.min(1, (point.intensity || 0) / 25));
    const color = baseColor.clone().lerp(brightColor, intensity);
    colors[index * 3] = color.r;
    colors[index * 3 + 1] = color.g;
    colors[index * 3 + 2] = color.b;
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return geometry;
}

export function RadarPointCloud({ mode, onModeChange, points, pointSets, viewKey }) {
  const mountRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const cloudRef = useRef(null);
  const fittedViewKeyRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#f8fafc");

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
    // Match the Python visualizer convention: z is height/up.
    camera.up.set(0, 0, 1);
    camera.position.set(6, -8, 5);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    // Allow free inspection by panning, rotating, and zooming. The Reset View
    // button restores the centered radar view when needed.
    controls.enablePan = true;
    controlsRef.current = controls;
    resetCameraToRadarView(camera, controls);

    const grid = new THREE.GridHelper(AXIS_SIZE * 2, AXIS_SIZE * 2, "#9aa5b1", "#d7dce2");
    // Three.js GridHelper is x/z by default. Rotate it to the radar x/y ground
    // plane so z remains vertical.
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    addMetricAxes(scene);

    const material = new THREE.PointsMaterial({
      size: 0.06,
      sizeAttenuation: true,
      vertexColors: true,
    });
    const cloud = new THREE.Points(new THREE.BufferGeometry(), material);
    scene.add(cloud);
    cloudRef.current = cloud;

    const resize = () => {
      const rect = mount.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      renderer.setSize(rect.width, rect.height, false);
      camera.aspect = rect.width / rect.height;
      camera.updateProjectionMatrix();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mount);
    resize();

    let animationFrame = 0;
    const animate = () => {
      // OrbitControls damping needs an animation loop even when data is static.
      controls.update();
      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(animate);
    };
    animate();

    return () => {
      window.cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      controls.dispose();
      cloud.geometry.dispose();
      material.dispose();
      const disposeMaterial = (objectMaterial) => {
        if (Array.isArray(objectMaterial)) {
          objectMaterial.forEach(disposeMaterial);
          return;
        }
        objectMaterial?.dispose?.();
      };

      scene.traverse((object) => {
        if (object.isSprite) {
          object.material.map?.dispose();
          disposeMaterial(object.material);
        }
        if (object.isLineSegments) {
          object.geometry.dispose();
          disposeMaterial(object.material);
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  useEffect(() => {
    const cloud = cloudRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!cloud || !camera || !controls) return;

    const oldGeometry = cloud.geometry;
    cloud.geometry = pointsToGeometry(points);
    oldGeometry.dispose();

    const fitKey = viewKey || mode;
    if (fittedViewKeyRef.current !== fitKey) {
      // File/source/mode changes return to the same radar coordinate baseline.
      // Continuous live frames still update points only, so the camera stays stable.
      resetCameraToRadarView(camera, controls);
      fittedViewKeyRef.current = fitKey;
    }
  }, [mode, points, viewKey]);

  const resetView = () => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    resetCameraToRadarView(camera, controls);
    fittedViewKeyRef.current = viewKey || mode;
  };

  return (
    <section className="panel pointcloud-panel">
      <div className="pointcloud-header">
        <div>
          <div className="panel-label">Radar Point Cloud</div>
          <div className="pointcloud-mode-note">
            {POINT_MODES.find((option) => option.key === mode)?.description}
          </div>
        </div>
        <div className="pointcloud-view-tools">
          <button
            className="reset-view-button"
            onClick={resetView}
            aria-label="Reset camera to centered radar view"
            title="Reset camera to the centered radar view"
            type="button"
          >
            Reset View
          </button>
        </div>
        <div className="pointcloud-modes" aria-label="Radar point cloud display mode">
          {POINT_MODES.map((option) => {
            const count = pointSets?.[option.key]?.length ?? 0;
            const isActive = mode === option.key;
            return (
              <button
                className={isActive ? "active" : ""}
                disabled={!count && option.key !== "raw_radar"}
                key={option.key}
                onClick={() => onModeChange(option.key)}
                type="button"
              >
                {option.label}
              </button>
            );
          })}
        </div>
      </div>
      <div ref={mountRef} className="pointcloud-scene" />
    </section>
  );
}
