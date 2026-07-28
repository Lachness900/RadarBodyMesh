import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const CAMERA_FOV = 35;
const VIEW_PADDING = 1.12;

function disposeMaterial(material) {
  if (Array.isArray(material)) {
    material.forEach(disposeMaterial);
    return;
  }
  if (!material) return;

  Object.values(material).forEach((value) => {
    if (value?.isTexture) value.dispose();
  });
  material.dispose();
}

function disposeModel(model) {
  model?.traverse((object) => {
    if (!object.isMesh) return;
    object.geometry.dispose();
    disposeMaterial(object.material);
  });
}

/**
 * Place the model on y=0 while preserving its original scale and orientation.
 * The returned target sits at the body's visual centre for stable orbiting.
 */
function centerModel(model) {
  const initialBox = new THREE.Box3().setFromObject(model);
  const initialCenter = initialBox.getCenter(new THREE.Vector3());

  model.position.x -= initialCenter.x;
  model.position.y -= initialBox.min.y;
  model.position.z -= initialCenter.z;
  model.updateMatrixWorld(true);

  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  return {
    size,
    target: new THREE.Vector3(0, box.min.y + size.y * 0.5, 0),
  };
}

/**
 * Fit the full T Pose using the smaller horizontal/vertical camera field.
 * This keeps the outstretched hands visible in the narrow prediction panel.
 */
function resetReferenceView(camera, controls, modelFrame) {
  if (!modelFrame) return;

  const verticalFov = THREE.MathUtils.degToRad(camera.fov);
  const horizontalFov =
    2 * Math.atan(Math.tan(verticalFov / 2) * Math.max(camera.aspect, 0.01));
  const widthDistance =
    modelFrame.size.x / (2 * Math.tan(Math.max(horizontalFov, 0.01) / 2));
  const heightDistance =
    modelFrame.size.y / (2 * Math.tan(Math.max(verticalFov, 0.01) / 2));
  const distance = Math.max(widthDistance, heightDistance) * VIEW_PADDING;

  camera.up.set(0, 1, 0);
  camera.position.set(0, modelFrame.target.y, distance);
  camera.near = Math.max(0.01, distance / 100);
  camera.far = Math.max(100, distance * 20);
  camera.lookAt(modelFrame.target);
  camera.updateProjectionMatrix();

  controls.target.copy(modelFrame.target);
  controls.minDistance = distance * 0.35;
  controls.maxDistance = distance * 3;
  controls.update();
  controls.saveState();
}

export function ReferencePoseViewer({ assetUrl, emptyLabel, poseLabel }) {
  const mountRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const modelFrameRef = useRef(null);
  const [loadState, setLoadState] = useState(assetUrl ? "loading" : "unavailable");

  useEffect(() => {
    if (!assetUrl) {
      setLoadState("unavailable");
      return undefined;
    }

    const mount = mountRef.current;
    if (!mount) return undefined;

    let disposed = false;
    let animationFrame = 0;
    let loadedModel = null;

    setLoadState("loading");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#f8fafc");

    const camera = new THREE.PerspectiveCamera(CAMERA_FOV, 1, 0.01, 100);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight("#f8fbff", "#7f8b8c", 2.2));
    const keyLight = new THREE.DirectionalLight("#ffffff", 3.2);
    keyLight.position.set(2.5, 3.5, 4);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight("#bdeff0", 1.4);
    fillLight.position.set(-3, 2, -2);
    scene.add(fillLight);

    const grid = new THREE.GridHelper(3, 10, "#aab5bc", "#dce2e5");
    grid.material.transparent = true;
    grid.material.opacity = 0.42;
    scene.add(grid);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controlsRef.current = controls;

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

    const loader = new GLTFLoader();
    loader.load(
      assetUrl,
      (gltf) => {
        if (disposed) {
          disposeModel(gltf.scene);
          return;
        }
        loadedModel = gltf.scene;
        loadedModel.name = `${poseLabel} Reference`;
        scene.add(loadedModel);
        modelFrameRef.current = centerModel(loadedModel);
        resetReferenceView(camera, controls, modelFrameRef.current);
        setLoadState("ready");
      },
      undefined,
      () => {
        if (!disposed) setLoadState("error");
      },
    );

    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(animate);
    };
    animate();

    return () => {
      disposed = true;
      window.cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      controls.dispose();
      disposeModel(loadedModel);
      grid.geometry.dispose();
      disposeMaterial(grid.material);
      renderer.dispose();
      renderer.domElement.remove();
      cameraRef.current = null;
      controlsRef.current = null;
      modelFrameRef.current = null;
    };
  }, [assetUrl, poseLabel]);

  const resetView = () => {
    resetReferenceView(
      cameraRef.current,
      controlsRef.current,
      modelFrameRef.current,
    );
  };

  if (!assetUrl) {
    return (
      <div className="reference-viewer reference-unavailable">
        <div className="reference-viewer-label">Reference Pose</div>
        <span>{emptyLabel}</span>
      </div>
    );
  }

  return (
    <div className="reference-viewer">
      <div className="reference-viewer-header">
        <div className="reference-viewer-label">Reference Pose</div>
        <button
          disabled={loadState !== "ready"}
          onClick={resetView}
          title="Reset reference pose view"
          type="button"
        >
          Reset
        </button>
      </div>
      <div
        ref={mountRef}
        className="reference-pose-scene"
        aria-label={`Interactive ${poseLabel} reference model`}
      />
      {loadState !== "ready" && (
        <div className={`reference-load-state ${loadState}`}>
          {loadState === "error" ? "Reference unavailable" : "Loading reference"}
        </div>
      )}
    </div>
  );
}
