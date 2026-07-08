import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/**
 * Compute a camera target and distance from the raw xyz point cloud.
 * This is only for framing the Three.js camera; it does not alter point data.
 */
function pointCloudBounds(points) {
  if (!points?.length) {
    return {
      center: new THREE.Vector3(2.5, 0, 0),
      radius: 5,
    };
  }

  const xs = points.map((point) => point.x).filter(Number.isFinite);
  const ys = points.map((point) => point.y).filter(Number.isFinite);
  const zs = points.map((point) => point.z).filter(Number.isFinite);
  if (!xs.length || !ys.length || !zs.length) {
    return {
      center: new THREE.Vector3(2.5, 0, 0),
      radius: 5,
    };
  }

  const min = new THREE.Vector3(Math.min(...xs), Math.min(...ys), Math.min(...zs));
  const max = new THREE.Vector3(Math.max(...xs), Math.max(...ys), Math.max(...zs));
  const center = min.clone().add(max).multiplyScalar(0.5);
  const radius = Math.max(1, max.distanceTo(min) * 0.65);
  return { center, radius };
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

export function RadarPointCloud({ points }) {
  const mountRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const cloudRef = useRef(null);
  const userMovedCameraRef = useRef(false);

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
    controls.addEventListener("start", () => {
      // Once a user rotates/zooms manually, stop auto-fitting the camera on
      // every frame so interaction does not fight the live data stream.
      userMovedCameraRef.current = true;
    });
    controlsRef.current = controls;

    const grid = new THREE.GridHelper(10, 10, "#9aa5b1", "#d7dce2");
    // Three.js GridHelper is x/z by default. Rotate it to the radar x/y ground
    // plane so z remains vertical.
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    scene.add(new THREE.AxesHelper(2));

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

    if (!userMovedCameraRef.current) {
      // Keep the first view centered on incoming data. After manual camera
      // movement, preserve the user's chosen viewpoint.
      const { center, radius } = pointCloudBounds(points);
      controls.target.copy(center);
      camera.position.set(center.x + radius * 1.3, center.y - radius * 1.7, center.z + radius);
      camera.near = Math.max(0.01, radius / 100);
      camera.far = Math.max(100, radius * 20);
      camera.updateProjectionMatrix();
      controls.update();
    }
  }, [points]);

  return (
    <section className="panel pointcloud-panel">
      <div className="panel-label">Radar Point Cloud</div>
      <div ref={mountRef} className="pointcloud-scene" />
    </section>
  );
}
