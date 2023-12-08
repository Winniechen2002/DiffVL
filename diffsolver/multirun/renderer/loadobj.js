import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'https://unpkg.com/three@v0.149.0/examples/jsm/loaders/OBJLoader.js';


// Set up scene, camera, and renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);


// Initialize the OBJLoader
const loader = new OBJLoader();

// Create and add ambient light
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

// Create and add directional light
const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(1, 1, 1);
scene.add(directionalLight);

// Load the OBJ file
loader.load(
    'bunny.obj', // URL of the OBJ file
    (object) => {
        // Add the loaded object to the scene
        // Traverse the object and change the material color to green
        object.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                child.material.color.set(0x00ff00); // Green color
            }
        });
        // Add the loaded object to the scene
        object.scale.set(100, 100, 100);
        scene.add(object);

        // Calculate the bounding box of the object
        const boundingBox = new THREE.Box3().setFromObject(object);
        console.log(boundingBox);

        // Calculate the center of the bounding box
        const center = new THREE.Vector3();
        boundingBox.getCenter(center);

        // Set the camera position
        const cameraDistance = 1.5; // You can adjust this value to change the distance between the camera and the object
        const size = boundingBox.getSize(new THREE.Vector3());
        const maxDimension = Math.max(size.x, size.y, size.z);
        const cameraPosition = center.clone().add(new THREE.Vector3(0, 0, maxDimension * cameraDistance));

        camera.position.copy(cameraPosition);
        camera.lookAt(center);
    },
    (xhr) => {
        // Optional: show loading progress
        console.log((xhr.loaded / xhr.total) * 100 + '% loaded');
    },
    (error) => {
        // Handle errors during loading
        console.error('An error occurred while loading the OBJ file:', error);
    }
);

// Set camera position
camera.position.z = 10;

// Add OrbitControls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Animation loop
function animate() {
    requestAnimationFrame(animate);

    controls.update();

    renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener('resize', function () {
    const width = window.innerWidth;
    const height = window.innerHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
});

