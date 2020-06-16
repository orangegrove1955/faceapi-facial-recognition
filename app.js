const imageUpload = document.getElementById("image-upload");

const startRecognition = () => {
  document.body.append("Loaded");
};

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
]).then(startRecognition);
