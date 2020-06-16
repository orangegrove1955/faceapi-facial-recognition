const imageUpload = document.getElementById("image-upload");
const imageWrapper = document.getElementById("image");

const startRecognition = () => {
  document.body.append("Loaded");
  // Add event listener to determine when file is uploaded
  imageUpload.addEventListener("change", async () => {
    const image = await faceapi.bufferToImage(imageUpload.files[0]);

    console.log(image);
    image.style.width = "100%";
    imageWrapper.append(image);
    const detections = await faceapi
      .detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceDescriptors();
  });
};

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
]).then(startRecognition);
