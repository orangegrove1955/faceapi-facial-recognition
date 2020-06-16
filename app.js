const imageUpload = document.getElementById("image-upload");
const imageWrapper = document.getElementById("image");
imageWrapper.style.position = "relative";

const startRecognition = () => {
  // TODO: Change to loading animation that stopped when ready
  document.body.append("Loaded");

  // Add event listener to determine when file is uploaded
  imageUpload.addEventListener("change", async () => {
    const image = await faceapi.bufferToImage(imageUpload.files[0]);

    image.style.width = "100%";
    imageWrapper.append(image);

    // Create canvas to overlay image
    const canvas = faceapi.createCanvasFromMedia(image);
    imageWrapper.append(canvas);

    // Change dimensions to overlay correctly
    const displaySize = { width: image.width, height: image.height };
    faceapi.matchDimensions(canvas, displaySize);

    // Get detections from image
    const detections = await faceapi
      .detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceDescriptors();

    // Resize detections to fit onto canvas, then create box for each
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    resizedDetections.forEach((detection) => {
      const box = detection.detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, { label: "Face" });
      drawBox.draw(canvas);
    });
  });
};

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
]).then(startRecognition);
