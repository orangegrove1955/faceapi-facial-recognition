const imageUpload = document.getElementById("image-upload");
const imageWrapper = document.getElementById("image");
const loader = document.getElementById("loading");
imageWrapper.style.position = "relative";

// Global declaration to be able to delete items if user changes image
let image;
let canvas;

const labeledImagesLocation =
  "https://raw.githubusercontent.com/orangegrove1955/faceapi-facial-recognition/master/labeled_images/";

/** When files are uploaded, display image and perform detections */
const onFileUpload = async () => {
  // If an image already exists from user uploading an item, remove it
  if (image) image.remove();
  if (canvas) canvas.remove();

  loader.style.display = "flex";
  image = await faceapi.bufferToImage(imageUpload.files[0]);

  image.style.width = "100%";
  image.style.height = "auto";
  imageWrapper.append(image);

  // Create canvas to overlay image
  canvas = faceapi.createCanvasFromMedia(image);
  imageWrapper.append(canvas);

  // Load Face Descriptions for recognition
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

  // Change dimensions to overlay correctly
  const displaySize = { width: image.width, height: image.height };
  faceapi.matchDimensions(canvas, displaySize);

  // Get detections from image
  const detections = await faceapi
    .detectAllFaces(image)
    .withFaceLandmarks()
    .withFaceDescriptors();

  // Resize detections to fit onto canvas, then find any matching results and create box for each
  const resizedDetections = faceapi.resizeResults(detections, displaySize);
  const results = resizedDetections.map((resizedDetection) =>
    faceMatcher.findBestMatch(resizedDetection.descriptor)
  );
  console.log("results", results);
  results.forEach((result, i) => {
    console.log("result", result);
    const box = resizedDetections[i].detection.box;
    const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
    drawBox.draw(canvas);
  });

  loader.style.display = "none";
};

/** Start recognition once all models are loaded from faceapi */
const startRecognition = () => {
  // TODO: Change to loading animation that stopped when ready
  loader.style.display = "none";

  // Add event listener to determine when file is uploaded
  imageUpload.addEventListener("change", onFileUpload);
};

/** Load images from labeled folder for facial recognition training */
const loadLabeledImages = () => {
  const labels = [
    "Black Widow",
    "Captain America",
    "Captain Marvel",
    "Hawkeye",
    "Jim Rhodes",
    "Thor",
    "Tony Stark",
  ];
  return Promise.all(
    // Load images for all labels
    labels.map(async (label) => {
      const descriptionOfFace = [];
      // There are two images for each face, so loop through both, getting image from internet
      // and using to get facial detections
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          `${labeledImagesLocation}/${label}/${i}.jpg`
        );
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptionOfFace.push(detections.descriptor);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptionOfFace);
    })
  );
};

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
]).then(startRecognition);
