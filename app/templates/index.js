let video = document.querySelector(".video-container video");
let onBtn = document.querySelector(".toggle-btn.active");
let offBtn = document.querySelector(".toggle-btn.off");
let recallElement = document.getElementById("recall");
let precisionElement = document.getElementById("precision");
let f1ScoreElement = document.getElementById("f1-score");
let recallGraph = document.querySelector(".recall-graph .fill");
let precisionGraph = document.querySelector(".precision-graph .fill");
let f1ScoreGraph = document.querySelector(".f1-score-graph .fill");
let fileInput = document.querySelector("#file");
let mediaContainer = document.querySelector("#media-container");
let stream = null;

onBtn.addEventListener("click", function () {
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices
            .getUserMedia({ video: true })
            .then(function (mediaStream) {
                stream = mediaStream;
                video.srcObject = mediaStream;
                video.play();
                video.style.width = "100%";
                video.style.height = "100%";
            })
            .catch(function (error) {
                console.error("Error accessing webcam:", error);
            });
    } else {
        console.error("Media devices not supported!");
    }
});

offBtn.addEventListener("click", function () {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        video.src = "videoplayback.mp4";
        video.play();
    }
});

function updateGraph(element, value) {
    element.style.height = value + "%";
}

function fetchMetrics() {
    fetch("/metrics") // Change to your actual Flask endpoint
        .then(response => response.json())
        .then(data => {
            if (data.recall) {
                recallElement.textContent = data.recall + "%";
                updateGraph(recallGraph, data.recall);
            }
            if (data.precision) {
                precisionElement.textContent = data.precision + "%";
                updateGraph(precisionGraph, data.precision);
            }
            if (data.f1_score) {
                f1ScoreElement.textContent = data.f1_score + "%";
                updateGraph(f1ScoreGraph, data.f1_score);
            }
        })
        .catch(error => {
            console.error("Failed to fetch metrics:", error);
        });
}

setInterval(fetchMetrics, 5000);

fileInput.addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (!file) return;
    const fileURL = URL.createObjectURL(file);
    if (file.type.startsWith("image/")) {
        mediaContainer.innerHTML = `<img src="${fileURL}" alt="Uploaded Image" style="max-width: 100%; height: auto;">`;
    } else if (file.type.startsWith("video/")) {
        mediaContainer.innerHTML = `<video src="${fileURL}" controls autoplay loop muted style="max-width: 100%;"></video>`;
    }
});
