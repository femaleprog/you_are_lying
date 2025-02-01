function analyzeStory() {
    const story = document.getElementById("storyInput").value;
    const responseContainer = document.getElementById("responseContainer");
    const isCoherentEl = document.getElementById("isCoherent");
    const feedbackEl = document.getElementById("feedback");

    if (!story.trim()) {
        alert("Please enter a story before analyzing.");
        return;
    }

    fetch("http://127.0.0.1:8000/analyze-story", {  // Make sure this matches your backend URL
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ story: story })
    })
    .then(response => response.json())
    .then(data => {
        isCoherentEl.textContent = data.is_coherent ? "Yes" : "No";
        feedbackEl.textContent = data.feedback;
        responseContainer.style.display = "block";
    })
    .catch(error => {
        console.error("Error:", error);
        alert("Failed to analyze the story. Please check if the backend is running.");
    });
}

