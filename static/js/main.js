let selectedClass = null;

// ----------------------------------------
// Normalize helper
// ----------------------------------------
function normalize(name) {
    if (!name) return "";
    return name.trim().toLowerCase().replace(/ /g, "_");
}

// ----------------------------------------
// CLASS DROPDOWN (Load class.json details)
// ----------------------------------------
document.getElementById("class-dropdown").addEventListener("change", async function () {

    selectedClass = this.value;
    let classDetailsBox = document.getElementById("class-details-box");

    if (!selectedClass) {
        classDetailsBox.innerText = "Select a class to view details...";
        return;
    }

    try {
        const res = await fetch("/static/class.json");  // FIXED PATH
        const jsonData = await res.json();

        // Try all key forms
        const keyNorm = normalize(selectedClass);
        const keyLower = selectedClass.toLowerCase();

        let details =
            jsonData[selectedClass] ||
            jsonData[keyNorm] ||
            jsonData[keyLower];

        if (details) {
            classDetailsBox.innerText = JSON.stringify(details, null, 2);
        } else {
            classDetailsBox.innerText = "No details found for this class in class.json";
        }

    } catch (err) {
        classDetailsBox.innerText = "Failed to load class.json";
    }
});

// ----------------------------------------
// PREDICT BUTTON HANDLER
// ----------------------------------------
document.querySelectorAll(".predict-btn").forEach(btn => {
    btn.addEventListener("click", async function () {

        let fileInput = document.getElementById("file-input");

        if (!fileInput.files.length) {
            alert("Please upload an image first.");
            return;
        }
        if (!selectedClass) {
            alert("Please select a class first.");
            return;
        }

        const modelType = this.dataset.type;
        const origText = this.innerText;
        this.innerText = "Predicting...";
        this.disabled = true;

        let formData = new FormData();
        formData.append("file", fileInput.files[0]);
        formData.append("model_type", modelType);
        formData.append("selected_class", selectedClass);

        try {
            const res = await fetch("/predict", { method: "POST", body: formData });
            const data = await res.json();

            if (!data.success) {
                document.getElementById("prediction-output").innerHTML =
                    `<div style="color:red;">Error: ${data.error}</div>`;
                return;
            }

            const predLabel = data.predicted_label || "--";
            const modelUsed = data.model_used || "";
            const metrics = data.class_metrics || {};
            const nutrition = data.nutrition || {};

            // ------------------------------
            // METRICS
            // ------------------------------
            const accuracy = metrics.accuracy ?? "NA";
            const precision = metrics.precision ?? "NA";
            const recall = metrics.recall ?? "NA";

            let html = `
                <table class="pred-table">
                    <tr><th>Field</th><th>Value</th></tr>
                    <tr><td>Predicted Class</td><td>${predLabel}</td></tr>
                    <tr><td>Selected Class</td><td>${data.selected_class}</td></tr>
                    <tr><td>Model Used</td><td>${modelUsed}</td></tr>
                    <tr><td>Confidence</td><td>${(data.confidence * 100).toFixed(2)}%</td></tr>
                    <tr><td>Accuracy</td><td>${accuracy}</td></tr>
                    <tr><td>Precision</td><td>${precision}</td></tr>
                    <tr><td>Recall</td><td>${recall}</td></tr>
                </table>
            `;

            // ------------------------------
            // NUTRITION
            // ------------------------------
            if (Object.keys(nutrition).length > 0) {
                html += `
                    <div style="margin-top:8px;">
                        <strong>Nutrition:</strong><br>
                        Protein: ${nutrition.protein ?? "NA"}<br>
                        Fiber: ${nutrition.fiber ?? "NA"}<br>
                        Calories: ${nutrition.calories ?? "NA"}<br>
                        Carbohydrates: ${nutrition.carbohydrates ?? "NA"}<br>
                        Fat: ${nutrition.fat ?? "NA"}
                    </div>
                `;
            }

            // ------------------------------
            // CONFUSION MATRIX (table)
            // ------------------------------
            const conf = data.confusion_matrix_full;
            const labels = data.model_labels_order || [];

            if (conf && Array.isArray(conf)) {
                const n = conf.length;

                html += `
                    <div style="margin-top:15px;">
                        <strong>Confusion Matrix (rows = actual, cols = predicted)</strong>
                        <table class="confusion-table">
                            <thead>
                                <tr>
                                    <th></th>
                `;

                const headerLabels =
                    labels.length === n ? labels : [...Array(n)].map((_, i) => "C" + (i + 1));

                for (let h of headerLabels) html += `<th>${h}</th>`;
                html += `</tr></thead><tbody>`;

                for (let i = 0; i < n; i++) {
                    html += `<tr><th>${headerLabels[i]}</th>`;
                    for (let j = 0; j < n; j++) html += `<td>${conf[i][j]}</td>`;
                    html += `</tr>`;
                }

                html += `</tbody></table></div>`;
            }

            document.getElementById("prediction-output").innerHTML = html;

        } catch (err) {
            document.getElementById("prediction-output").innerHTML =
                `<div style="color:red;">Error: ${err}</div>`;
        }

        this.innerText = origText;
        this.disabled = false;
    });
});

function toggleTheme() {
    document.body.classList.toggle("dark");
}
