let selectedClass = null;

// ----------------------------------------
// Normalize helper
// ----------------------------------------
function normalize(name) {
    if (!name) return "";
    return name.trim().toLowerCase().replace(/ /g, "_");
}

// ----------------------------------------
// CLASS DROPDOWN
// ----------------------------------------
document.getElementById("class-dropdown").addEventListener("change", async function () {
    selectedClass = this.value;
    let box = document.getElementById("class-details-box");

    if (!box) return;  // safe check

    if (!selectedClass) {
        box.innerText = "Select a class to view details...";
        return;
    }

    try {
        const res = await fetch("/static/class.json");
        const jsonData = await res.json();

        const keyNorm = normalize(selectedClass);
        const keyLower = selectedClass.toLowerCase();

        let info =
            jsonData[selectedClass] ||
            jsonData[keyNorm] ||
            jsonData[keyLower];

        if (info) {
            box.innerText = JSON.stringify(info, null, 2);
        } else {
            box.innerText = "No details found in class.json";
        }

    } catch (err) {
        box.innerText = "Failed to load class.json";
    }
});

// ----------------------------------------
// PREDICTION
// ----------------------------------------
document.querySelectorAll(".predict-btn").forEach(btn => {
    btn.addEventListener("click", async function () {

        let fileInput = document.getElementById("file-input");
        const outputBox = document.getElementById("prediction-output");

        if (!fileInput.files.length) {
            alert("Please upload an image.");
            return;
        }
        if (!selectedClass) {
            alert("Please select a class.");
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

            this.innerText = origText;
            this.disabled = false;

            if (!data.success) {
                outputBox.innerHTML = `<div style="color:red;">Error: ${data.error}</div>`;
                return;
            }

            displayResults(data);

        } catch (err) {
            this.innerText = origText;
            this.disabled = false;

            outputBox.innerHTML = `<div style="color:red;">Error: ${err}</div>`;
        }

    });
});

// ----------------------------------------
// DISPLAY RESULTS
// ----------------------------------------
function displayResults(data) {

    const output = document.getElementById("prediction-output");

    const predicted = data.predicted_label || "--";
    const selected = data.selected_class || "--";
    const modelUsed = data.model_used || "--";
    const conf = data.confidence ?? 0;

    // Metrics
    const accuracy = data.accuracy ?? "NA";
    const precision = data.precision ?? "NA";
    const recall = data.recall ?? "NA";
    const f1 = data.f1_score ?? "NA";

    // Confusion Matrix
    const matrix = data.confusion_matrix_full || [];
    const labels = data.confusion_matrix_labels || [];

    // Percentage formatter
    const pct = v => (typeof v === "number" ? (v * 100).toFixed(2) + "%" : v);

    let html = `
        <table class="pred-table">
            <tr><th>Field</th><th>Value</th></tr>
            <tr><td>Predicted Class</td><td>${predicted}</td></tr>
            <tr><td>Selected Class</td><td>${selected}</td></tr>
            <tr><td>Model Used</td><td>${modelUsed}</td></tr>
            <tr><td>Confidence</td><td>${(conf * 100).toFixed(2)}%</td></tr>
            <tr><td>Accuracy</td><td>${pct(accuracy)}</td></tr>
            <tr><td>Precision</td><td>${pct(precision)}</td></tr>
            <tr><td>Recall</td><td>${pct(recall)}</td></tr>
            <tr><td>F1 Score</td><td>${pct(f1)}</td></tr>
        </table>
    `;

    // ------------------------------
    // CONFUSION MATRIX
    // ------------------------------
    if (matrix.length > 0) {

        const headerLabels = labels.length === matrix.length
            ? labels
            : [...Array(matrix.length)].map((_, i) => `C${i + 1}`);

        html += `
            <div style="margin-top:15px;">
                <strong>Confusion Matrix (rows = actual, cols = predicted)</strong>
                <table class="confusion-table">
                    <thead>
                        <tr><th></th>
        `;

        headerLabels.forEach(h => html += `<th>${h}</th>`);
        html += `</tr></thead><tbody>`;

        matrix.forEach((row, i) => {
            html += `<tr><th>${headerLabels[i]}</th>`;
            row.forEach(v => html += `<td>${v}</td>`);
            html += "</tr>";
        });

        html += "</tbody></table></div>";
    }

    output.innerHTML = html;
}


// ----------------------------------------
// CLASS DROPDOWN (LOAD NUTRITION DETAILS)
// ----------------------------------------
document.getElementById("class-dropdown").addEventListener("change", async function () {

    selectedClass = this.value;
    let box = document.getElementById("class-details-box");

    if (!selectedClass) {
        box.innerHTML = "Select a class to view details...";
        return;
    }

    try {
        const res = await fetch("/static/class.json");
        const jsonData = await res.json();

        const keyNorm = normalize(selectedClass);
        const keyLower = selectedClass.toLowerCase();

        const info =
            jsonData[selectedClass] ||
            jsonData[keyNorm] ||
            jsonData[keyLower];

        if (!info) {
            box.innerHTML = "No nutrition info available for this class.";
            return;
        }

        // Nutrition format
        box.innerHTML = `
            <strong>Nutrition Details</strong><br>
            <div class="nutri-line"><b>Calories:</b> ${info.calories ?? "NA"}</div>
            <div class="nutri-line"><b>Protein:</b> ${info.protein ?? "NA"}</div>
            <div class="nutri-line"><b>Fat:</b> ${info.fat ?? "NA"}</div>
            <div class="nutri-line"><b>Carbohydrates:</b> ${info.carbohydrates ?? "NA"}</div>
            <div class="nutri-line"><b>Fiber:</b> ${info.fiber ?? "NA"}</div>
        `;

    } catch (err) {
        console.log(err);
        box.innerHTML = "Failed to load class.json";
    }
});


// ----------------------------------------
// THEME SWITCHER
// ----------------------------------------
function toggleTheme() {
    document.body.classList.toggle("dark");
}
