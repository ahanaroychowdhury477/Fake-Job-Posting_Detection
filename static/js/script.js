document.getElementById("predictForm").addEventListener("submit", async function(e) {

    e.preventDefault();

    const formData = new FormData(this);
    document.getElementById("loading").style.display = "block";
document.getElementById("result").innerHTML = "";

    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    document.getElementById("loading").style.display = "none";

    const result = document.getElementById("result");

    if (data.status === "warning") {
        result.innerHTML = `<h3 style="color:red;">${data.message}</h3>`;
    } else {

        result.innerHTML = `
        <div class="result-card ${data.prediction === "Fake Job Posting" ? "fake" : "real"}">

            <h2>${data.prediction}</h2>

            <div class="circle">
    <svg>
        <circle cx="70" cy="70" r="60"></circle>
        <circle
            class="progress-circle"
            cx="70"
            cy="70"
            r="60"
            style="stroke-dashoffset:${377-(377*data.confidence/100)}">
        </circle>
    </svg>

    <div class="number">
        ${data.confidence}%
    </div>
</div>
            <p>
${data.prediction === "Fake Job Posting"
? "⚠️ This posting contains suspicious characteristics commonly found in fraudulent job advertisements."
: "✅ This posting appears to match patterns commonly found in genuine job advertisements."}
</p>

            <div class="progress">
                <div class="${data.prediction === 'Fake Job Posting' ? 'progress-circle fake-circle' : 'progress-circle real-circle'}"
                     style="width:${data.confidence}%">
                </div>
            </div>

        </div>
        `;
        
    }

});
document.querySelector("button[type='reset']").addEventListener("click", function () {
    document.getElementById("result").innerHTML = "";
    document.getElementById("loading").style.display = "none";
});