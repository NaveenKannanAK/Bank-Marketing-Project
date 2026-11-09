
document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");

    if (form) {
        form.addEventListener("submit", function () {

            const button = form.querySelector(
                ".predict-button"
            );

            if (form.checkValidity()) {
                button.disabled = true;
                button.innerHTML = "⏳ Generating Prediction...";
            }
        });
    }

});
