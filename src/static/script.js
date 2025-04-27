// static/script.js
document.addEventListener("DOMContentLoaded", function () {
    const heading = document.querySelector("h1");
    if (heading) {
        heading.addEventListener("click", () => {
            alert("You clicked the heading!");
        });
    }
});
