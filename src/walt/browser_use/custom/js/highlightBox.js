(
  args = {
    left: 0,
    top: 0,
    width: 0,
    height: 0,
    color: "blue",
  }
) => {
  const { left, top, width, height, color } = args;
  console.log(left, top, width);
  const overlay = document.createElement("div");
  // overlay.style.all = "initial";
  overlay.style.position = "fixed";
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;
  overlay.style.top = `${top - 2}px`;
  overlay.style.left = `${left - 2}px`;
  overlay.style.color = `${color}`;
  overlay.style.opacity = 1;
  overlay.style.border = "2px solid transparent";
  overlay.style.borderRadius = "10px";
  overlay.style.boxShadow = `0 0 0px ${color}`;
  overlay.style.pointerEvents = "none";
  overlay.style.zIndex = 2147483646;
  document.body.appendChild(overlay);

  //   overlay.setAttribute(
  //     "style",
  //     `
  //     all: initial;
  //     position: fixed;
  //     border: 2px solid transparent;  /* Start with transparent border */
  //     borderRadius: 10px;  /* Add rounded corners */
  //     boxShadow: 0 0 0px ${color};  /* Initial boxShadow with 0px spread */
  //     left: ${left - 2}px;  /* Adjust left position to accommodate initial shadow spread */
  //     top: ${top - 2}px;  /* Adjust top position likewise */
  //     width: ${width}px;
  //     height: ${height}px;
  //     z-index: 2147483646; /* Maximum value - 1 */
  //     pointerEvents: none; /* Ensure the overlay does not interfere with user interaction */
  // `
  //   );
  console.log(overlay);
  // // Animate the boxShadow to create a "wave" effect
  try {
    let spread = 0; // Initial spread radius of the boxShadow
    const waveInterval = setInterval(() => {
      {
        spread += 10; // Increase the spread radius to simulate the wave moving outward
        overlay.style.boxShadow = `0 0 40px ${spread}px ${color}`; // Update boxShadow to new spread radius
        overlay.style.opacity = 1 - spread / 38; // Gradually decrease opacity to fade out the wave
        if (spread >= 38) {
          {
            // Assuming 76px ~ 2cm spread radius
            clearInterval(waveInterval); // Stop the animation once the spread radius reaches 2cm
            document.body.removeChild(overlay); // Remove the overlay from the document
          }
        }
      }
    }, 200);
  } catch (error) {
    console.log(error);
  }
};
