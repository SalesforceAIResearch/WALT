(
  args = {
    targetX: 0,
    targetY: 0,
    speed: 0,
  }
) => {
  const { targetX, targetY, speed } = args;
  // create cursor if needed
  try {
    if (!("browsergym_visual_cursor" in window)) {
      if (window.trustedTypes && window.trustedTypes.createPolicy) {
        window.trustedTypes.createPolicy("cursorPolicy", {
          createHTML: (string, sink) => string,
        });
      }
      let cursor = document.createElement("div");
      cursor.setAttribute("id", "browsergym-visual-cursor");
      cursor.innerHTML = `
          <svg width="50px" height="50px" viewBox="213 106 713 706" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M213.333 106.667L426.667 853.333 512 512 853.333 426.667 213.333 106.667z" fill="blue"/>
          </svg>`;
      cursor.setAttribute(
        "style",
        `
          all: initial;
          position: fixed;
          opacity: 0.7; /* Slightly transparent */
          z-index: 2147483647; /* Maximum value */
          pointer-events: none; /* Ensures the SVG doesn't interfere with page interactions */
      `
      );

      // Calculate center position within the viewport
      const centerX = window.innerWidth / 2;
      const centerY = window.innerHeight / 2;

      cursor.style.left = `${centerX}px`;
      cursor.style.top = `${centerY}px`;

      // save cursor element
      window.browsergym_visual_cursor = cursor;
      window.browsergym_visual_cursor_n_owners = 0;
    }

    // recover cursor
    let cursor = window.browsergym_visual_cursor;

    // attach cursor to document
    document.body.appendChild(cursor);
    window.browsergym_visual_cursor_n_owners += 1;
    let x = parseFloat(cursor.style.left);
    let y = parseFloat(cursor.style.top);

    let dx = targetX - x;
    let dy = targetY - y;
    let dist = Math.hypot(dx, dy);
    let movement_time = (dist / speed) * 1000; // seconds to milliseconds
    let still_wait_time = 1000;

    // Adjust steps based on distance to keep movement speed consistent
    // 1 step per 10 pixels of distance, adjust as needed
    let steps = Math.max(1, Math.trunc(dist / 10));

    let step_dx = dx / steps;
    let step_dy = dy / steps;
    let step_dist = dist / steps;
    let step_wait_time = Math.max(10, movement_time / steps);

    let step = 0;
    let time_still = 0;
    const cursorInterval = setInterval(() => {
      // move cursor
      if (step < steps) {
        x += step_dx;
        y += step_dy;
        cursor.style.left = `${x}px`;
        cursor.style.top = `${y}px`;
      }
      // still cursor (wait a bit)
      else if (time_still < still_wait_time) {
        time_still += step_wait_time;
      }
      // stop and detach cursor
      else {
        clearInterval(cursorInterval);
        window.browsergym_visual_cursor_n_owners -= 1;
        if (window.browsergym_visual_cursor_n_owners <= 0) {
          document.body.removeChild(cursor);
        }
      }
      step += 1;
    }, step_wait_time);
  } catch (error) {
    console.log(error);
  }

  // return movement_time;
};
