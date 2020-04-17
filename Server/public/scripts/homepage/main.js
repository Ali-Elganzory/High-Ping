window.onload = function () {
  var canvas = document.getElementById("canvas"),
    ctx = canvas.getContext("2d"),
    width = (canvas.width = window.innerWidth),
    height = (canvas.height = window.innerHeight);

  var sun = particle.create(width / 2, height / 2, 0, 0),
    planet = particle.create(width / 2 + 200, height / 2, 10, -Math.PI / 2);

  sun.mass = 30000;

  update();

  function update() {
    ctx.clearRect(0, 0, width, height);

    planet.gravitateTo(sun);
    planet.update();

    ctx.beginPath();
    ctx.fillStyle = "#ffff00";
    ctx.arc(sun.x, sun.y, 20, 0, 2 * Math.PI);
    ctx.fill();

    ctx.beginPath();
    ctx.fillStyle = "#0000ff";
    ctx.arc(planet.x, planet.y, 5, 0, 2 * Math.PI);
    ctx.fill();

    requestAnimationFrame(update);
  }
};
